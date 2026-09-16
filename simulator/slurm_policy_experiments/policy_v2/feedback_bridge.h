/* Experimental simulator-only bridge. Disabled unless POLICY_V2_CONFIG is set. */
#include <sys/socket.h>
#include <sys/wait.h>
#include <poll.h>
#include <unistd.h>
#include <errno.h>
extern double *sim_timeval_scale;
#define V2_MAX_JOBS 4096
typedef struct {
    sim_event_submit_batch_job_t *payload;
    int64_t arrived;
    int submitted;
    int reported;
    int state;
} v2_job;
static v2_job v2_jobs[V2_MAX_JOBS];
static int v2_count=0, v2_fd=-1, v2_failed=0, v2_busy=0;
static int64_t v2_wake=INT64_MAX;
static FILE *v2_log=NULL;

static int64_t v2_epoch(void) {
    return simulator_start_time + slurm_sim_conf->microseconds_before_first_job + slurm_sim_conf->first_job_delay;
}
static int v2_pending(void) {
    int n=0;
    for(int i=0;i<v2_count;i++) if(!v2_jobs[i].submitted) n++;
    return n;
}
static void v2_release(v2_job *job) {
    job->submitted=1;
    submit_job(job->payload);
    if(v2_log) {
        fprintf(v2_log,"{\"event\":\"release\",\"id\":%u,\"slurm_id\":%u,\"release_s\":%.6f,\"arrived_s\":%.6f}\n",
                job->payload->job_sim_id,job->payload->job_id,
                (get_sim_utime()-v2_epoch())/1000000.0,(job->arrived-v2_epoch())/1000000.0);
        fflush(v2_log);
    }
}
static void v2_fail(const char *reason) {
    v2_failed=1;
    if(v2_fd>=0) { close(v2_fd); v2_fd=-1; }
    error("POLICY_V2_FAILSAFE: %s",reason);
    if(v2_log) {fprintf(v2_log,"{\"event\":\"failsafe\",\"reason\":\"%s\"}\n",reason);fflush(v2_log);}
}
static void v2_arrive(sim_event_submit_batch_job_t *payload,int64_t when) {
    if(!getenv("POLICY_V2_CONFIG")) {submit_job(payload);return;}
    if(v2_count>=V2_MAX_JOBS) {v2_fail("bridge_capacity_exceeded");submit_job(payload);return;}
    v2_jobs[v2_count++]=(v2_job){.payload=payload,.arrived=when,.state=-1};
    v2_wake=0;
}
static int v2_connect(void) {
    int sockets[2];
    if(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)) return -1;
    pid_t child=fork();
    if(child<0) {close(sockets[0]);close(sockets[1]);return -1;}
    if(child==0) {
        close(sockets[0]);
        dup2(sockets[1],STDIN_FILENO);dup2(sockets[1],STDOUT_FILENO);close(sockets[1]);
        execlp("python3","python3","/opt/policy-v2/feedback.py",getenv("POLICY_V2_CONFIG"),(char*)NULL);
        _exit(127);
    }
    close(sockets[1]);v2_fd=sockets[0];return 0;
}
static void v2_tick(int64_t now) {
    if(!getenv("POLICY_V2_CONFIG") || !v2_count || v2_busy) return;
    v2_busy=1;
    if(!v2_log) v2_log=fopen("/opt/slurm-sim/etc/bridge.jsonl","a");
    int dirty=0;
    for(int i=0;i<v2_count;i++) {
        v2_job *j=&v2_jobs[i];
        if(!j->submitted) {if(!j->reported) dirty=1;continue;}
        job_record_t *actual=find_job_record(j->payload->job_id);
        int state=actual ? (IS_JOB_RUNNING(actual)?2:(IS_JOB_PENDING(actual)?1:3)) : 3;
        if(state!=j->state) dirty=1;
        j->state=state;
    }
    if(!dirty && (!v2_pending() || now<v2_wake)) {v2_busy=0;return;}
    if(v2_failed) goto fallback;
    /* Controller wall time must not become artificial simulated waiting time. */
    double scale=*sim_timeval_scale;
    set_sim_time_scale(0);
    now=get_sim_utime();
    if(v2_fd<0 && v2_connect()) {set_sim_time_scale(scale);v2_fail("controller_start");goto fallback;}
    char *message=NULL;
    size_t length=0;
    FILE *buffer=open_memstream(&message,&length);
    fprintf(buffer,"{\"now_s\":%.6f,\"arrivals\":[",(now-v2_epoch())/1000000.0);
    int comma=0;
    for(int i=0;i<v2_count;i++) if(!v2_jobs[i].reported) {
        v2_job *j=&v2_jobs[i];
        fprintf(buffer,"%s[\"sim_%06u\",%.6f]",comma++?",":"",j->payload->job_sim_id,(j->arrived-v2_epoch())/1000000.0);
    }
    fprintf(buffer,"],\"states\":[");comma=0;
    for(int i=0;i<v2_count;i++) if(v2_jobs[i].submitted) {
        v2_job *j=&v2_jobs[i];
        job_record_t *actual=find_job_record(j->payload->job_id);
        fprintf(buffer,"%s{\"id\":\"sim_%06u\",\"state\":\"%s\",\"start_s\":%.6f}",comma++?",":"",
                j->payload->job_sim_id,j->state==1?"pending":(j->state==2?"running":"completed"),
                actual && actual->start_time ? (actual->start_time*1000000LL-v2_epoch())/1000000.0 : 0.0);
    }
    fprintf(buffer,"]}\n");fclose(buffer);
    /* Preserve the exact observations passed to the policy for audit. */
    FILE *observations=fopen("/opt/slurm-sim/etc/observations.jsonl","a");
    if(observations) {fwrite(message,1,length,observations);fclose(observations);}
    size_t sent=0;
    while(sent<length) {
        struct pollfd p={.fd=v2_fd,.events=POLLOUT};
        if(poll(&p,1,5000)<=0) break;
        ssize_t n=send(v2_fd,message+sent,length-sent,MSG_NOSIGNAL);
        if(n<=0) break;
        sent+=n;
    }
    free(message);
    char response[65536];size_t nread=0;
    if(sent==length) while(nread<sizeof(response)-1) {
        struct pollfd p={.fd=v2_fd,.events=POLLIN};
        if(poll(&p,1,5000)<=0) break;
        ssize_t n=recv(v2_fd,response+nread,1,0);
        if(n!=1) break;
        if(response[nread++]=='\n') break;
    }
    response[nread]='\0';
    set_sim_time_scale(scale);
    if(!nread || response[nread-1]!='\n') {v2_fail("controller_timeout_or_exit");goto fallback;}
    char *cursor=response,*end;
    int64_t next=strtoll(cursor,&end,10);
    if(end==cursor) {v2_fail("invalid_next_wake");goto fallback;}
    cursor=end;
    long count=strtol(cursor,&end,10);
    if(end==cursor || count<0 || count>v2_pending()) {v2_fail("invalid_release_count");goto fallback;}
    cursor=end;
    int indices[V2_MAX_JOBS];
    for(int k=0;k<count;k++) {
        long id=strtol(cursor,&end,10);indices[k]=-1;
        if(end==cursor) {v2_fail("invalid_release_id");goto fallback;}
        cursor=end;
        for(int i=0;i<v2_count;i++) if(v2_jobs[i].payload->job_sim_id==id && !v2_jobs[i].submitted) indices[k]=i;
        for(int i=0;i<k;i++) if(indices[i]==indices[k]) indices[k]=-1;
        if(indices[k]<0) {v2_fail("unknown_or_duplicate_release");goto fallback;}
    }
    for(int i=0;i<v2_count;i++) v2_jobs[i].reported=1;
    v2_wake=next+v2_epoch();
    if(v2_wake<=now && count<v2_pending()) {v2_fail("invalid_past_wake");goto fallback;}
    for(int i=0;i<count;i++) v2_release(&v2_jobs[indices[i]]);
    v2_busy=0;return;
fallback:
    for(int i=0;i<v2_count;i++) if(!v2_jobs[i].submitted) v2_release(&v2_jobs[i]);
    v2_wake=INT64_MAX;v2_busy=0;
}
