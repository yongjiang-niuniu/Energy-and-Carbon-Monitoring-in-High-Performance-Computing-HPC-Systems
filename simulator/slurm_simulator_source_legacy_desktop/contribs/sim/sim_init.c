#include "slurm/slurm.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>

#include "../../src/common/log.h"
#include "../../src/common/xstring.h"
#include "../../src/common/xmalloc.h"


#include "../../contribs/sim/sim_time.h"
#include "../../contribs/sim/sim_conf.h"
#include "../../contribs/sim/sim_events.h"
#include "../../contribs/sim/sim_jobs.h"
#include "../../contribs/sim/sim_users.h"
#include "../../contribs/sim/sim.h"
#include "../../contribs/sim/sim_comm.h"

/* Shared Memory */

void         * sim_shmem_data = NULL;
int64_t *sim_timeval_shift = NULL;
double *sim_timeval_scale = NULL;

int64_t m_sim_timeval_shift = 0;
double m_sim_timeval_scale = 1.0;


int64_t simulator_start_time=0;

int64_t sim_constructor_start_time=0;


int64_t sim_slurmdbd_agent_sleep_till = 0;
int64_t sim_slurmdbd_agent_count = 0;

slurm_msg_t * sim_request_msg = NULL;;
slurm_msg_t * sim_response_msg = NULL;



void * (*sim_set_db_inx_thread_ref)(void *no_data) = NULL;
void * (*sim_decay_thread_ref)(void *no_data)=NULL;
void * (*sim_slurmdbd_agent_ref)(void *no_data)=NULL;

/* reference to sched_plugin */
uint64_t (*sim_backfill_agent_ref)(void)=NULL;

void (*sim_slurmctld_req_ref)(slurm_msg_t *msg)=NULL;

//extern void init_sim_time(uint32_t start_time, double scale, int set_time, int set_time_to_real);
//extern int sim_read_users(void);
//extern int sim_print_users(void);

static int shared_memory_size()
{
	return sizeof(*sim_timeval_shift) + sizeof(*sim_timeval_scale) + 16;
}


static int build_shared_memory()
{
	int fd;

	fd = shm_open(slurm_sim_conf->shared_memory_name, O_CREAT | O_RDWR, S_IRWXU | S_IRWXG | S_IRWXO);
	if (fd < 0) {
		int err = errno;
		error("Sim: Error opening %s -- %s", slurm_sim_conf->shared_memory_name,strerror(err));
		return -1;
	}

	if (ftruncate(fd, shared_memory_size())) {
		info("Sim: Warning!  Can not truncate shared memory segment.");
	}

	sim_shmem_data = mmap(0, shared_memory_size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

	if(!sim_shmem_data){
		debug("Sim: mapping %s file can not be done\n", slurm_sim_conf->shared_memory_name);
		return -1;
	}

	return 0;
}

/*
 * slurmd build shared memory (because it run first) and
 * Slurmctld attached to it
 */
extern int attach_shared_memory()
{
	int fd;
	int new_shared_memory=0;

	fd = shm_open(slurm_sim_conf->shared_memory_name, O_RDWR, S_IRWXU | S_IRWXG | S_IRWXO );
	if (fd >= 0) {
		if (ftruncate(fd, shared_memory_size())) {
			info("Sim: Warning! Can't truncate shared memory segment.");
		}
		sim_shmem_data = mmap(0, shared_memory_size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	} else {
		build_shared_memory();
		new_shared_memory=1;
	}

	if (!sim_shmem_data) {
		error("Sim: mapping %s file can not be done", slurm_sim_conf->shared_memory_name);
		return -1;
	}

	/* Initializing pointers to shared memory */
	int offset = 0;
	sim_timeval_shift  = sim_shmem_data + offset;
	offset += sizeof(*sim_timeval_shift);
	sim_timeval_scale  = sim_shmem_data + offset;
	offset += sizeof(*sim_timeval_scale);

	return new_shared_memory;
}


extern char *__progname;


/*
 * "Constructor" function to be called before the main of each Slurm
 * entity (e.g. slurmctld, slurmd and commands).
 */

void __attribute__ ((constructor)) sim_init(void)
{
	sim_constructor_start_time = get_real_utime();
	//info("Sim: Slurm simulator init (%s).", __progname);

	/*struct timespec ts;
	timespec_get(&ts, TIME_UTC);*/
	//time_t t = sim_constructor_start_time/1000000;
	//char buff[100];
	//strftime(buff, sizeof buff, "%D %T", gmtime(&t));
	//printf("Current time: %s.%09ld UTC\n", buff, sim_constructor_start_time%1000000);



	int set_time = 0;
	int set_time_to_real = 1;

	sim_timeval_shift = &m_sim_timeval_shift;
	sim_timeval_scale = &m_sim_timeval_scale;


	read_sim_conf();
    //print_sim_conf();

	sim_read_users();
	//sim_print_users();

	int new_shared_memory = attach_shared_memory();

	if (new_shared_memory < 0) {
		error("Error attaching/building shared memory and maping it");
		exit(1);
	};

	if(new_shared_memory==1) {
		set_time = 1;
	}
	if(xstrcmp(__progname, "slurmdbd") == 0) {
		set_time = 1;
	}

	if(slurm_sim_conf->time_start==0) {
		set_time_to_real = 1;
	} else {
		set_time_to_real = 0;
	}

	init_sim_time(slurm_sim_conf->time_start, 1.0,
			set_time, set_time_to_real);

	simulator_start_time = process_create_time_sim;


	sim_main_thread = pthread_self();


	char *outstr=NULL;
	xiso8601timecat(outstr, true);
	debug("time: %s %" PRId64 " %" PRId64, outstr, get_real_utime(), get_sim_utime());
	xfree(outstr);
}


/*

extern int sim_pthread_create (pthread_t *newthread,
		const pthread_attr_t *attr,
		void *(*start_routine) (void *),
		void *arg,
		const char *id,
		const char *func)
{
	//slurmctld: debug:  id: '&thread_id_event_thread'
	//slurmctld: debug:  func: 'sim_events_thread'
	//slurmctld: debug:  id: '&backfill_thread'
	//slurmctld: debug:  func: 'backfill_agent
	// @TODO check that 'id' do not change and they are unique across slurm
	if (strcmp("&slurmctld_config.thread_id_rpc", id) == 0) {
		debug("Sim: thread_id_rpc.");
	} else if (strcmp("&slurmctld_config.thread_id_sig", id) == 0) {
		debug("Sim: thread_id_sig ... skip.");
		return 0;
	} else if (strcmp("&slurmctld_config.thread_id_save", id) == 0) {
		debug("Sim: thread_id_save ... ");
		//return 0;
	} else if (strcmp("&slurmctld_config.thread_id_power", id) == 0) {
		debug("Sim: thread_id_power ... skip.");
		return 0;
	} else if (strcmp("&slurmctld_config.thread_id_purge_files", id) == 0) {
		debug("Sim: thread_id_purge_files ... skip.");
		return 0;
	}
	//debug("id: '%s'", id);
	//debug("func: '%s'", func);

	int err = pthread_create(newthread, attr, start_routine, arg);

	if (strcmp("&backfill_thread", id) == 0) {
		debug("backfill_thread");
		sim_plugin_sched_thread=*newthread;
		sim_plugin_sched_thread_isset = 1;
	} else if (strcmp("&builtin_thread", id) == 0) {
		debug("builtin_thread");
		sim_plugin_sched_thread = *newthread;
		sim_plugin_sched_thread_isset = 1;
	}
	return err;
}*/


int endswith(const char* withwhat, const char* what)
{
	if(withwhat==NULL && what==NULL)
		return 1;
	if(withwhat==NULL)
		return 0;
    int l1 = strlen(withwhat);
    int l2 = strlen(what);
    if (l1 > l2)
        return 0;

    return strcmp(withwhat, what + (l2 - l1)) == 0;
}
