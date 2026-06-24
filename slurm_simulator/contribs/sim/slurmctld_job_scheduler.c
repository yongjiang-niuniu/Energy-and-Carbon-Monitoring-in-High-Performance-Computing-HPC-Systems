//#define agent_queue_request slurmctld_agent_queue_request
#include "../../src/slurmctld/job_scheduler.c"
//#undef agent_queue_request


#include "../../contribs/sim/sim_time.h"
#include "../../contribs/sim/sim_conf.h"
#include "../../contribs/sim/sim_events.h"
#include "../../contribs/sim/sim_jobs.h"
#include "../../contribs/sim/sim.h"
#include "../../contribs/sim/slurmctld_sim.h"

extern int get_sched_requests()
{
	return sched_requests;
}

/* simulate a single loop of _sched_agent
 * return true if run scheduler*/
extern bool sim_sched_agent_loop(int64_t now64) {
	long delta_t;
	struct timeval now;
	int job_cnt;
	bool full_queue;

	static bool first_run = true;

	if (slurmctld_config.shutdown_time) {
		return SLURM_SUCCESS;
	}
	if (first_run) {
		//slurm_mutex_lock(&sched_mutex);
		sim_sched_thread_cond_wait_till = 0;
		first_run = false;
	}

	if (slurmctld_config.shutdown_time) {
		//slurm_mutex_unlock(&sched_mutex);
		return false;
	}
	if (!sched_requests) {
		return false;
	}
	if (sim_sched_thread_cond_wait_till > now64) {
		return false;
	}

	gettimeofday(&now, NULL);
	delta_t = (now.tv_sec - sched_last.tv_sec) * USEC_IN_SEC;
	delta_t += now.tv_usec - sched_last.tv_usec;



	if (sched_requests && delta_t > sched_min_interval) {
		full_queue = sched_full_queue;
		sched_full_queue = false;
		sched_requests = 0;
		//slurm_mutex_unlock(&sched_mutex);

		job_cnt = _schedule(full_queue);
		gettimeofday(&now, NULL);
		sched_last.tv_sec = now.tv_sec;
		sched_last.tv_usec = now.tv_usec;
		sim_sched_thread_cond_wait_till = (int64_t)10000000*sched_last.tv_sec + sched_last.tv_usec + sched_min_interval;
		if (job_cnt) {
			/* jobs were started, save state */
			schedule_node_save();        /* Has own locking */
			schedule_job_save();        /* Has own locking */
		}
		return true;
	}

	return false;
}
