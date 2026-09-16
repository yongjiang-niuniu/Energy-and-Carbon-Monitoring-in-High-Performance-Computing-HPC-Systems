#ifndef _SIM_JOBS_H
#define _SIM_JOBS_H

#include <stdint.h>

/******************************************************************************
 * Active Simulated Jobs
 ******************************************************************************/
/* sim_job contain information needed during job being in queue or running */
typedef struct sim_job sim_job_t;
typedef struct sim_job {
	int64_t walltime; /*job duration, INT64_MAX or any large value would results in job running till time limit, usec*/
	uint32_t job_id;	/* job ID */
	uint32_t job_sim_id;	/* job sim ID */
	int64_t submit_time; /* submit_time in usec*/
	int64_t start_time; /* start_time in usec*/
	int comp_job; /*job is complete and epilog is scheduled*/
	int requested_kill_timelimit; /* received REQUEST_KILL_TIMELIMIT */

	sim_job_t *next_sim_job;
	sim_job_t *previous_sim_job;
} sim_job_t;

extern pthread_mutex_t active_job_mutex;

extern sim_job_t * sim_first_active_job;
extern sim_job_t * sim_last_active_job;

extern sim_job_t * sim_insert_sim_active_job(sim_event_submit_batch_job_t* event_submit_batch_job);
extern int sim_remove_active_sim_job(uint32_t job_id);
extern sim_job_t *sim_find_active_sim_job(uint32_t job_id);
extern void sim_print_active_jobs();

extern uint32_t get_job_sim_id(const char *job_name);

#endif

