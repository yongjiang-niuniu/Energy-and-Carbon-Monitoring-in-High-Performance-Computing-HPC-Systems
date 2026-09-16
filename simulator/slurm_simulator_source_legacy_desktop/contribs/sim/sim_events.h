#ifndef _SIM_EVENTS_H
#define _SIM_EVENTS_H

#include <stdint.h>

/******************************************************************************
 * Simulation Events
 ******************************************************************************/
typedef enum {
	SIM_TIME_ZERO = 1001,
	SIM_TIME_INF,
	SIM_NODE_REGISTRATION,
	SIM_SUBMIT_BATCH_JOB,
	SIM_COMPLETE_BATCH_SCRIPT,
	SIM_EPILOG_COMPLETE,
	SIM_CANCEL_JOB,
	//SIM_RUN_BACKFILL_SCHEDULER,
	SIM_ACCOUNTING_UPDATE,
	SIM_PRIORITY_DECAY,
	SIM_SET_DB_INDEX,
} sim_event_type_t;

typedef struct sim_event_submit_batch_job {
	int64_t wall_utime; /*actual walltime*/
	uint32_t job_id;	/* job ID */
	uint32_t job_sim_id;
	char **argv;
	int argc;
} sim_event_submit_batch_job_t;

typedef struct sim_event {
	int64_t when; /* time of event in usec*/
	struct sim_event *next;
	struct sim_event *previous;
	sim_event_type_t type; /* event type */
	void *payload; /* event type */
} sim_event_t;

extern int sim_n_noncyclic_events;
extern int sim_n_cyclic_events;
extern sim_event_t * sim_next_event;
extern sim_event_t * sim_first_event;
extern sim_event_t * sim_last_event;

extern void sim_init_events();
extern void sim_print_events();
extern void sim_print_event(sim_event_t * event);

extern void sim_insert_event(int64_t when, int type, void *payload);
extern void sim_insert_event_comp_job(uint32_t job_id);
extern void sim_insert_event_rpc_epilog_complete(uint32_t job_id);
extern void sim_job_requested_kill_timelimit(uint32_t job_id);
extern sim_event_t * sim_pop_next_event();

extern pthread_mutex_t events_mutex;




#endif

