#ifndef _SIM_COMM_H
#define _SIM_COMM_H

#include <stdint.h>

typedef struct slurm_msg slurm_msg_t;

extern slurm_msg_t * sim_request_msg;
extern slurm_msg_t * sim_response_msg;


// different variables and routines declarations for
// tricking communications within slurm
extern int (*sim_slurmctrld_pthread_create_ref)(pthread_t *newthread,
												const pthread_attr_t *attr,
												void *(*start_routine) (void *),
												void *arg,
												const char *id,
												const char *func,
												const char *sarg,
												const char *funccall,
												const char *filename,
												const char *note,
												const int line);

extern void (*sim_slurmctrld_cond_broadcast_ref)(pthread_cond_t * cond,
												 const char *scond,
												 const char *filename,
												 const int line,
												 const char *func);


extern void * (*sim_set_db_inx_thread_ref)(void *no_data);
extern void * (*sim_slurmdbd_agent_ref)(void *no_data);
extern void * (*sim_decay_thread_ref)(void *no_data);

extern uint64_t (*sim_backfill_agent_ref)(void);

extern int64_t sim_slurmdbd_agent_sleep_till;
extern int64_t sim_slurmdbd_agent_count;

extern void (*sim_slurmctld_req_ref)(slurm_msg_t *msg);

#endif