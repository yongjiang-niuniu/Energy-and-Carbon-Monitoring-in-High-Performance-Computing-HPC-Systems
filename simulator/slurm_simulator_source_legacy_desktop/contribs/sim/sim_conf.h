#ifndef _SIM_CONF_H
#define _SIM_CONF_H

#include <stdint.h>

/******************************************************************************
 * Simulator Configuration Parameters
 ******************************************************************************/

/* Slurm simulator configuration parameters */
typedef struct slurm_sim_conf {
	uint64_t	time_start;	/* initial starting time in usec will be overwritten by time from first job */
	uint64_t	time_stop;	/* final time when simulation should stop, 0-never stop, 1-stop after all jobs are done*/
	uint64_t    microseconds_before_first_job;
	double      clock_scaling;
	/* shared memory name, used to sync slurmdbd and slurmctrld, should be
	 * different if multiple simulation is running at same time */
	char *      shared_memory_name;
	char *      events_file;
	uint64_t    time_after_all_events_done; /* time after all is done in usec*/

	/* additional delay between first job submision, usec.
	 * It is essentially same as microseconds_before_first_job but introduced to have one to one match of
	 * microseconds_before_first_job with regular slurm runs and account for extra time simulator takes to spin-off*/
	int64_t    first_job_delay;
	int64_t    comp_job_delay; /* delay between job is complete and epilog complete, usec*/
	int64_t    timelimit_delay; /* delay between job reaching timelimit is complete and epilog complete, usec*/

} slurm_sim_conf_t;

/* simulator configuration */
extern slurm_sim_conf_t *slurm_sim_conf;

/* read simulator configuration file */
extern int read_sim_conf(void);

/* print simulator configuration */
extern int print_sim_conf(void);

#endif
