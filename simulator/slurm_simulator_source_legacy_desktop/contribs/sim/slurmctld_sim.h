#ifndef _SLURMCTRLD_TIME_H
#define _SLURMCTRLD_TIME_H

/* functions declarations used in simulated slurm controller */
#include <stdint.h>

/* simulate a single loop of _sched_agent
 * return true if run scheduler*/
extern bool sim_sched_agent_loop(int64_t now64);


extern void sim_schedule_plugin_run_once();

extern void sim_mini_loop();

#endif