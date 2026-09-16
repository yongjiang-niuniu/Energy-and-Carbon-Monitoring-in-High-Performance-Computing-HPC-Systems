#ifndef _SIM_H
#define _SIM_H


extern int64_t simulator_start_time;
extern int64_t sim_constructor_start_time;

/*threads*/
extern pthread_t sim_main_thread;
extern pthread_t sim_sched_thread;
extern pthread_t sim_plugin_backfill_thread;
extern pthread_t sim_thread_priority_multifactor;
extern pthread_t sim_agent_init;

//utils
int endswith(const char* withwhat, const char* what);

#endif
