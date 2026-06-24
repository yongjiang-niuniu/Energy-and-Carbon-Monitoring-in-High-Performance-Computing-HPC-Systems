#ifndef _SIM_TIME_H
#define _SIM_TIME_H

#include <stdint.h>

/* actual sleeping in sleep functions between sim time check */
extern int64_t real_sleep_usec;

/* process creation real time in microseconds */
extern int64_t process_create_time_real;
/* process creation simulated time  in microseconds */
extern int64_t process_create_time_sim;

/* return real time in microseconds */
extern int64_t get_real_utime();
/* return simulated time in microseconds */
extern int64_t get_sim_utime();

/* return process create time in microseconds */
extern int64_t get_process_create_time();

extern void set_sim_time_and_scale(int64_t cur_sim_time, double scale);
extern void set_sim_time_scale(double scale);
extern void set_sim_time(int64_t cur_sim_time);


extern int64_t sim_main_thread_sleep_till;
extern int64_t sim_sched_thread_cond_wait_till;
extern int64_t sim_plugin_backfill_thread_sleep_till;
extern int64_t sim_thread_priority_multifactor_sleep_till;
extern int64_t sim_agent_init_sleep_till;

extern void iso8601_from_utime(char **buf, uint64_t utime, bool msec);

/* initialize simulation time */
extern void init_sim_time(uint64_t start_time, double scale, int set_time, int set_time_to_real);

/* time scaling model */
extern void sim_backfill_step_scale(uint64_t start_sim_utime,uint64_t start_real_utime,int n);
extern void sim_backfill_scale(uint64_t start_sim_utime,uint64_t start_real_utime,int n);

#endif
