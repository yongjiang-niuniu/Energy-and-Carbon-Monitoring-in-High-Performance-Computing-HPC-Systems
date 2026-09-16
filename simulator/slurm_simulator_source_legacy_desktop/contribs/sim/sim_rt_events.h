#ifndef _SIM_RT_EVENT_H
#define _SIM_RT_EVENT_H

// track runtime events to identify safe time for time skipping

#define MAX_RT_EVENT_TYPES 4

typedef enum {
	GENERAL_SSIM_RT_EVENT=1,
	SUBMIT_JOB_SSIM_RT_EVENT,
	SCHED_SSIM_RT_EVENT,
	EPILOG_COMPLETE_SSIM_RT_EVENT
} slurm_sim_rt_event_t;

int _event_expect(slurm_sim_rt_event_t event_type, const char *s_event_type, const char *func, const char *filename, const int line);
#define event_expect(event_type) _event_expect(event_type, #event_type, __func__, __FILE__, __LINE__)


int _event_started(slurm_sim_rt_event_t event_type, const char *s_event_type, const char *func, const char *filename, const int line);
#define event_started(event_type) _event_started(event_type, #event_type, __func__, __FILE__, __LINE__)

int _event_ended(slurm_sim_rt_event_t event_type, const char *s_event_type, const char *func, const char *filename, const int line);
#define event_ended(event_type) _event_ended(event_type, #event_type, __func__, __FILE__, __LINE__)


//extern int64_t *rt_events;

#endif
