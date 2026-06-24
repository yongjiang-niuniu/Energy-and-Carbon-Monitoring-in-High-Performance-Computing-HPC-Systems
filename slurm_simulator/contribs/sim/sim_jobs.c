#include "slurm/slurm.h"

#include "../../src/common/xmalloc.h"
#include "../../src/common/xstring.h"

#include "../../src/common/log.h"

#include "../../src/slurmctld/slurmctld.h"

#include "../../contribs/sim/sim_time.h"
#include "../../contribs/sim/sim_conf.h"
#include "../../contribs/sim/sim_events.h"
#include "../../contribs/sim/sim_jobs.h"
#include "../../contribs/sim/sim.h"

pthread_mutex_t active_job_mutex = PTHREAD_MUTEX_INITIALIZER;

sim_job_t * sim_first_active_job = NULL;
sim_job_t * sim_last_active_job = NULL;

void sim_insert_active_job(sim_job_t * active_job)
{
	pthread_mutex_lock(&active_job_mutex);

	sim_job_t * previous_sim_job=NULL;
	sim_job_t * next_sim_job=sim_first_active_job;
	while(next_sim_job != NULL && next_sim_job->job_id < active_job->job_id) {
		previous_sim_job = next_sim_job;
		next_sim_job = next_sim_job->next_sim_job;
	}

	active_job->next_sim_job = next_sim_job;
	active_job->previous_sim_job = previous_sim_job;

	if(next_sim_job != NULL) {
		next_sim_job->previous_sim_job = active_job;
	} else {
		sim_last_active_job = active_job;
	}
	if(previous_sim_job != NULL) {
		previous_sim_job->next_sim_job = active_job;
	} else {
		sim_first_active_job = active_job;
	}

	pthread_mutex_unlock(&active_job_mutex);
}

/* return number of removed jobs */
extern int sim_remove_active_sim_job(uint32_t job_id)
{
	pthread_mutex_lock(&active_job_mutex);
	sim_job_t * m_sim_job=sim_first_active_job;
	while(m_sim_job != NULL) {
		if(m_sim_job->job_id == job_id) {

			if(m_sim_job->next_sim_job != NULL && m_sim_job->previous_sim_job != NULL) {
				// job in the middle
				m_sim_job->previous_sim_job->next_sim_job = m_sim_job->next_sim_job;
				m_sim_job->next_sim_job->previous_sim_job = m_sim_job->previous_sim_job;
			} else if(m_sim_job->next_sim_job == NULL && m_sim_job->previous_sim_job == NULL) {
				// the only job
				sim_first_active_job = NULL;
			} else if(m_sim_job->next_sim_job == NULL && m_sim_job->previous_sim_job != NULL) {
				// last active job
				m_sim_job->previous_sim_job->next_sim_job = NULL;
			} else if(m_sim_job->next_sim_job != NULL && m_sim_job->previous_sim_job == NULL) {
				// first active job
				m_sim_job->next_sim_job->previous_sim_job = NULL;
				sim_first_active_job = m_sim_job->next_sim_job;
			}
			xfree(m_sim_job);
			pthread_mutex_unlock(&active_job_mutex);
			return 1;
		}
		m_sim_job = m_sim_job->next_sim_job;
	}
	pthread_mutex_unlock(&active_job_mutex);
	return 0;
}
extern void sim_print_active_jobs()
{
	info("Simulation Active Jobs:");
	sim_job_t * next_sim_job=sim_first_active_job;
	while(next_sim_job != NULL) {
		info("job_sim_id %d --jid %d --sim-walltime %" PRId64, next_sim_job->job_sim_id, next_sim_job->job_id, next_sim_job->walltime);
		next_sim_job = next_sim_job->next_sim_job;
	}
}

extern sim_job_t * sim_insert_sim_active_job(sim_event_submit_batch_job_t* event_submit_batch_job)
{
	sim_job_t *active_job=xcalloc(1,sizeof(*active_job));
	active_job->job_id = event_submit_batch_job->job_id;
	active_job->job_sim_id = event_submit_batch_job->job_sim_id;
	active_job->walltime = event_submit_batch_job->wall_utime;
	active_job->submit_time = get_sim_utime();

	sim_insert_active_job(active_job);

	return active_job;
}

extern sim_job_t *sim_find_active_sim_job(uint32_t job_id)
{
	sim_job_t * next_sim_job=sim_first_active_job;
	while(next_sim_job != NULL) {
		if(next_sim_job->job_id == job_id) {
			return next_sim_job;
		}
		next_sim_job = next_sim_job->next_sim_job;
	}

	if(next_sim_job==NULL) {
		// look by job_sim_id
		job_record_t *job_ptr = find_job_record(job_id);
		uint32_t job_sim_id = get_job_sim_id(job_ptr->name);

		next_sim_job=sim_first_active_job;
		while(next_sim_job != NULL) {
			if(next_sim_job->job_sim_id == job_sim_id) {

				pthread_mutex_lock(&events_mutex);
				next_sim_job->job_id = job_id;
				pthread_mutex_unlock(&events_mutex);
				return next_sim_job;
			}
			next_sim_job = next_sim_job->next_sim_job;
		}

	}
	return NULL;
}

extern uint32_t get_job_sim_id(const char *job_name)
{
	if(xstrncmp(job_name, "jobid_", 6)!=0) {
		error("Set job names to jobid_<integer>!");
		exit(1);
	}
	return atoi(xstrchr(job_name,'_')+1);
}
