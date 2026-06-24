typedef struct agent_arg agent_arg_t;

#define agent_queue_request slurmctld_agent_queue_request
#include "../../src/slurmctld/agent.c"
#undef agent_queue_request


#include "../../contribs/sim/sim_time.h"
#include "../../contribs/sim/sim_conf.h"
#include "../../contribs/sim/sim_events.h"
#include "../../contribs/sim/sim_jobs.h"
#include "../../contribs/sim/sim.h"

extern void sim_complete_job(uint32_t job_id);


extern bool sim_job_epilog_complete(uint32_t job_id, char *node_name,
                                    uint32_t return_code)
{
    bool status;
    status = job_epilog_complete(job_id, node_name, return_code);
    if(status) {
        run_scheduler = true;
        debug2("%s: job_id=%d job_epilog_complete on is_kill_msg run_scheduler=true",
               __func__, job_id);
    }
    return status;
}

// this wrap actual agent_queue_request
// handle faken requests
void agent_queue_request(agent_arg_t *agent_arg_ptr)
{
	bool call_slurmctld_agent_queue_request=true;
	kill_job_msg_t * kill_job;
	batch_job_launch_msg_t *launch_msg_ptr;
//	job_record_t *job_ptr;
//	time_t now;
	//queued_request_t *queued_req_ptr = NULL;
	//__real_agent_queue_request(agent_arg_ptr);
	//return;

	debug("Sim: __wrap_agent_queue_request msg_type=%s", rpc_num2string(agent_arg_ptr->msg_type));
	//__real_agent_queue_request(agent_arg_ptr);

	switch(agent_arg_ptr->msg_type) {
	case REQUEST_BATCH_JOB_LAUNCH:
		launch_msg_ptr = (batch_job_launch_msg_t *)agent_arg_ptr->msg_args;
		sim_insert_event_comp_job(launch_msg_ptr->job_id);
		call_slurmctld_agent_queue_request=false;
		break;
	case REQUEST_KILL_TIMELIMIT:
		kill_job = (kill_job_msg_t*)agent_arg_ptr->msg_args;
		// Previously commented: complete_job(kill_job->job_id);
		sim_job_requested_kill_timelimit(kill_job->step_id.job_id);
		call_slurmctld_agent_queue_request=false;
		break;
	case REQUEST_TERMINATE_JOB:
		// this initiated from job_compleate by jobs finishing by themselves
		// can be called again if there are problems
		kill_job = (kill_job_msg_t*)agent_arg_ptr->msg_args;
		sim_job_requested_kill_timelimit(kill_job->step_id.job_id);
		call_slurmctld_agent_queue_request=false;
		break;
	case REQUEST_NODE_REGISTRATION_STATUS:
		debug("Sim: __wrap_agent_queue_request msg_type=%s", rpc_num2string(agent_arg_ptr->msg_type));
		call_slurmctld_agent_queue_request=false;
		break;
	case REQUEST_PING:
	case REQUEST_HEALTH_CHECK:
	case REQUEST_ACCT_GATHER_UPDATE:
		/*
		 * Periodic node ping/health/acct agent requests are meaningless
		 * in the simulator (there are no real slurmds to contact) and
		 * the real agent path aborts with "Not implemented agent
		 * request". Simply drop them.
		 */
		debug("Sim: dropping periodic agent request msg_type=%s", rpc_num2string(agent_arg_ptr->msg_type));
		call_slurmctld_agent_queue_request=false;
		break;
	case REQUEST_PARTITION_INFO:
		call_slurmctld_agent_queue_request=true;
		break;
	case REQUEST_NODE_INFO:
		call_slurmctld_agent_queue_request=true;
		break;
	case REQUEST_SUBMIT_BATCH_JOB:
		call_slurmctld_agent_queue_request=true;
		break;
	default:
		error("Sim: unknown for sim request will use normal slurm (msg_type=%s)", rpc_num2string(agent_arg_ptr->msg_type));
		call_slurmctld_agent_queue_request=true;
		break;
	}

	if(call_slurmctld_agent_queue_request) {
		slurmctld_agent_queue_request(agent_arg_ptr);
	}
}

void sim_notify_slurmctld_nodes()
{
    if (run_scheduler) {
        run_scheduler = false;
        /* below functions all have their own locking */
        queue_job_scheduler();
    }
}
