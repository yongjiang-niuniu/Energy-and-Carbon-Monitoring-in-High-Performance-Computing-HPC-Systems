
// include slurm.h first to ignore include in sbatch.c
#include "slurm/slurm.h"



#define main sbatch_main
//#define slurm_submit_batch_job wrap_slurm_submit_batch_job
//extern int wrap_slurm_submit_batch_job(job_desc_msg_t *req,
//				  submit_response_msg_t **resp);
//#define slurm_conf_init(...) {};
//#define log_init(...) {};
#include "../../src/sbatch/sbatch.c"
#undef main
//#include "../../src/sbatch/opt.c"
//#include "../sbatch/xlate.c"

#include "../../contribs/sim/sim_time.h"
#include "../../contribs/sim/sim_conf.h"
#include "../../contribs/sim/sim_events.h"
#include "../../contribs/sim/sim_rt_events.h"
#include "../../contribs/sim/sim_jobs.h"
#include "../../contribs/sim/sim.h"
#include "../../src/slurmctld/proc_req.h"
//extern int wrap_slurm_submit_batch_job(job_desc_msg_t *req,
//				  submit_response_msg_t **resp) {
//	return 0;
//}



extern void submit_job(sim_event_submit_batch_job_t* event_submit_batch_job)
{
	/*
	 * got main function from sbatch and replaced all exit with return
	 *
	 * DONFORGET to add sim_insert_sim_active_job and walltime retirements (done)
	 */
	int argc=event_submit_batch_job->argc;
	char **argv=event_submit_batch_job->argv;

	//log_options_t logopt = LOG_OPTS_STDERR_ONLY;
	job_desc_msg_t *desc = NULL, *first_desc = NULL;
	submit_response_msg_t *resp = NULL;
	char *script_name;
	char *script_body=xstrdup("#!/bin/bash\nsleep 30\n");
	char **het_job_argv;
	int script_size = 0, het_job_argc, het_job_argc_off = 0, het_job_inx;
	int i, rc = SLURM_SUCCESS, retries = 0, het_job_limit = 0;
	bool het_job_fini = false;
	List job_env_list = NULL, job_req_list = NULL;
	sbatch_env_t *local_env = NULL;
	bool quiet = false;

	/* force line-buffered output on non-tty outputs */
	if (!isatty(STDOUT_FILENO))
		setvbuf(stdout, NULL, _IOLBF, 0);
	if (!isatty(STDERR_FILENO))
		setvbuf(stderr, NULL, _IOLBF, 0);

	//slurm_conf_init(NULL);
    slurm_init(NULL);
	//log_init(xbasename(argv[0]), logopt, 0, NULL);

	_set_exit_code();
	if (spank_init_allocator() < 0) {
		error("SBATCH: Failed to initialize plugin stack");
		return;
	}

	/* Be sure to call spank_fini when sbatch exits
	 */
	if (atexit((void (*) (void)) spank_fini) < 0)
		error("SBATCH: Failed to register atexit handler for plugins: %m");

	script_name = process_options_first_pass(argc, argv);

	/* Preserve quiet request which is lost in second pass */
	quiet = opt.quiet;

	/* reinit log with new verbosity (if changed by command line) */
//	if (opt.verbose || opt.quiet) {
//		logopt.stderr_level += opt.verbose;
//		logopt.stderr_level -= opt.quiet;
//		logopt.prefix_level = 1;
//		log_alter(logopt, 0, NULL);
//	}

//	if (sbopt.wrap != NULL) {
//		script_body = _script_wrap(sbopt.wrap);
//	} else {
//		script_body = _get_script_buffer(script_name, &script_size);
//	}
	if (script_body == NULL) {
		error("SBATCH: script_body is NULL");
		return;
	}

	het_job_argc = argc - opt.argc;
	het_job_argv = argv;
	for (het_job_inx = 0; !het_job_fini; het_job_inx++) {
		bool more_het_comps = false;
		init_envs(&het_job_env);
		process_options_second_pass(het_job_argc, het_job_argv,
					    &het_job_argc_off, het_job_inx,
					    &more_het_comps, script_name ?
					    xbasename (script_name) : "stdin",
					    script_body, script_size);
		if ((het_job_argc_off >= 0) &&
		    (het_job_argc_off < het_job_argc) &&
		    !xstrcmp(het_job_argv[het_job_argc_off], ":")) {
			/* het_job_argv[0] moves from "salloc" to ":" */
			het_job_argc -= het_job_argc_off;
			het_job_argv += het_job_argc_off;
		} else if (!more_het_comps) {
			het_job_fini = true;
		}

		/*
		 * Note that this handling here is different than in
		 * salloc/srun. Instead of sending the file contents as the
		 * burst_buffer field in job_desc_msg_t, it will be spliced
		 * in to the job script.
		 */
		if (opt.burst_buffer_file) {
			buf_t *buf = create_mmap_buf(opt.burst_buffer_file);
			if (!buf) {
				error("Invalid --bbf specification");
				exit(error_exit);
			}
            run_command_add_to_script(&script_body, get_buf_data(buf));
            FREE_NULL_BUFFER(buf);
		}

		if (spank_init_post_opt() < 0) {
			error("SBATCH: Plugin stack post-option processing failed");
			return;
		}

		if (opt.get_user_env_time < 0) {
			/* Moab doesn't propagate the user's resource limits, so
			 * slurmd determines the values at the same time that it
			 * gets the user's default environment variables. */
			(void) _set_rlimit_env();
		}

		/*
		 * if the environment is coming from a file, the
		 * environment at execution startup, must be unset.
		 */
		if (sbopt.export_file != NULL)
			env_unset_environment();

		_set_prio_process_env();
		_set_spank_env();
		_set_submit_dir_env();
		_set_umask_env();
		if (local_env && !job_env_list) {
			job_env_list = list_create(NULL);
			list_append(job_env_list, local_env);
			job_req_list = list_create(NULL);
			list_append(job_req_list, desc);
		}
		local_env = xmalloc(sizeof(sbatch_env_t));
		memcpy(local_env, &het_job_env, sizeof(sbatch_env_t));

		desc = slurm_opt_create_job_desc(&opt, true);
		if (_fill_job_desc_from_opts(desc) == -1)
			exit(error_exit);
		if (!first_desc)
			first_desc = desc;
		if (het_job_inx || !het_job_fini) {
			set_env_from_opts(&opt, &first_desc->environment,
					  het_job_inx);
		} else
			set_env_from_opts(&opt, &first_desc->environment, -1);
		if (!job_req_list) {
			desc->script = (char *) script_body;
		} else {
			list_append(job_env_list, local_env);
			list_append(job_req_list, desc);
		}
	}
	het_job_limit = het_job_inx;
	if (!desc) {	/* For CLANG false positive */
		error("SBATCH: Internal parsing error");
		return;
	}

	if (job_env_list) {
		ListIterator desc_iter, env_iter;
		i = 0;
		desc_iter = list_iterator_create(job_req_list);
		env_iter  = list_iterator_create(job_env_list);
		desc      = list_next(desc_iter);
		while (desc && (local_env = list_next(env_iter))) {
			set_envs(&desc->environment, local_env, i++);
			desc->env_size = envcount(desc->environment);
		}
		list_iterator_destroy(env_iter);
		list_iterator_destroy(desc_iter);

	} else {
		set_envs(&desc->environment, &het_job_env, -1);
		desc->env_size = envcount(desc->environment);
	}
	if (!desc) {	/* For CLANG false positive */
		error("SBATCH: Internal parsing error");
		return;
	}

	/*
	 * If can run on multiple clusters find the earliest run time
	 * and run it there
	 */
	if (opt.clusters) {
		if (job_req_list) {
			rc = slurmdb_get_first_het_job_cluster(job_req_list,
					opt.clusters, &working_cluster_rec);
		} else {
			rc = slurmdb_get_first_avail_cluster(desc,
					opt.clusters, &working_cluster_rec);
		}
		if (rc != SLURM_SUCCESS) {
			print_db_notok(opt.clusters, 0);
			error("SBATCH: slurmdb_get_first_avail_cluster failed");
			return;
		}
	}

	if (sbopt.test_only) {
		if (job_req_list)
			rc = slurm_het_job_will_run(job_req_list);
		else
			rc = slurm_job_will_run(desc);

		if (rc != SLURM_SUCCESS) {
			slurm_perror("SBATCH: allocation failure");
			return;
		}
		return;
	}

	//Do not submit
	sim_job_t *active_job=sim_insert_sim_active_job(event_submit_batch_job);

	while (true) {
		static char *msg;
		if (job_req_list)
			rc = slurm_submit_batch_het_job(job_req_list, &resp);
		else {
			//rc = slurm_submit_batch_job(desc, &resp);

//			slurm_msg_t req_msg;
//			slurm_msg_t resp_msg;
//
//			slurm_msg_t_init(&req_msg);
//			slurm_msg_t_init(&resp_msg);
//
//			/*
//			 * set Node and session id for this request
//			 */
//			if (desc->alloc_sid == NO_VAL)
//				desc->alloc_sid = getsid(0);
//
//			req_msg.msg_type = REQUEST_SUBMIT_BATCH_JOB;
//			req_msg.data     = desc;
//			req_msg.conn     = NULL;
//			//if (req_msg.flags & SLURM_GLOBAL_AUTH_KEY) {
//			//	auth_cred = auth_g_create(req_msg.auth_index, _global_auth_key());
//			//} else {
//			req_msg.auth_cred = auth_g_create(req_msg.auth_index, slurm_conf.authinfo);
//			auth_g_verify(req_msg.auth_cred, slurm_conf.authinfo);
//			//}
//			req_msg.auth_uid = auth_g_get_uid(req_msg.auth_cred);
//			req_msg.auth_uid_set = true;
//			slurmctld_req(&req_msg);
			rc = slurm_submit_batch_job(desc, &resp);
		}
		if(resp != NULL) {
			// insert job to active simulated job list
			if (active_job->job_id == 0) {
				pthread_mutex_lock(&events_mutex);
				active_job->job_id = resp->job_id;
				event_submit_batch_job->job_id = resp->job_id;
				pthread_mutex_unlock(&events_mutex);
			}
		}
		if (rc >= 0)
			break;
		if (errno == ESLURM_ERROR_ON_DESC_TO_RECORD_COPY) {
			msg = "Slurm job queue full, sleeping and retrying";
		} else if (errno == ESLURM_NODES_BUSY) {
			msg = "Job creation temporarily disabled, retrying";
		} else if (errno == EAGAIN) {
			msg = "Slurm temporarily unable to accept job, "
			      "sleeping and retrying";
		} else
			msg = NULL;
		if ((msg == NULL) || (retries >= MAX_RETRIES)) {
			error("SBATCH: Batch job submission failed: %m");
			return;
		}

		if (retries)
			debug("%s", msg);
		else if (errno == ESLURM_NODES_BUSY)
			info("%s", msg); /* Not an error, powering up nodes */
		else
			error("%s", msg);
		slurm_free_submit_response_response_msg(resp);
		sleep(++retries);
	}

	if (!resp) {
		error("SBATCH: Batch job submission failed: %m");
		return;
	}

	print_multi_line_string(resp->job_submit_user_msg, -1, LOG_LEVEL_INFO);

	/* run cli_filter post_submit */
	for (i = 0; i < het_job_limit; i++)
		cli_filter_g_post_submit(i, resp->job_id, NO_VAL);

	if (!quiet) {
		if (!sbopt.parsable) {
			printf("Submitted batch job %u", resp->job_id);
			if (working_cluster_rec)
				printf(" on cluster %s",
				       working_cluster_rec->name);
			printf("\n");
		} else {
			printf("%u", resp->job_id);
			if (working_cluster_rec)
				printf(";%s", working_cluster_rec->name);
			printf("\n");
		}
	}

	if (sbopt.wait)
		rc = _job_wait(resp->job_id);

#ifdef MEMORY_LEAK_DEBUG
	slurm_select_fini();
	slurm_reset_all_options(&opt, false);
	slurm_auth_fini();
	slurm_conf_destroy();
	log_fini();
#endif /* MEMORY_LEAK_DEBUG */
	xfree(script_body);

	//
	if(resp != NULL) {
		// insert job to active simulated job list
		if(active_job->job_id==0) {
			pthread_mutex_lock(&events_mutex);
			active_job->job_id = resp->job_id;
			event_submit_batch_job->job_id = resp->job_id;
			pthread_mutex_unlock(&events_mutex);
		}
		if(active_job->job_id != resp->job_id) {
			error("Job id in event list (%d) does not match to one returned from sbatch (%d)",
					event_submit_batch_job->job_id, resp->job_id);
		}

	} else {
		error("Job was not submitted!");
	}

	return;
}


