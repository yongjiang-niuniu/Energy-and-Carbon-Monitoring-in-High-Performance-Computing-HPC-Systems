#
# Makefile for slurmctld

AUTOMAKE_OPTIONS = foreign
CLEANFILES = core.*

AM_CPPFLAGS = -I$(top_srcdir)

# noinst_LTLIBRARIES = libslurmctld.la
# libslurmctld_la_LDFLAGS  = $(LIB_LDFLAGS) -module --export-dynamic
# libslurmctld_la_SOURCES =
slurmctld_SOURCES =     \
	acct_policy.c	\
	acct_policy.h	\
	agent.h		\
	backup.c	\
	crontab.c 	\
	fed_mgr.c 	\
	fed_mgr.h 	\
	front_end.c	\
	front_end.h	\
	gang.c		\
	gang.h		\
	gres_ctld.c	\
	gres_ctld.h	\
	groups.c	\
	groups.h	\
	heartbeat.c	\
	heartbeat.h	\
	job_mgr.c 	\
	job_scheduler.h	\
	licenses.c	\
	licenses.h	\
	locks.c   	\
	locks.h  	\
	node_mgr.c 	\
	node_scheduler.c \
	node_scheduler.h \
	partition_mgr.c \
	ping_nodes.c	\
	ping_nodes.h	\
	port_mgr.c	\
	port_mgr.h	\
	power_save.c	\
	power_save.h	\
	prep_slurmctld.c \
	proc_req.c	\
	proc_req.h	\
	rate_limit.c	\
	rate_limit.h	\
	read_config.c	\
	read_config.h	\
	reservation.c	\
	reservation.h	\
	rpc_queue.c	\
	rpc_queue.h	\
	slurmctld.h	\
	slurmscriptd.c \
	slurmscriptd.h \
	slurmscriptd_protocol_defs.c \
	slurmscriptd_protocol_defs.h \
	slurmscriptd_protocol_pack.c \
	slurmscriptd_protocol_pack.h \
	srun_comm.c	\
	srun_comm.h	\
	state_save.c	\
	state_save.h	\
	statistics.c	\
	step_mgr.c	\
	trigger_mgr.c	\
	trigger_mgr.h

if ENABLE_SIMULATOR
slurmctld_SOURCES +=	\
    ../../contribs/sim/sim_events.c	\
    ../../contribs/sim/sim_jobs.c	\
    ../../contribs/sim/sim_sbatch.c	 ../sbatch/opt.c opt.h ../sbatch/xlate.c \
    ../../contribs/sim/sim_slurmd.c	\
	../../contribs/sim/sim_slurmd_req.c	\
	../../contribs/sim/sim_slurmd_get_mach_stat.c	\
	../../contribs/sim/slurmctld_controller.c	\
	../../contribs/sim/slurmctld_agent.c \
    ../../contribs/sim/slurmctld_job_scheduler.c
else
slurmctld_SOURCES += controller.c agent.c job_scheduler.c
endif

depend_libs =
depend_ldadd =
depend_ldflags =

if ENABLE_SIMULATOR
        depend_libs += \
    $(top_builddir)/src/bcast/libfile_bcast.la \
	$(top_builddir)/src/slurmd/common/libslurmd_common.la
        depend_ldadd += $(NUMA_LIBS)
# These 2 depend_* should look the same in ../slurmstepd/Makefile.am
depend_ldadd += $(HWLOC_LIBS) $(PAM_LIBS) $(UTIL_LIBS)
depend_ldflags += $(HWLOC_LDFLAGS) $(SLURMD_INTERFACES)

# This is needed to actually link to the libs that it doesn't use.
depend_ldflags += -Wl,--no-as-needed
# same function waitpid_timeout defined twice rename one
        slurmctld_CFLAGS = -Dwaitpid_timeout=slurmctld_waitpid_timeout

endif

sbin_PROGRAMS = slurmctld

slurmctld_LDADD = $(depend_libs) $(LIB_SLURM) $(SLURMCTLD_INTERFACES) $(depend_ldadd)
slurmctld_LDFLAGS = $(CMD_LDFLAGS) $(depend_ldflags)

slurmctld_DEPENDENCIES = $(LIB_SLURM_BUILD) $(SLURMCTLD_INTERFACES)
if ENABLE_SIMULATOR
slurmctld_DEPENDENCIES += $(SLURMD_INTERFACES)
endif

force:
$(slurmctld_DEPENDENCIES) : force
	@cd `dirname $@` && $(MAKE) `basename $@`
