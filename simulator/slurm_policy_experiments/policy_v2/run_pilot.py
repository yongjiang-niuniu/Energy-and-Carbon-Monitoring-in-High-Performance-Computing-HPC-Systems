"""Run one isolated pilot under the shared three-hour deadline."""
import hashlib
import json
from pathlib import Path
import sys
import time
from bounded import Budget


def run(root, name):
    root=Path(root).resolve()
    scenario=root/name
    budget=Budget(root)
    remaining=budget.record["deadline_epoch"]-time.time()
    if remaining<120: raise TimeoutError("Not enough aggregate budget to launch safely")
    limit=int(min(600,remaining-90))
    frozen=json.loads((root/"pilot_inputs.json").read_text())["hashes"]
    for path,expected in frozen.items():
        if path.startswith(name+"/"):
            assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected,path
    if (scenario/"slurmctld.log").exists(): raise FileExistsError("Never overwrite pilot evidence")
    container="stanage-v2-"+name.replace("_","-")
    command=["docker","--context","colima","run","--rm","--name",container,"--hostname","node001",
             "--cpus=2","--memory=2g","--pids-limit=256","--ulimit","core=0","--network","bridge",
             "-e","TZ=UTC","-e","POLICY_V2_CONFIG=/opt/slurm-sim/etc/controller.json",
             "-v",f"{scenario}:/opt/slurm-sim/etc", "slurm-sim-policy-v2:20260910-r2",
             "bash","-c",f"cd /opt/slurm-sim/etc && timeout -s KILL {limit} slurmctld -D -i"]
    code=budget.run(name,command,limit+30,container)
    log=(scenario/"slurmctld.log").read_text(errors="replace") if (scenario/"slurmctld.log").exists() else ""
    print(json.dumps({"process_exit":code,"all_done":"All done." in log,
                      "bridge_failsafe":"POLICY_V2_FAILSAFE" in log}))
    expected_failure=(scenario/"expected_failure.json").exists()
    observed_failure="POLICY_V2_FAILSAFE" in log
    return 0 if "All done." in log and observed_failure==expected_failure else 1


if __name__=="__main__":
    raise SystemExit(run(sys.argv[1],sys.argv[2]))
