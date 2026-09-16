"""Close only after all required evidence exists and no pilot remains active."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import analyse_pilots

root=Path(sys.argv[1]).resolve()
analyse_pilots.main(root)
result=json.loads((root/"results.json").read_text())
assert not result["frozen_source_changes"]
assert not result["pilot_input_changes"]
by_case={row["case"]:row for row in result["experiments"]}
required=[f"{stem}_{mode}" for stem in ("functional_check2","real64_check2") for mode in ("baseline","feedback")]
for case in required:
    record=by_case[case]
    assert record["complete_jobs"]==record["jobs"],case
    assert record["panels"]["controller_integrity"]["status"]=="pass",case
    assert record["panels"]["carbon_coverage"]["status"]=="pass",case
assert by_case["functional_check2_feedback"]["held_jobs"]>0
assert by_case["functional_check2_feedback"]["cancelled_jobs"]>0
assert result["fault_injection"]["pass"]
active=subprocess.check_output(["docker","--context","colima","ps","--format","{{.Names}}"],text=True)
assert not any(name.startswith("stanage-v2-") for name in active.splitlines())
phases=[json.loads(line) for line in (root/"phases.jsonl").read_text().splitlines()]
tests=[row for row in phases if row["phase"].startswith("regression")]
assert tests[-1]["exit_code"]==0
source=Path(__file__).resolve().parent
hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir() if path.is_file()}
overlay=json.loads((root/"build_01/overlay.json").read_text())
old=Path(overlay["source"])/"contribs/sim/slurmctld_controller.c"
assert hashlib.sha256(old.read_bytes()).hexdigest()==overlay["original_controller_sha256"]
image=json.loads(subprocess.check_output(["docker","--context","colima","image","inspect","slurm-sim-policy-v2:20260910-r2"],text=True))[0]["Id"]
runtime=subprocess.check_output(["docker","--context","colima","run","--rm","--network","none",image,
                                 "sha256sum","/opt/policy-v2/feedback.py","/opt/policy-v2/selection.py"],text=True)
for line in runtime.splitlines():
    value,path=line.split()
    assert hashes[Path(path).name]==value
budget=json.loads((root/"budget.json").read_text())
now=time.time()
assert now<budget["deadline_epoch"],"Do not label an over-budget run compliant"
record={"status":"completed_and_closed","finished_utc":dt.datetime.fromtimestamp(now,dt.timezone.utc).isoformat(),
        "aggregate_elapsed_seconds":now-budget["started_epoch"],"limit_seconds":10800,
        "sum_phase_wall_seconds":sum(row["elapsed_seconds"] for row in phases),
        "new_training":False,"full_day_replay":False,"frozen_campaign_changed":False,
        "more_runs_scheduled":False,"active_pilot_containers":False,
        "image_id":image,"current_source_hashes":hashes,
        "controller_image_matches_current_runtime_sources":True,
        "legacy_controller_source_unchanged":True}
with (root/"finished.json").open("x") as stream:
    json.dump(record,stream,indent=2)
print(json.dumps(record,indent=2))
