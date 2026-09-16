"""Preserve the first integration attempt and freeze the corrected check."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

root=Path(sys.argv[1])
manifest=json.loads((root/"pilot_inputs.json").read_text())
for mode in ("baseline","feedback"):
    target=root/("functional_check2_"+mode)
    target.mkdir()
    for name in ("workload_profile.csv","sim.events","users.sim","slurm.conf","sim.conf","gres.conf","carbon.csv","controller.json","workload_generation_summary.json"):
        shutil.copy2(root/("functional_"+mode)/name,target/name)
        manifest["hashes"][str((target/name).relative_to(root))]=hashlib.sha256((target/name).read_bytes()).hexdigest()
manifest["integration_fix"]={"change":"arrival-anchored candidate grid and feasible runner-up under user budget",
                            "parameter_changes":False,"first_attempt_retained":True,
                            "not_performance_tuning":True,"image":"slurm-sim-policy-v2:20260910-r2"}
(root/"pilot_inputs.json").write_text(json.dumps(manifest,indent=2))
