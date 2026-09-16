"""A deliberate controller startup failure, not an efficacy experiment."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

root=Path(sys.argv[1]);target=root/"controller_failure_check"
target.mkdir()
names=("workload_profile.csv","sim.events","users.sim","slurm.conf","sim.conf","gres.conf","carbon.csv")
for name in names: shutil.copy2(root/"functional_check2_baseline"/name,target/name)
(target/"controller.json").write_text("{}\n")
(target/"expected_failure.json").write_text(json.dumps({"fault":"missing controller configuration fields",
    "expected":"logged failsafe, all arrived and future jobs released, all jobs complete, no deadlock"},indent=2))
manifest=json.loads((root/"pilot_inputs.json").read_text())
for path in target.iterdir():
    manifest["hashes"][str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
(root/"pilot_inputs.json").write_text(json.dumps(manifest,indent=2))
