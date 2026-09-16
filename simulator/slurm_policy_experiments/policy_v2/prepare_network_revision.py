"""Keep the failed network-isolated attempt; do not alter its inputs or logs."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

root=Path(sys.argv[1])
manifest=json.loads((root/"pilot_inputs.json").read_text())
for mode in ("baseline","feedback"):
    target=root/("real64_check2_"+mode);target.mkdir()
    for name in ("workload_profile.csv","sim.events","users.sim","slurm.conf","sim.conf","gres.conf","carbon.csv","controller.json","workload_generation_summary.json"):
        shutil.copy2(root/("real64_"+mode)/name,target/name)
        manifest["hashes"][str((target/name).relative_to(root))]=hashlib.sha256((target/name).read_bytes()).hexdigest()
manifest["network_fix"]={"failed_case":"real64_baseline","reason":"network=none caused Slurm address resolution failures and ReqNodeNotAvail",
                         "corrected_mode":"Docker bridge, no published ports; same mode as existing project runners",
                         "policy_or_input_changes":False,"real64_feedback_initial_attempt":"not run"}
(root/"pilot_inputs.json").write_text(json.dumps(manifest,indent=2))
