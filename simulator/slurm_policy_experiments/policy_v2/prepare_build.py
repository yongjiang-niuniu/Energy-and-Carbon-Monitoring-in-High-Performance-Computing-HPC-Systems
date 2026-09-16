"""Build an isolated source overlay; never patch the frozen simulator tree."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / "slurm_simulator_source_legacy_desktop"


def prepare(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(SOURCE, destination / "source", ignore=shutil.ignore_patterns(".git", "core", "*.log"))
    target = destination / "source/contribs/sim/slurmctld_controller.c"
    text = target.read_text()
    original_hash = hashlib.sha256(text.encode()).hexdigest()
    changes = [
        ("extern int sim_init_slurmd();", '#include "feedback_bridge.h"\nextern int sim_init_slurmd();'),
        ("submit_job((sim_event_submit_batch_job_t*)event->payload);", "v2_arrive((sim_event_submit_batch_job_t*)event->payload,event->when);"),
        ("\t// run main scheduler if needed", "\tif(recursion_depth==1) { v2_tick(now); now=get_sim_utime(); }\n\t// run main scheduler if needed"),
        ("if(sim_n_noncyclic_events<=0 &&", "if(v2_pending()==0 && sim_n_noncyclic_events<=0 &&"),
        ("int64_t skipping_to_utime = sim_main_thread_sleep_till;", "int64_t skipping_to_utime = sim_main_thread_sleep_till;\n\tif(v2_pending() && v2_wake<skipping_to_utime) skipping_to_utime=v2_wake;"),
    ]
    for old, new in changes:
        if text.count(old) != 1:
            raise ValueError(f"Ambiguous source anchor: {old}")
        text = text.replace(old, new)
    target.write_text(text)
    shutil.copy2(ROOT / "feedback_bridge.h", target.parent)
    shutil.copytree(ROOT, destination / "policy", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "Dockerfile.feedback", destination / "Dockerfile")
    (destination / "overlay.json").write_text(json.dumps({"original_controller_sha256": original_hash,
        "overlay_controller_sha256": hashlib.sha256(text.encode()).hexdigest(), "source": str(SOURCE),
        "scope": "independent simulator-only feedback pilot; no frozen source edits"}, indent=2))


if __name__ == "__main__":
    prepare(sys.argv[1])
