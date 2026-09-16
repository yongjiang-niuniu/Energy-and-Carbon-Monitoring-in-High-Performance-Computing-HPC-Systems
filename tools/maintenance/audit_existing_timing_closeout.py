"""Describe recorded timing only; never launch or alter a replay."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "simulator/slurm_policy_experiments/campaigns/multiday_20260908"
OUTPUT = ROOT / "work_logs/research_closeout_20260910"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_log(path, target_ids):
    stamp = re.compile(r"^\[([^]]+)\]")
    event = re.compile(r"\] (\d+)\s+1005\s*$")
    request = re.compile(r"REQUEST_COMPLETE_BATCH_SCRIPT .*JobId=(\d+)")
    done = re.compile(r"_job_complete: JobId=(\d+) done")
    previous = None
    pending = None
    records = {}
    maximum_gap = {"seconds": 0}
    excerpts = []
    with path.open(errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            match = stamp.match(line)
            if not match:
                continue
            now = datetime.fromisoformat(match[1]).replace(tzinfo=timezone.utc)
            if previous is not None:
                seconds = (now - previous[0]).total_seconds()
                if seconds > maximum_gap["seconds"]:
                    maximum_gap = {"seconds": seconds, "before_line": previous[1],
                                   "after_line": number, "before": previous[0].isoformat(),
                                   "after": now.isoformat()}
            previous = (now, number)
            event_match = event.search(line)
            if event_match:
                pending = {"event_line": number, "event_logged_at": now.isoformat(),
                           "event_due_us": int(event_match[1]), "event_line_text": line.strip()}
            rpc = request.search(line)
            if rpc:
                job = int(rpc[1])
                if job in target_ids:
                    if pending is None:
                        raise ValueError(f"No preceding completion-event marker: {job}")
                    records[job] = {**pending, "rpc_line": number, "rpc_at": now.isoformat()}
                    excerpts.extend([pending["event_line_text"], line.strip()])
                pending = None
            complete = done.search(line)
            if complete and int(complete[1]) in target_ids:
                job = int(complete[1])
                records[job].update({"completion_line": number, "completion_at": now.isoformat()})
                excerpts.append(line.strip())
    if set(records) != target_ids:
        raise ValueError("Missing raw-log completion evidence")
    return records, maximum_gap, excerpts


def main():
    manifest = json.loads((CAMPAIGN / "manifest.json").read_text())
    state = json.loads((CAMPAIGN / "state.json").read_text())
    if state["status"] != "complete":
        raise ValueError("Only inspect the completed batch")
    preserved = [CAMPAIGN / "state.json", CAMPAIGN / "manifest.json", CAMPAIGN / "code_hashes.json"]
    sources = {str(p): digest(p) for p in preserved}
    names = ["baseline", "baseline_repeat_1", "baseline_repeat_2"]
    names += [p["name"] for p in manifest["policies"]] + ["adaptive_balanced_repeat"]
    groups, severe, event_rows = [], [], []
    for date in manifest["dates"]:
        for scenario in names:
            path = CAMPAIGN / "days" / date / scenario / "job_results.csv"
            sources[str(path)] = digest(path)
            frame = pd.read_csv(path)
            starts = pd.to_datetime(frame.sim_start_log_utc, utc=True)
            ends = pd.to_datetime(frame.sim_end_log_utc, utc=True)
            expected = starts + pd.to_timedelta(frame.runtime_s, unit="s")
            frame["signed_runtime_error_s"] = (ends - expected).dt.total_seconds()
            frame["absolute_error_s"] = frame.signed_runtime_error_s.abs()
            frame["expected_completion_from_log_start"] = expected
            frame["overrun_begins_after_last_start"] = expected > starts.max()
            if not frame.terminal_status.eq("completed").all():
                raise ValueError("Incomplete accepted replay")
            for (evaluation, carry_type), part in frame.groupby(["is_evaluation", "carry_in_type"], dropna=False):
                groups.append({"date": date, "scenario": scenario, "is_evaluation": bool(evaluation),
                               "carry_in_type": str(carry_type), "jobs": len(part),
                               "over_60s": int(part.absolute_error_s.gt(60).sum()),
                               "over_1h": int(part.absolute_error_s.gt(3600).sum()),
                               "p99_abs_error_s": float(part.absolute_error_s.quantile(.99)),
                               "max_abs_error_s": float(part.absolute_error_s.max()),
                               "overrun_over_60s_after_last_start": int((part.signed_runtime_error_s.gt(60) & part.overrun_begins_after_last_start).sum())})
            large = frame.loc[frame.absolute_error_s.gt(3600)]
            if large.empty:
                continue
            log = path.parent / "slurmctld.log"
            sources[str(log)] = digest(log)
            records, gap, excerpts = inspect_log(log, set(large.slurm_job_id_expected.astype(int)))
            for index, row in large.iterrows():
                record = records[int(row.slurm_job_id_expected)]
                due = pd.to_datetime(record["event_due_us"], unit="us", utc=True)
                logged_end = pd.to_datetime(record["completion_at"], utc=True)
                if abs((logged_end - ends.loc[index]).total_seconds()) > 1e-6:
                    raise ValueError("CSV and raw-log completion differ")
                severe.append({"date": date, "scenario": scenario,
                               "sim_job_id": row.sim_job_id, "slurm_job_id_expected": int(row.slurm_job_id_expected),
                               "is_evaluation": bool(row.is_evaluation), "carry_in_type": row.carry_in_type,
                               "configured_runtime_s": float(row.runtime_s),
                               "observed_runtime_s": float(row.sim_observed_runtime_s),
                               "runtime_error_s": float(row.signed_runtime_error_s),
                               "start_log_utc": starts.loc[index].isoformat(),
                               "last_any_job_start_log_utc": starts.max().isoformat(),
                               "expected_completion_log_utc": expected.loc[index].isoformat(),
                               "event_due_log_utc": due.isoformat(),
                               "completion_log_utc": logged_end.isoformat(),
                               "due_vs_expected_error_s": (due-expected.loc[index]).total_seconds(),
                               "overrun_begins_after_last_start": bool(row.overrun_begins_after_last_start),
                               "later_job_starts_after_expected_completion": int(starts.gt(expected.loc[index]).sum()),
                               **{k: v for k, v in record.items() if k != "event_line_text"}})
            event_rows.append({"date": date, "scenario": scenario, "largest_adjacent_log_gap": gap,
                               "completion_excerpt": excerpts})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(groups).to_csv(OUTPUT / "runtime_by_cohort.csv", index=False)
    pd.DataFrame(severe).to_csv(OUTPUT / "large_runtime_errors_with_event_evidence.csv", index=False)
    (OUTPUT / "completion_event_excerpts.json").write_text(json.dumps(event_rows, indent=2) + "\n")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "analysis_only": True,
        "new_replays": 0, "existing_replays_read": len(names) * len(manifest["dates"]),
        "over_1h_jobs": len(severe),
        "over_1h_evaluation_jobs": sum(r["is_evaluation"] for r in severe),
        "over_1h_carry_in_jobs": sum(not r["is_evaluation"] for r in severe),
        "all_large_overruns_begin_after_all_jobs_started": all(r["overrun_begins_after_last_start"] for r in severe),
        "max_event_due_vs_start_plus_runtime_s": max(abs(r["due_vs_expected_error_s"]) for r in severe),
        "largest_recorded_log_gap_s": max(r["largest_adjacent_log_gap"]["seconds"] for r in event_rows),
        "scope_limit": "No causal claim about other timing errors, baseline variation or the unverified trigger of the log gap.",
        "inputs_sha256": sources,
    }
    for path, before in sources.items():
        if digest(Path(path)) != before:
            raise ValueError(f"Source changed during read-only audit: {path}")
    (OUTPUT / "timing_closeout_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "inputs_sha256"}, indent=2))


if __name__ == "__main__":
    main()
