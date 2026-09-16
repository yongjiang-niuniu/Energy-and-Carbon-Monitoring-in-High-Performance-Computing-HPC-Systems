"""Separate execution evidence from scientific acceptance."""
from __future__ import annotations

import pandas as pd


def decision_report(profile):
    candidates = pd.DataFrame(profile.attrs.get("v2_candidate_audit", []))
    if candidates.empty:
        candidates = pd.DataFrame(columns=["sim_job_id", "candidate_index", "accepted", "rejection_reason"])
    rows = []
    for row in profile.to_dict("records"):
        job = str(row["sim_job_id"])
        delay = max(float(row["release_dt_s"])-float(row["eligible_dt_s"]), 0)
        observed = candidates.loc[candidates.sim_job_id.eq(job)]
        explicit = next((row[key] for key in ("low_impact_rejection_reason", "safe_rejection_reason")
                         if isinstance(row.get(key), str) and row[key]), "")
        if delay > 0:
            reason = "accepted_delay"
        elif not row.get("is_evaluation", True):
            reason = "protected_carry_in"
        elif not row.get("is_flexible", True):
            reason = "not_flexible"
        elif pd.notna(row.get("runtime_prediction_reliable")) and not row["runtime_prediction_reliable"]:
            reason = "runtime_prediction_unreliable"
        elif isinstance(explicit, str) and explicit:
            reason = explicit
        elif not observed.empty:
            reason = ";".join(sorted({x for text in observed.rejection_reason for x in str(text).split(";") if x}))
        else:
            reason = "rule_no_move_details_not_recorded"
        rows.append({"sim_job_id": job, "action": "delay" if delay else "release",
                     "policy_delay_s": delay, "reason": reason,
                     "candidate_count": len(observed), "reason_is_reconstructed": bool(observed.empty and not explicit)})
    return pd.DataFrame(rows), candidates


def validation_panels(results, input_match=None, carry_in=None, carbon_coverage=None, replay_evidence=None):
    def panel(value, description):
        return {"status": "not_assessed" if value is None else ("pass" if value else "fail"), "description": description}
    required = results[["sim_submit_log_utc", "sim_start_log_utc", "sim_end_log_utc"]]
    complete = len(results) > 0 and results.terminal_status.eq("completed").all() and required.notna().all().all()
    errors = (results.sim_observed_runtime_s-results.runtime_s).abs()
    return {
        "completion": panel(bool(complete), "All jobs submitted, started and completed"),
        "input_consistency": panel(input_match, "Job and resource invariants"),
        "initial_state": panel(carry_in, "Running/queued/held reconstruction"),
        "timing_fidelity": {"status": "not_assessed" if errors.isna().any() else ("warning" if errors.gt(60).any() else "pass"),
                            "over_60s_jobs": int(errors.gt(60).sum()), "tolerance_s": 60,
                            "description": "Diagnostic tolerance, not a changed frozen acceptance rule"},
        "carbon_coverage": panel(carbon_coverage, "Same complete model-time coverage"),
        "repeatability": panel(replay_evidence, "Must be assessed from repeats, not inferred from completion"),
        "scientific_acceptance": "not_inferred_from_execution",
    }
