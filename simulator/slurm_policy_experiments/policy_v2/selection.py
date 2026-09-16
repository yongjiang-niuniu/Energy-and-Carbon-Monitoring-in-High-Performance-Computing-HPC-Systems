"""Shared feasible-first selection and scalar decision records."""
from __future__ import annotations

import math
from numbers import Real


def scalar_record(option):
    output = {}
    for key, value in option.items():
        if isinstance(value, Real):
            output[key] = float(value) if math.isfinite(value) else None
        elif isinstance(value, (str, bool)) or value is None:
            output[key] = value
        elif hasattr(value, "isoformat"):
            output[key] = value.isoformat()
    return output


def choose(options, gates, score_key="utility", release_key="release", maximize=True):
    audit, feasible = [], []
    for number, option in enumerate(options):
        reasons = [name for name, predicate in gates if not predicate(option)]
        score = option.get(score_key)
        if not isinstance(score, Real) or not math.isfinite(score):
            reasons.append("non_finite_score")
        audit.append({**scalar_record(option), "candidate_index": number,
                      "accepted": False, "rejection_reason": ";".join(reasons)})
        if not reasons:
            feasible.append((number, option))
    if not feasible:
        return None, audit
    key = lambda pair: ((-1 if maximize else 1) * pair[1][score_key], pair[1][release_key])
    index, best = min(feasible, key=key)
    audit[index]["accepted"] = True
    for row in audit:
        if not row["accepted"] and not row["rejection_reason"]:
            row["rejection_reason"] = "feasible_lower_rank"
    return best, audit


def record_choice(frame, index, options, gates, **kwargs):
    best, audit = choose(options, gates, **kwargs)
    records = frame.attrs.setdefault("v2_candidate_audit", [])
    job = str(frame.at[index, "sim_job_id"])
    records.extend({"sim_job_id": job, **row} for row in audit)
    return best
