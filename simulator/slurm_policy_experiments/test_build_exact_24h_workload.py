from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("exact", ROOT / "build_exact_24h_workload.py")
EXACT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EXACT)


def test_classify_jobs_uses_carry_in_precedence() -> None:
    t0 = pd.Timestamp("2025-11-12T00:00:00Z")
    t1 = t0 + pd.Timedelta(hours=24)
    rows = [
        ("2025-11-10", "2025-11-10", "2025-11-11", "2025-11-13"),
        ("2025-11-10", "2025-11-11", "2025-11-12T01:00", "2025-11-12T02:00"),
        ("2025-11-11", "2025-11-12T00:15", "2025-11-12T00:30", "2025-11-12T01:00"),
        ("2025-11-12", "2025-11-12T02:00", "2025-11-12T02:30", "2025-11-12T03:00"),
    ]
    frame = pd.DataFrame(rows, columns=["_submit", "_eligible", "_start", "_end"])
    for column in frame:
        frame[column] = pd.to_datetime(frame[column], utc=True, format="mixed")

    selected, audit = EXACT.classify_jobs(frame, t0, t1)

    assert selected["carry_in_type"].tolist() == ["running", "queued", "held", "evaluation"]
    assert audit["held_and_evaluation_definition_overlap"] == 1


def test_evaluation_window_excludes_t1() -> None:
    t0 = pd.Timestamp("2025-11-12T00:00:00Z")
    t1 = t0 + pd.Timedelta(hours=24)
    frame = pd.DataFrame(
        {
            "_submit": pd.to_datetime(["2025-11-12T23:00Z", "2025-11-12T23:00Z"]),
            "_eligible": pd.to_datetime(["2025-11-12T23:59Z", "2025-11-13T00:00Z"]),
            "_start": pd.to_datetime(["2025-11-13T00:01Z", "2025-11-13T00:01Z"]),
            "_end": pd.to_datetime(["2025-11-13T00:02Z", "2025-11-13T00:02Z"]),
        }
    )

    selected, _ = EXACT.classify_jobs(frame, t0, t1)

    assert selected["carry_in_type"].tolist() == ["evaluation"]
