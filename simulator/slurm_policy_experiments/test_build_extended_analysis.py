from __future__ import annotations

import pandas as pd

from build_extended_analysis import concentration, prediction_metrics


def test_concentration_handles_zero_and_dominant_user() -> None:
    assert concentration(pd.Series([0, 0])) == (0.0, 0.0)
    top, hhi = concentration(pd.Series([3, 1]))
    assert top == 0.75
    assert hhi == 0.625


def test_prediction_metrics_reports_quantile_coverage() -> None:
    frame = pd.DataFrame(
        {
            "runtime_s": [10, 20],
            "predicted_runtime_s_q50": [15, 15],
            "predicted_runtime_s_q90": [30, 30],
        }
    )
    metrics = prediction_metrics(frame, "runtime_s", "runtime")
    assert metrics["runtime_q50_coverage_pct"] == 50.0
    assert metrics["runtime_q90_coverage_pct"] == 100.0
