"""
Unit tests for PCR raw log parsing.
"""

from pathlib import Path
import pandas as pd
import pytest

from pcr_ml.parser import parse_pcr_file, load_all_runs


def test_parse_sample_file(tmp_path):
    sample_content = """====================================================
PCR: S2 CONTROL
Time, Cycle, Phase, T_OUT, T_IN, F T_IN, LED PWM, Status
====================================================
Time:1.0, Cycle:1, INIT, T_OUT:25.00, T_IN:25.50, F T_IN:25.50, LED PWM:255, HEATING_FULL
Time:2.0, Cycle:1, INIT, T_OUT:26.00, T_IN:27.00, F T_IN:27.00, LED PWM:255, HEATING_FULL
Time:3.0, Cycle:1, INIT, T_OUT:28.00, T_IN:30.00, F T_IN:30.00, LED PWM:255, HEATING_FULL
Time:4.0, Cycle:1, INIT, T_OUT:31.00, T_IN:34.00, F T_IN:34.00, LED PWM:255, HEATING_FULL
"""
    log_file = tmp_path / "test_run.txt"
    log_file.write_text(sample_content, encoding="utf-8")

    df = parse_pcr_file(log_file, run_name="unit_test")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert list(df["cycle"]) == [1, 1, 1, 1]
    assert df["t_out"].iloc[0] == 25.00
    assert df["t_in"].iloc[0] == 25.50
    assert df["run_name"].iloc[0] == "unit_test"


def test_load_all_runs_discovers_files():
    repo_dir = Path(__file__).resolve().parents[1]
    runs = load_all_runs(repo_dir)
    assert len(runs) >= 5
    for name, df in runs.items():
        assert not df.empty
        assert "t_out" in df.columns
        assert "t_in" in df.columns
        assert "cycle" in df.columns
        assert "phase" in df.columns
