"""
Unit tests for feature engineering and circular buffer simulation.
"""

import numpy as np
import pandas as pd
import pytest

from pcr_ml.features import (
    engineer_features,
    extract_phase_supercategories,
    ArduinoCircularBufferSimulator,
    DEFAULT_FEATURES,
)


def test_phase_supercategories_mapping():
    phases = pd.Series(["INIT", "INIT_DENAT", "CYCLE_DENAT", "CYCLE_ANNEAL", "CYCLE_EXTENSION", "FINAL_EXTENSION"])
    super_phases = extract_phase_supercategories(phases)
    assert super_phases.tolist() == ["DENAT", "DENAT", "DENAT", "ANNEAL", "EXTEND", "EXTEND"]


def test_feature_engineering_lags_and_derivatives():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "cycle": [1, 1, 1, 1, 1, 1],
        "phase": ["CYCLE_DENAT"] * 6,
        "t_out": [20.0, 22.0, 25.0, 29.0, 34.0, 40.0],
        "t_in": [21.0, 23.0, 26.0, 30.0, 35.0, 41.0],
    }
    df = pd.DataFrame(data)
    df_feat = engineer_features(df, sample_interval_ms=1000)

    # 3 warmup rows dropped
    assert len(df_feat) == 3
    # Row 4 (t_out=29.0): lag1=25.0, lag2=22.0, lag3=20.0, dT/dt=29-25=4.0
    row4 = df_feat.iloc[0]
    assert row4["t_out"] == 29.0
    assert row4["t_out_lag1"] == 25.0
    assert row4["t_out_lag2"] == 22.0
    assert row4["t_out_lag3"] == 20.0
    assert row4["dT_OUT_dt"] == 4.0
    assert row4["super_phase"] == "DENAT"


def test_arduino_circular_buffer_simulator():
    sim = ArduinoCircularBufferSimulator(buffer_size=31)
    
    # Push 30 samples (buffer should not be ready yet)
    for i in range(30):
        res = sim.update(float(i))
        assert res is None

    # Push 31st sample (buffer is now full)
    # Samples: 0..30 (sample 30 is head, sample 20 is 10 steps back / 1s, sample 10 is 20 steps back / 2s, sample 0 is 30 steps back / 3s)
    res = sim.update(30.0)
    assert res is not None
    assert res["t_out"] == 30.0
    assert res["t_out_lag1"] == 20.0
    assert res["t_out_lag2"] == 10.0
    assert res["t_out_lag3"] == 0.0
    assert res["dT_OUT_dt"] == 10.0
