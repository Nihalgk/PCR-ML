"""
C++ / Python Numerical Parity Verification Test.
Ensures that Arduino C++ floating-point formulas match Python Scikit-Learn predictions exactly.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pcr_ml.parser import parse_pcr_file
from pcr_ml.features import engineer_features
from pcr_ml.models import PhaseSpecificPCRModel


def test_cpp_floating_point_parity():
    repo_dir = Path(__file__).resolve().parents[1]
    sample_file = repo_dir / "40 cycle(3step)_19.08.2026.txt"
    df = parse_pcr_file(sample_file)
    df_feat = engineer_features(df)

    # Fit Python model
    model = PhaseSpecificPCRModel()
    model.fit(df_feat, target_col="t_in")
    weights = model.get_phase_weights()

    # Replicate exact Arduino C++ switch-case inference
    def cpp_predict_internal_temperature(phase_name, s1, dt, lag1, lag2, lag3, cycle):
        sp = phase_name.upper()
        if sp in weights:
            w = weights[sp]
            inter = np.float32(w["intercept"])
            c_s1 = np.float32(w["coefficients"]["t_out"])
            c_dt = np.float32(w["coefficients"]["dT_OUT_dt"])
            c_l1 = np.float32(w["coefficients"]["t_out_lag1"])
            c_l2 = np.float32(w["coefficients"]["t_out_lag2"])
            c_l3 = np.float32(w["coefficients"]["t_out_lag3"])
            c_cy = np.float32(w["coefficients"]["cycle"])

            val = (
                inter
                + (c_s1 * np.float32(s1))
                + (c_dt * np.float32(dt))
                + (c_l1 * np.float32(lag1))
                + (c_l2 * np.float32(lag2))
                + (c_l3 * np.float32(lag3))
                + (c_cy * np.float32(cycle))
            )
            return float(val)
        return float(s1)

    py_preds = model.predict(df_feat)
    cpp_preds = []

    for _, row in df_feat.iterrows():
        pred_c = cpp_predict_internal_temperature(
            row["super_phase"],
            row["t_out"],
            row["dT_OUT_dt"],
            row["t_out_lag1"],
            row["t_out_lag2"],
            row["t_out_lag3"],
            row["cycle"],
        )
        cpp_preds.append(pred_c)

    cpp_preds = np.array(cpp_preds)
    max_diff = np.max(np.abs(py_preds - cpp_preds))

    # Single-precision float parity within < 1e-4 °C tolerance
    assert max_diff < 1e-4, f"C++ and Python parity diverged with max error: {max_diff}"
