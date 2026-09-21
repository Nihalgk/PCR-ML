"""
Unit tests for PCR-ML model training and predictions.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pcr_ml.parser import parse_pcr_file
from pcr_ml.features import engineer_features
from pcr_ml.models import PCRLinearModel, PhaseSpecificPCRModel, calculate_metrics


@pytest.fixture
def sample_data():
    repo_dir = Path(__file__).resolve().parents[1]
    sample_file = repo_dir / "data" / "40 cycle(3step)_19.08.2026.txt"
    df = parse_pcr_file(sample_file)
    df_feat = engineer_features(df)
    return df_feat


def test_linear_model_fit_and_metrics(sample_data):
    model = PCRLinearModel()
    model.fit(sample_data, sample_data["t_in"])
    preds = model.predict(sample_data)
    assert len(preds) == len(sample_data)

    metrics = calculate_metrics(sample_data["t_in"], preds)
    assert metrics["mae"] < 8.0
    assert metrics["r2"] > 0.80


def test_phase_specific_model_convergence(sample_data):
    model = PhaseSpecificPCRModel()
    model.fit(sample_data, target_col="t_in")
    preds = model.predict(sample_data)
    assert len(preds) == len(sample_data)
    assert not np.isnan(preds).any()

    metrics = calculate_metrics(sample_data["t_in"], preds)
    # Phase specific model should achieve high accuracy on training run
    assert metrics["mae"] < 4.0
    assert metrics["r2"] > 0.85
