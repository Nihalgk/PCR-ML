"""
Evaluation and Cross-Validation module for PCR ML thermal lag models.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, max_error

from .features import DEFAULT_FEATURES, engineer_features
from .models import PCRLinearModel, PhaseSpecificPCRModel, PCRRandomForestModel, calculate_metrics


def evaluate_leave_one_out(
    runs: Dict[str, pd.DataFrame],
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Perform Leave-One-Run-Out Cross-Validation across all experimental runs.

    Parameters
    ----------
    runs : Dict[str, pd.DataFrame]
        Dictionary of run names and parsed DataFrames.
    features : Optional[List[str]]
        Features to evaluate.

    Returns
    -------
    pd.DataFrame
        Table of per-run and average cross-validation metrics.
    """
    features = features or DEFAULT_FEATURES
    
    # Ensure features are engineered
    prepared_runs = {}
    for name, df in runs.items():
        if "t_out_lag1" not in df.columns:
            prepared_runs[name] = engineer_features(df)
        else:
            prepared_runs[name] = df.copy()

    results = []

    for test_name in prepared_runs.keys():
        train_dfs = [df for name, df in prepared_runs.items() if name != test_name]
        if not train_dfs:
            continue
            
        df_train = pd.concat(train_dfs, ignore_index=True)
        df_test = prepared_runs[test_name]

        # 1. Global Linear
        m_linear = PCRLinearModel(features=features)
        m_linear.fit(df_train, df_train["t_in"])
        pred_linear = m_linear.predict(df_test)
        mae_linear = mean_absolute_error(df_test["t_in"], pred_linear)

        # 2. Phase-Specific Linear
        m_phase = PhaseSpecificPCRModel(features=features)
        m_phase.fit(df_train, target_col="t_in")
        pred_phase = m_phase.predict(df_test)
        mae_phase = mean_absolute_error(df_test["t_in"], pred_phase)

        # 3. Random Forest Benchmark
        m_rf = PCRRandomForestModel(features=features, n_estimators=40, max_depth=8)
        m_rf.fit(df_train, df_train["t_in"])
        pred_rf = m_rf.predict(df_test)
        mae_rf = mean_absolute_error(df_test["t_in"], pred_rf)

        results.append({
            "Test Run": test_name,
            "Samples": len(df_test),
            "Global Linear MAE (°C)": round(mae_linear, 3),
            "Phase-Specific MAE (°C)": round(mae_phase, 3),
            "Random Forest MAE (°C)": round(mae_rf, 3),
        })

    df_res = pd.DataFrame(results)
    return df_res


def evaluate_model_pipeline(
    df: pd.DataFrame,
    features: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive pipeline evaluation including phase and hold breakdown.
    """
    features = features or DEFAULT_FEATURES
    if "t_out_lag1" not in df.columns:
        df = engineer_features(df)

    m_phase = PhaseSpecificPCRModel(features=features)
    m_phase.fit(df, target_col="t_in")
    df_eval = df.copy()
    df_eval["pred_phase"] = m_phase.predict(df)

    report = {
        "overall": calculate_metrics(df_eval["t_in"], df_eval["pred_phase"]),
    }

    # Per phase
    for sp in ["DENAT", "ANNEAL", "EXTEND"]:
        mask = df_eval["super_phase"] == sp
        if mask.sum() > 0:
            report[sp.lower()] = calculate_metrics(
                df_eval.loc[mask, "t_in"],
                df_eval.loc[mask, "pred_phase"]
            )

    # Holds vs Ramps
    if "is_hold" in df_eval.columns:
        hold_mask = df_eval["is_hold"]
        if hold_mask.sum() > 0:
            report["holds"] = calculate_metrics(
                df_eval.loc[hold_mask, "t_in"],
                df_eval.loc[hold_mask, "pred_phase"]
            )
        ramp_mask = ~df_eval["is_hold"]
        if ramp_mask.sum() > 0:
            report["ramps"] = calculate_metrics(
                df_eval.loc[ramp_mask, "t_in"],
                df_eval.loc[ramp_mask, "pred_phase"]
            )

    return report
