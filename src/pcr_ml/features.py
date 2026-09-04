"""
Feature engineering module for PCR thermal modeling and embedded inference simulation.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


FEATURE_SET_M1 = ["t_out"]
FEATURE_SET_M2 = ["t_out", "dT_OUT_dt"]
FEATURE_SET_M3 = ["t_out", "dT_OUT_dt", "t_out_lag1", "t_out_lag2", "t_out_lag3", "cycle"]
FEATURE_SET_M4 = ["t_out", "dT_OUT_dt", "t_out_lag1", "t_out_lag2", "t_out_lag3", "cycle", "t_out_sq"]

DEFAULT_FEATURES = FEATURE_SET_M3


def extract_phase_supercategories(phase_series: pd.Series) -> pd.Series:
    """
    Map fine-grained firmware phases into 3 canonical PCR thermodynamic super-phases:
    - DENAT: Denaturation (~95°C) [INIT, INIT_DENAT, CYCLE_DENAT, DENAT]
    - ANNEAL: Primer Annealing (~60°C) [CYCLE_ANNEAL, ANNEAL]
    - EXTEND: Polymerase Extension (~72°C) [CYCLE_EXTENSION, EXTEND, FINAL_EXTENSION]
    """
    phase_clean = phase_series.astype(str).str.upper().str.strip()
    
    super_phase = phase_clean.copy()
    
    # Denaturation mappings
    super_phase = super_phase.replace({
        "INIT": "DENAT",
        "INIT_DENAT": "DENAT",
        "CYCLE_DENAT": "DENAT",
    })
    
    # Annealing mappings
    super_phase = super_phase.replace({
        "CYCLE_ANNEAL": "ANNEAL",
    })
    
    # Extension mappings
    super_phase = super_phase.replace({
        "CYCLE_EXTENSION": "EXTEND",
        "FINAL_EXTENSION": "EXTEND",
        "EXTENSION": "EXTEND",
    })
    
    # Handle any unmapped phase
    valid_phases = {"DENAT", "ANNEAL", "EXTEND"}
    super_phase = super_phase.apply(lambda p: p if p in valid_phases else "DENAT")
    
    return super_phase


def is_temperature_holding(row: pd.Series) -> bool:
    """
    Determine if a telemetry record is in a stable thermal hold vs dynamic ramp.
    """
    sp = row.get("super_phase")
    t_in = row.get("t_in", np.nan)
    if pd.isna(t_in):
        return False
        
    if sp == "DENAT":
        return t_in >= 94.0
    elif sp == "ANNEAL":
        return t_in <= 61.0
    elif sp == "EXTEND":
        return t_in >= 71.0
    return False


def engineer_features(
    df: pd.DataFrame,
    sample_interval_ms: int = 1000
) -> pd.DataFrame:
    """
    Engineer time-lagged, derivative, and polynomial thermal features from raw telemetry.

    Parameters
    ----------
    df : pd.DataFrame
        Parsed telemetry DataFrame with 't_out', 'cycle', 'phase'.
    sample_interval_ms : int
        Telemetry sample rate in milliseconds (default: 1000ms).

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with engineered features and NaN rows dropped.
    """
    df = df.copy()
    
    # Determine shift step sizes based on sampling frequency
    # 1.0s, 2.0s, 3.0s lags
    lag_step = 1 if sample_interval_ms >= 500 else int(round(1000 / sample_interval_ms))
    
    group_col = "run_name" if "run_name" in df.columns else None

    if group_col:
        df["t_out_lag1"] = df.groupby(group_col)["t_out"].shift(1 * lag_step)
        df["t_out_lag2"] = df.groupby(group_col)["t_out"].shift(2 * lag_step)
        df["t_out_lag3"] = df.groupby(group_col)["t_out"].shift(3 * lag_step)
    else:
        df["t_out_lag1"] = df["t_out"].shift(1 * lag_step)
        df["t_out_lag2"] = df["t_out"].shift(2 * lag_step)
        df["t_out_lag3"] = df["t_out"].shift(3 * lag_step)

    # First derivative dT/dt over 1 second window
    df["dT_OUT_dt"] = df["t_out"] - df["t_out_lag1"]
    
    # Quadratic non-linearity term
    df["t_out_sq"] = df["t_out"] ** 2

    # Super-phase categorization
    if "phase" in df.columns:
        df["super_phase"] = extract_phase_supercategories(df["phase"])
        df["is_hold"] = df.apply(is_temperature_holding, axis=1)

    # Drop warm-up rows before history buffer is full
    df = df.dropna(subset=["t_out_lag1", "t_out_lag2", "t_out_lag3", "dT_OUT_dt"]).reset_index(drop=True)

    return df


class ArduinoCircularBufferSimulator:
    """
    Exact Python simulation of the 31-slot circular buffer implemented in Arduino C++ firmware.
    Buffer updates every 100ms and extracts 1s, 2s, and 3s lags.
    """

    def __init__(self, buffer_size: int = 31):
        self.buffer_size = buffer_size
        self.buffer = [float("nan")] * buffer_size
        self.head = -1
        self.count = 0

    def update(self, s1: float) -> Optional[dict]:
        """
        Push new sample to circular buffer and return computed lag features.
        """
        if np.isnan(s1):
            return None

        self.head = (self.head + 1) % self.buffer_size
        self.buffer[self.head] = float(s1)

        if self.count < self.buffer_size:
            self.count += 1

        # Check if buffer is fully warmed up
        if self.count >= self.buffer_size:
            idx1 = (self.head - 10 + self.buffer_size) % self.buffer_size  # 1.0s ago
            idx2 = (self.head - 20 + self.buffer_size) % self.buffer_size  # 2.0s ago
            idx3 = (self.head - 30 + self.buffer_size) % self.buffer_size  # 3.0s ago

            lag1 = self.buffer[idx1]
            lag2 = self.buffer[idx2]
            lag3 = self.buffer[idx3]
            dt = s1 - lag1

            return {
                "t_out": s1,
                "dT_OUT_dt": dt,
                "t_out_lag1": lag1,
                "t_out_lag2": lag2,
                "t_out_lag3": lag3,
            }
        return None
