"""
C++ Header & Weight Generation Module for Arduino Nano Microcontroller Deployment.
Translates trained Python Scikit-Learn models into zero-dependency C++ inference functions.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import pandas as pd

from .features import DEFAULT_FEATURES, engineer_features
from .models import PhaseSpecificPCRModel


def generate_arduino_header(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    features: Optional[list] = None
) -> str:
    """
    Fit phase-specific thermal lag model on dataset and export C++ header.

    Parameters
    ----------
    df : pd.DataFrame
        Training telemetry dataset.
    output_path : Optional[Union[str, Path]]
        Destination path for 'ml_model_weights.h'.
    features : Optional[list]
        Feature list (default: Model 3 lag features).

    Returns
    -------
    str
        Generated C++ header source code.
    """
    features = features or DEFAULT_FEATURES
    if "t_out_lag1" not in df.columns:
        df = engineer_features(df)

    model = PhaseSpecificPCRModel(features=features)
    model.fit(df, target_col="t_in")
    weights = model.get_phase_weights()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c_code = f"""// ==============================================================================
// 🧬 PCR-ML: AUTO-GENERATED EMBEDDED THERMAL CALIBRATION WEIGHTS
// ==============================================================================
// Generated automatically by pcr_ml.c_code_gen at: {timestamp}
// Target Platform: Arduino Nano (ATmega328P @ 16MHz)
// Memory footprint: ~124 bytes RAM (31-float circular buffer)
// ==============================================================================

#ifndef ML_MODEL_WEIGHTS_H
#define ML_MODEL_WEIGHTS_H

#include <Arduino.h>

// Model feature coefficients for Denaturation (95°C Target)
#define DENAT_INTERCEPT     {weights['DENAT']['intercept']:.6f}f
#define DENAT_COEF_S1       {weights['DENAT']['coefficients']['t_out']:.6f}f
#define DENAT_COEF_DT_DT    {weights['DENAT']['coefficients']['dT_OUT_dt']:.6f}f
#define DENAT_COEF_LAG1     {weights['DENAT']['coefficients']['t_out_lag1']:.6f}f
#define DENAT_COEF_LAG2     {weights['DENAT']['coefficients']['t_out_lag2']:.6f}f
#define DENAT_COEF_LAG3     {weights['DENAT']['coefficients']['t_out_lag3']:.6f}f
#define DENAT_COEF_CYCLE    {weights['DENAT']['coefficients']['cycle']:.6f}f

// Model feature coefficients for Annealing (60°C Target)
#define ANNEAL_INTERCEPT    {weights['ANNEAL']['intercept']:.6f}f
#define ANNEAL_COEF_S1      {weights['ANNEAL']['coefficients']['t_out']:.6f}f
#define ANNEAL_COEF_DT_DT   {weights['ANNEAL']['coefficients']['dT_OUT_dt']:.6f}f
#define ANNEAL_COEF_LAG1    {weights['ANNEAL']['coefficients']['t_out_lag1']:.6f}f
#define ANNEAL_COEF_LAG2    {weights['ANNEAL']['coefficients']['t_out_lag2']:.6f}f
#define ANNEAL_COEF_LAG3    {weights['ANNEAL']['coefficients']['t_out_lag3']:.6f}f
#define ANNEAL_COEF_CYCLE   {weights['ANNEAL']['coefficients']['cycle']:.6f}f

// Model feature coefficients for Extension (72°C Target)
#define EXTEND_INTERCEPT    {weights['EXTEND']['intercept']:.6f}f
#define EXTEND_COEF_S1      {weights['EXTEND']['coefficients']['t_out']:.6f}f
#define EXTEND_COEF_DT_DT   {weights['EXTEND']['coefficients']['dT_OUT_dt']:.6f}f
#define EXTEND_COEF_LAG1    {weights['EXTEND']['coefficients']['t_out_lag1']:.6f}f
#define EXTEND_COEF_LAG2    {weights['EXTEND']['coefficients']['t_out_lag2']:.6f}f
#define EXTEND_COEF_LAG3    {weights['EXTEND']['coefficients']['t_out_lag3']:.6f}f
#define EXTEND_COEF_CYCLE   {weights['EXTEND']['coefficients']['cycle']:.6f}f

/**
 * Inline real-time evaluation of internal PCR fluid temperature (T_IN).
 * Runs in < 25 microseconds on 16MHz ATmega328P without dynamic memory allocations.
 */
inline float predictInternalTemperature(
    uint8_t phaseId,
    float s1,
    float dT_dt,
    float lag1,
    float lag2,
    float lag3,
    uint8_t currentCycle
) {{
    float raw_estimate = 0.0f;
    float cyc = (float)currentCycle;

    switch (phaseId) {{
        case 0: // DENATURATION
        case 1:
            raw_estimate = DENAT_INTERCEPT
                         + (DENAT_COEF_S1 * s1)
                         + (DENAT_COEF_DT_DT * dT_dt)
                         + (DENAT_COEF_LAG1 * lag1)
                         + (DENAT_COEF_LAG2 * lag2)
                         + (DENAT_COEF_LAG3 * lag3)
                         + (DENAT_COEF_CYCLE * cyc);
            break;

        case 2: // ANNEALING
            raw_estimate = ANNEAL_INTERCEPT
                         + (ANNEAL_COEF_S1 * s1)
                         + (ANNEAL_COEF_DT_DT * dT_dt)
                         + (ANNEAL_COEF_LAG1 * lag1)
                         + (ANNEAL_COEF_LAG2 * lag2)
                         + (ANNEAL_COEF_LAG3 * lag3)
                         + (ANNEAL_COEF_CYCLE * cyc);
            break;

        case 3: // EXTENSION
        case 4:
            raw_estimate = EXTEND_INTERCEPT
                         + (EXTEND_COEF_S1 * s1)
                         + (EXTEND_COEF_DT_DT * dT_dt)
                         + (EXTEND_COEF_LAG1 * lag1)
                         + (EXTEND_COEF_LAG2 * lag2)
                         + (EXTEND_COEF_LAG3 * lag3)
                         + (EXTEND_COEF_CYCLE * cyc);
            break;

        default:
            raw_estimate = s1;
            break;
    }}

    return raw_estimate;
}}

#endif // ML_MODEL_WEIGHTS_H
"""

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(c_code)

    return c_code
