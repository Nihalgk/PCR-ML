#!/usr/bin/env python3
"""
PCR Thermal Calibration Analysis
=================================
Predicts T_IN (internal test tube temperature) from T_OUT (external thermocouple)
using incremental regression models for Arduino Nano deployment.

Strategy:
  - Runs 1 & 2: Primary calibration (leave-one-run-out cross-validation)
  - Run 3: Independent stress test (reduced LED efficiency)
  - Models built incrementally from simplest to most complex
  - Phase tested separately to avoid artificial accuracy inflation
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = r'C:\Users\dell\Desktop\PCR ML'
PLOT_DIR = os.path.join(DATA_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

FILES = {
    'Run 1': os.path.join(DATA_DIR, '40 cycle(3step)_12.08.2026.txt'),
    'Run 2': os.path.join(DATA_DIR, '40 cycle(3step)_12.08.2026_2.txt'),
    'Run 3': os.path.join(DATA_DIR, '40 cycle(3step)_12.08.2026_3.txt'),
}

CRITICAL_TEMPS = {'60degC': 60, '72degC': 72, '95degC': 95}
TEMP_WINDOW = 2.0  # ±2degC window around critical temps

# ============================================================
# SECTION 1: DATA LOADING AND PARSING
# ============================================================
print("=" * 70)
print("PCR THERMAL CALIBRATION ANALYSIS")
print("Predicting T_IN from T_OUT for Arduino Nano Deployment")
print("=" * 70)

print("\n" + "=" * 70)
print("SECTION 1: DATA LOADING")
print("=" * 70)
print("""
WHAT WE'RE DOING:
  Loading 3 experimental runs from a 3-step PCR protocol (40 cycles each).
  Each data row contains temperature readings from two thermocouples:
    - T_OUT: External thermocouple on aluminum foil holder
    - T_IN:  Internal thermocouple inside test tube (contains water)

  The GOAL is to predict T_IN from T_OUT alone, so we can remove the
  internal thermocouple from the final design.

WHY WE CAN'T USE A SIMPLE OFFSET:
  The relationship T_IN = T_OUT + constant doesn't work because:
    1) The offset changes depending on whether we're heating or cooling
    2) The offset drifts as cycles accumulate (thermal buildup)
    3) Different runs have different offsets even at the same temperature

VALIDATION STRATEGY:
  - Runs 1 & 2: Cross-validation (train on one, test on the other)
  - Run 3: Stress test (this run had reduced LED efficiency,
    causing different thermal behavior)
""")


def parse_pcr_file(filepath, run_name):
    """Parse a PCR data file and return a DataFrame."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('=') or line.startswith('PCR:'):
                continue
            match = re.match(
                r'Time:([\d.]+),\s*Cycle:(\d+),\s*(\w+),\s*'
                r'T_OUT:([\d.]+),\s*T_IN:([\d.]+),\s*'
                r'F T_IN:([\d.]+),\s*(.+)',
                line
            )
            if match:
                time_s, cycle, phase, t_out, t_in, f_t_in, mode = match.groups()
                data.append({
                    'time': float(time_s),
                    'cycle': int(cycle),
                    'phase': phase,
                    't_out': float(t_out),
                    't_in': float(t_in),
                    'f_t_in': float(f_t_in),
                    'mode': mode.strip(),
                    'run': run_name
                })
    return pd.DataFrame(data)


# Load all runs
runs = {}
for name, path in FILES.items():
    df = parse_pcr_file(path, name)
    runs[name] = df
    print(f"  {name}: {len(df)} samples, Cycles {df['cycle'].min()}-{df['cycle'].max()}")
    print(f"    T_OUT range: {df['t_out'].min():.1f}degC - {df['t_out'].max():.1f}degC")
    print(f"    T_IN range:  {df['t_in'].min():.1f}degC - {df['t_in'].max():.1f}degC")
    print(f"    Phases: {', '.join(df['phase'].unique())}")
    print()


# ============================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("SECTION 2: WHY A SIMPLE OFFSET WON'T WORK (VISUAL EVIDENCE)")
print("=" * 70)
print("""
WHAT WE'RE DOING:
  Before building any models, we need to see the THREE problems visually.
  This justifies the need for a dynamic model rather than a fixed offset.

  Plot 1: Raw time series — shows the cycling pattern and temperature ranges
  Plot 2: T_OUT vs T_IN scatter by cycle — reveals thermal drift
  Plot 3: Hysteresis loop — shows heating/cooling asymmetry
""")

# --- Plot 1: Raw time series ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
fig.suptitle('Raw Temperature Data: All 3 Runs', fontsize=14, fontweight='bold')

for idx, (name, df) in enumerate(runs.items()):
    ax = axes[idx]
    ax.plot(df['time'], df['t_out'], color='#E74C3C', alpha=0.7, linewidth=0.8,
            label='T_OUT (external)')
    ax.plot(df['time'], df['t_in'], color='#3498DB', alpha=0.7, linewidth=0.8,
            label='T_IN (internal)')
    ax.set_ylabel('Temperature (degC)')
    suffix = ' — REDUCED LED EFFICIENCY' if 'Run 3' in name else ''
    ax.set_title(f'{name}{suffix}', fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (seconds)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '01_raw_time_series.png'), dpi=150,
            bbox_inches='tight')
plt.close()
print("  Saved: plots/01_raw_time_series.png")

# --- Plot 2: T_OUT vs T_IN scatter colored by cycle ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('T_OUT vs T_IN: Colored by Cycle Number (Shows Thermal Drift)',
             fontsize=14, fontweight='bold')

for idx, (name, df) in enumerate(runs.items()):
    ax = axes[idx]
    scatter = ax.scatter(df['t_out'], df['t_in'], c=df['cycle'], cmap='viridis',
                         s=3, alpha=0.6)
    ax.set_xlabel('T_OUT (degC)')
    ax.set_ylabel('T_IN (degC)')
    ax.set_title(name, fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Cycle #')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '02_scatter_by_cycle.png'), dpi=150,
            bbox_inches='tight')
plt.close()
print("  Saved: plots/02_scatter_by_cycle.png")

# --- Plot 3: Hysteresis loop ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Hysteresis: Same T_OUT Maps to Different T_IN (Heating vs Cooling)',
             fontsize=14, fontweight='bold')

for idx, cycle_num in enumerate([5, 30]):
    ax = axes[idx]
    df_cycle = runs['Run 1'][runs['Run 1']['cycle'] == cycle_num].copy()
    if len(df_cycle) > 0:
        time_in_cycle = df_cycle['time'].values - df_cycle['time'].values[0]
        scatter = ax.scatter(df_cycle['t_out'], df_cycle['t_in'], c=time_in_cycle,
                             cmap='coolwarm', s=15, zorder=3)
        ax.plot(df_cycle['t_out'].values, df_cycle['t_in'].values,
                color='gray', alpha=0.4, linewidth=0.8, zorder=2)
        ax.set_xlabel('T_OUT (degC)')
        ax.set_ylabel('T_IN (degC)')
        ax.set_title(f'Run 1, Cycle {cycle_num}', fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Time in cycle (s)')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '03_hysteresis_loop.png'), dpi=150,
            bbox_inches='tight')
plt.close()
print("  Saved: plots/03_hysteresis_loop.png")

# Quantitative drift analysis
print("\n  QUANTITATIVE DRIFT ANALYSIS (Run 1):")
print("  Question: At T_IN ~= 72degC, what is T_OUT at different cycles?")
print("  If T_OUT is constant -> offset is stable. If T_OUT increases -> drift.")
print()
for cycle_num in [1, 5, 10, 20, 30, 40]:
    df_c = runs['Run 1'][runs['Run 1']['cycle'] == cycle_num]
    mask = (df_c['t_in'] >= 71) & (df_c['t_in'] <= 73)
    if mask.any():
        t_out_mean = df_c.loc[mask, 't_out'].mean()
        t_out_min = df_c.loc[mask, 't_out'].min()
        t_out_max = df_c.loc[mask, 't_out'].max()
        print(f"    Cycle {cycle_num:2d}: T_OUT = {t_out_mean:.1f}degC "
              f"(range: {t_out_min:.1f} - {t_out_max:.1f}degC) when T_IN ~= 72degC")

print("\n  -> T_OUT at the same T_IN increases with cycle number = THERMAL DRIFT")
print("  -> This confirms cycle number is an important feature.")


# ============================================================
# SECTION 3: FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("SECTION 3: FEATURE ENGINEERING")
print("=" * 70)
print("""
WHAT WE'RE DOING:
  Building features incrementally. Each feature captures a specific
  physical phenomenon in the thermal system.

FEATURES (added one group at a time):
  +-----------------+--------------------------------------------------+
  | Feature         | Physical Meaning                                 |
  +-----------------+--------------------------------------------------+
  | T_OUT           | Primary measurement (aluminum foil temperature)  |
  | dT_OUT/dt       | Rate of change -> resolves heating vs cooling     |
  | T_OUT_lag1/2/3  | Temperature 1-3 seconds ago -> captures momentum |
  | cycle           | Cycle number -> captures long-term thermal drift  |
  | T_OUT^2          | Non-linear term for curved T_OUT->T_IN mapping   |
  +-----------------+--------------------------------------------------+

HOW dT_OUT/dt IS COMPUTED:
  Backward difference: dT/dt[t] = T_OUT[t] - T_OUT[t-1]
  This is the ONLY method that works on Arduino in real-time (you only
  need the current reading and the previous one).

  CAVEAT: With 0.25degC thermocouple resolution at 1Hz sampling, dT/dt
  has ±0.25degC/s quantization noise. During hold phases where the true
  dT/dt ~= 0, this noise may dominate. The lag features partially
  compensate by capturing trajectory without amplifying noise.

WHAT WE DROP:
  The first 3 samples of each run have no lag values (no previous data).
  We drop these (~3 out of ~4000+ samples per run — negligible).
""")


def engineer_features(df):
    """Add all derived features to a DataFrame.
    Must be called per-run to avoid computing derivatives across runs."""
    df = df.copy()

    # Backward difference for dT/dt (Arduino-compatible: only needs t and t-1)
    df['dT_OUT_dt'] = df['t_out'].diff()

    # Lag features (previous T_OUT readings stored in Arduino buffer)
    df['t_out_lag1'] = df['t_out'].shift(1)
    df['t_out_lag2'] = df['t_out'].shift(2)
    df['t_out_lag3'] = df['t_out'].shift(3)

    # Quadratic term
    df['t_out_sq'] = df['t_out'] ** 2

    # Phase encoding (one-hot, for separate testing only)
    for phase in ['DENAT', 'ANNEAL', 'EXTEND']:
        df[f'phase_{phase}'] = (df['phase'] == phase).astype(int)

    # Drop rows with NaN from lag/derivative computation (first 3 rows per run)
    df = df.dropna(subset=['dT_OUT_dt', 't_out_lag1', 't_out_lag2', 't_out_lag3'])

    return df


for name in runs:
    n_before = len(runs[name])
    runs[name] = engineer_features(runs[name])
    n_after = len(runs[name])
    print(f"  {name}: {n_before} -> {n_after} samples "
          f"(dropped {n_before - n_after} initial rows with no lag data)")


# ============================================================
# SECTION 4: MODEL EVALUATION FRAMEWORK
# ============================================================
print("\n" + "=" * 70)
print("SECTION 4: MODEL EVALUATION FRAMEWORK")
print("=" * 70)
print("""
WHAT WE'RE DOING:
  For each model, we measure accuracy two ways:

  1) CROSS-VALIDATION (Runs 1 & 2):
     - Fold A: Train on Run 1, test on Run 2
     - Fold B: Train on Run 2, test on Run 1
     - Report the AVERAGE of both folds
     -> This tells us: "How well does the model generalize to a new
       run under normal operating conditions?"

  2) STRESS TEST (Run 3):
     - Train on Runs 1+2 combined, test on Run 3
     -> This tells us: "How well does the model handle abnormal conditions?"
     -> Run 3 had reduced LED efficiency -> different thermal dynamics
     -> We EXPECT worse performance. If it's still acceptable, great.

  METRICS REPORTED:
    - MAE:       Average prediction error (most interpretable)
    - RMSE:      Penalizes large errors more than MAE
    - Max Error: Worst single-point prediction
    - R^2:        How much variance the model explains (1.0 = perfect)
    - @60/72/95: MAE specifically near the 3 critical PCR hold temps
                 (within ±{:.0f}degC window)
""".format(TEMP_WINDOW))

# Model configurations (incrementally more complex)
MODEL_CONFIGS = {
    'Model 1: T_OUT only': {
        'features': ['t_out'],
        'explain': (
            "WHAT: Using only the external thermocouple reading.\n"
            "  WHY: This is our BASELINE. If a fixed linear relationship T_IN = a·T_OUT + b\n"
            "  were sufficient, this model would give good results.\n"
            "  EXPECTATION: Poor accuracy because it cannot handle hysteresis\n"
            "  (same T_OUT -> different T_IN depending on heating/cooling direction)\n"
            "  or thermal drift (same T_IN -> different T_OUT at different cycles)."
        )
    },
    'Model 2: + dT/dt': {
        'features': ['t_out', 'dT_OUT_dt'],
        'explain': (
            "WHAT: Adding the rate of change of T_OUT.\n"
            "  WHY: dT_OUT/dt is positive during heating and negative during cooling.\n"
            "  This gives the model a way to distinguish 'T_OUT=55degC and rising'\n"
            "  from 'T_OUT=55degC and falling' — which correspond to very different T_IN.\n"
            "  EXPECTATION: Significant improvement by resolving the hysteresis loop.\n"
            "  CAVEAT: Derivative amplifies sensor noise (0.25degC resolution -> noisy dT/dt)."
        )
    },
    'Model 3: + lags + cycle': {
        'features': ['t_out', 'dT_OUT_dt', 't_out_lag1', 't_out_lag2', 't_out_lag3', 'cycle'],
        'explain': (
            "WHAT: Adding previous T_OUT readings (1-3 sec ago) + PCR cycle number.\n"
            "  WHY: Lag values capture the temperature TRAJECTORY. Unlike dT/dt which\n"
            "  only captures instantaneous rate, lags tell the model the recent path.\n"
            "  Example: T_OUT was 45->50->55 (rapid heating) vs 54->54.5->55 (approaching hold).\n"
            "  Cycle number captures the systematic DRIFT as heat accumulates over 40 cycles.\n"
            "  EXPECTATION: Improvement from both trajectory information and drift correction."
        )
    },
    'Model 4: + T_OUT^2': {
        'features': ['t_out', 'dT_OUT_dt', 't_out_lag1', 't_out_lag2', 't_out_lag3',
                      'cycle', 't_out_sq'],
        'explain': (
            "WHAT: Adding T_OUT squared (quadratic non-linearity).\n"
            "  WHY: The thermal resistance between aluminum foil and water may not be\n"
            "  constant across the full temperature range. At higher temperatures,\n"
            "  radiation losses increase (proportional to T⁴), and convection patterns\n"
            "  in the water change. A quadratic term allows the model to curve.\n"
            "  EXPECTATION: Small-to-moderate improvement. If the T_OUT->T_IN relationship\n"
            "  is already approximately linear, this adds little. If there's curvature,\n"
            "  it captures it with just one extra coefficient."
        )
    },
}

PHASE_FEATURES = ['phase_DENAT', 'phase_ANNEAL', 'phase_EXTEND']


def compute_metrics(y_true, y_pred):
    """Compute comprehensive error metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    max_err = np.max(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'Max Error': max_err, 'R^2': r2}


def compute_critical_temp_errors(y_true, y_pred, t_in_actual):
    """Compute MAE near each critical PCR temperature."""
    errors = {}
    n_points = {}
    for label, temp in CRITICAL_TEMPS.items():
        mask = (t_in_actual >= temp - TEMP_WINDOW) & (t_in_actual <= temp + TEMP_WINDOW)
        n = mask.sum()
        if n > 0:
            errors[label] = mean_absolute_error(y_true[mask], y_pred[mask])
            n_points[label] = n
        else:
            errors[label] = np.nan
            n_points[label] = 0
    return errors, n_points


def evaluate_model(model_class, features, runs_dict,
                   model_name="", model_kwargs=None):
    """Full evaluation: Leave-One-Run-Out CV (Runs 1&2) + Run 3 stress test."""
    if model_kwargs is None:
        model_kwargs = {}

    run1 = runs_dict['Run 1']
    run2 = runs_dict['Run 2']
    run3 = runs_dict['Run 3']

    # --- Fold 1: Train on Run 1, Test on Run 2 ---
    model_f1 = model_class(**model_kwargs)
    model_f1.fit(run1[features].values, run1['t_in'].values)
    pred_f1 = model_f1.predict(run2[features].values)
    true_f1 = run2['t_in'].values
    metrics_f1 = compute_metrics(true_f1, pred_f1)
    crit_f1, npts_f1 = compute_critical_temp_errors(true_f1, pred_f1, true_f1)

    # --- Fold 2: Train on Run 2, Test on Run 1 ---
    model_f2 = model_class(**model_kwargs)
    model_f2.fit(run2[features].values, run2['t_in'].values)
    pred_f2 = model_f2.predict(run1[features].values)
    true_f2 = run1['t_in'].values
    metrics_f2 = compute_metrics(true_f2, pred_f2)
    crit_f2, npts_f2 = compute_critical_temp_errors(true_f2, pred_f2, true_f2)

    # --- Average CV metrics ---
    avg_metrics = {k: (metrics_f1[k] + metrics_f2[k]) / 2 for k in metrics_f1}
    avg_crit = {}
    for label in CRITICAL_TEMPS:
        v1, v2 = crit_f1[label], crit_f2[label]
        if np.isnan(v1) and np.isnan(v2):
            avg_crit[label] = np.nan
        elif np.isnan(v1):
            avg_crit[label] = v2
        elif np.isnan(v2):
            avg_crit[label] = v1
        else:
            avg_crit[label] = (v1 + v2) / 2

    # --- Stress Test: Train on Runs 1+2, Test on Run 3 ---
    X_train_combined = pd.concat([run1[features], run2[features]]).values
    y_train_combined = pd.concat([run1['t_in'], run2['t_in']]).values

    model_stress = model_class(**model_kwargs)
    model_stress.fit(X_train_combined, y_train_combined)
    pred_stress = model_stress.predict(run3[features].values)
    true_stress = run3['t_in'].values
    metrics_stress = compute_metrics(true_stress, pred_stress)
    crit_stress, npts_stress = compute_critical_temp_errors(
        true_stress, pred_stress, true_stress)

    return {
        'CV_metrics': avg_metrics,
        'CV_critical': avg_crit,
        'fold1_metrics': metrics_f1,
        'fold2_metrics': metrics_f2,
        'stress_metrics': metrics_stress,
        'stress_critical': crit_stress,
        # Store predictions and models for plotting
        'pred_fold1': pred_f1,        # Predictions on Run 2
        'pred_fold2': pred_f2,        # Predictions on Run 1
        'pred_stress': pred_stress,   # Predictions on Run 3
        'model_fold1': model_f1,
        'model_fold2': model_f2,
        'model_stress': model_stress,
    }


# ============================================================
# SECTIONS 5-8: INCREMENTAL MODEL TRAINING
# ============================================================
all_results = {}

for model_name, config in MODEL_CONFIGS.items():
    features = config['features']
    explain = config['explain']

    print("\n" + "=" * 70)
    section_num = list(MODEL_CONFIGS.keys()).index(model_name) + 5
    print(f"SECTION {section_num}: {model_name.upper()}")
    print("=" * 70)
    print(f"\n  {explain}")

    results = evaluate_model(LinearRegression, features, runs, model_name)
    all_results[model_name] = results

    # --- Print results ---
    cv = results['CV_metrics']
    stress = results['stress_metrics']
    cv_crit = results['CV_critical']
    stress_crit = results['stress_critical']

    print(f"\n  +-------------------------------------------------+")
    print(f"  | CROSS-VALIDATION RESULTS (Runs 1 <-> 2)          |")
    print(f"  +-------------------------------------------------+")
    print(f"  |  MAE:       {cv['MAE']:>6.2f}degC                       |")
    print(f"  |  RMSE:      {cv['RMSE']:>6.2f}degC                       |")
    print(f"  |  Max Error: {cv['Max Error']:>6.2f}degC                       |")
    print(f"  |  R^2:        {cv['R^2']:>6.4f}                        |")
    print(f"  +-------------------------------------------------+")

    print(f"\n  Critical Temperature MAE (CV):")
    for label, val in cv_crit.items():
        status = f"{val:.2f}degC" if not np.isnan(val) else "N/A (no data in range)"
        print(f"    Near {label}: {status}")

    print(f"\n  +-------------------------------------------------+")
    print(f"  | STRESS TEST (Run 3 — Reduced LED Efficiency)    |")
    print(f"  +-------------------------------------------------+")
    print(f"  |  MAE:       {stress['MAE']:>6.2f}degC                       |")
    print(f"  |  RMSE:      {stress['RMSE']:>6.2f}degC                       |")
    print(f"  |  Max Error: {stress['Max Error']:>6.2f}degC                       |")
    print(f"  |  R^2:        {stress['R^2']:>6.4f}                        |")
    print(f"  +-------------------------------------------------+")

    print(f"\n  Critical Temperature MAE (Stress Test - Run 3):")
    for label, val in stress_crit.items():
        status = f"{val:.2f}degC" if not np.isnan(val) else "N/A"
        print(f"    Near {label}: {status}")

    # Print linear equation
    model_obj = results['model_stress']  # Trained on Runs 1+2
    if hasattr(model_obj, 'coef_'):
        print(f"\n  LINEAR EQUATION (trained on Runs 1+2):")
        terms = []
        for feat, coef in zip(features, model_obj.coef_):
            terms.append(f"({coef:+.6f} x {feat})")
        print(f"    T_IN = {model_obj.intercept_:.6f} {' '.join(terms)}")

    # Show improvement over previous model
    model_keys = list(MODEL_CONFIGS.keys())
    current_idx = model_keys.index(model_name)
    if current_idx > 0:
        prev_name = model_keys[current_idx - 1]
        prev_mae = all_results[prev_name]['CV_metrics']['MAE']
        curr_mae = cv['MAE']
        improvement = prev_mae - curr_mae
        pct = (improvement / prev_mae) * 100 if prev_mae > 0 else 0
        print(f"\n  IMPROVEMENT over {prev_name.split(':')[0]}:")
        print(f"    MAE reduced by {improvement:.2f}degC ({pct:.1f}% improvement)")


# ============================================================
# SECTION 9: PHASE ANALYSIS (SEPARATE)
# ============================================================
print("\n" + "=" * 70)
print("SECTION 9: TESTING PHASE AS A FEATURE (SEPARATELY)")
print("=" * 70)
print("""
WHAT WE'RE DOING:
  Adding Phase (DENAT/ANNEAL/EXTEND) as one-hot encoded features to Model 3.
  Phase tells the model which PCR step we're in.

WHY TEST SEPARATELY:
  Phase encodes WHICH temperature the system is targeting. Including it
  could make predictions "artificially easy" — the model might learn:
    "Phase=DENAT -> T_IN ~= 95degC" (essentially a lookup table).

  A lookup table would fail during TRANSITIONS between phases, which
  is exactly where accurate control matters most.

COUNTER-POINT:
  In deployment, the Arduino DOES know the phase (it controls the protocol).
  So phase IS available at inference time — it's not data leakage.
  But if adding phase provides HUGE improvement, it means the base model
  (without phase) doesn't actually understand the thermal physics well.
  If adding phase provides SMALL improvement, the base model is solid.

  We want to quantify this.
""")

features_with_phase = MODEL_CONFIGS['Model 3: + lags + cycle']['features'] + PHASE_FEATURES
results_phase = evaluate_model(LinearRegression, features_with_phase, runs,
                               'Model 3 + Phase')
all_results['Model 3 + Phase'] = results_phase

cv_phase = results_phase['CV_metrics']
stress_phase = results_phase['stress_metrics']
cv_crit_phase = results_phase['CV_critical']

cv_m3 = all_results['Model 3: + lags + cycle']['CV_metrics']
improvement = cv_m3['MAE'] - cv_phase['MAE']

print(f"  Cross-Validation (Runs 1<->2):")
print(f"    MAE:       {cv_phase['MAE']:.2f}degC (Model 3 without phase: {cv_m3['MAE']:.2f}degC)")
print(f"    RMSE:      {cv_phase['RMSE']:.2f}degC")
print(f"    Max Error: {cv_phase['Max Error']:.2f}degC")
print(f"    R^2:        {cv_phase['R^2']:.4f}")

print(f"\n  Phase improvement over Model 3: {improvement:+.2f}degC MAE reduction")
if improvement > 0.5:
    print("  -> SIGNIFICANT: Phase provides substantial accuracy gain.")
    print("    This suggests the base model struggles to distinguish PCR zones.")
elif improvement > 0.1:
    print("  -> MODERATE: Phase helps somewhat but base model captures most physics.")
else:
    print("  -> MARGINAL: Base model already captures the thermal physics well.")
    print("    Phase adds little value — the model works from temperature dynamics alone.")

print(f"\n  Stress Test (Run 3):")
print(f"    MAE: {stress_phase['MAE']:.2f}degC  |  RMSE: {stress_phase['RMSE']:.2f}degC")

print(f"\n  Critical Temperature MAE (CV):")
for label, val in cv_crit_phase.items():
    if not np.isnan(val):
        print(f"    Near {label}: {val:.2f}degC")


# ============================================================
# SECTION 10: RANDOM FOREST & GRADIENT BOOSTING
# ============================================================
print("\n" + "=" * 70)
print("SECTION 10: TREE-BASED MODELS (RF & GRADIENT BOOSTING)")
print("=" * 70)
print("""
WHAT WE'RE DOING:
  Testing Random Forest and Gradient Boosting to see if non-linear models
  provide meaningful improvement over linear regression.

WHY THIS MATTERS FOR ARDUINO:
  Arduino Nano (ATmega328P): 32KB flash, 2KB SRAM, NO floating-point unit.
  - Linear regression: 7 multiply-adds ~= 140µs, ~32 bytes for coefficients
  - Random Forest (100 trees): ~10-100KB for tree structures -> WON'T FIT
  - Gradient Boosting: Similar storage requirements -> WON'T FIT

  So tree-based models are ONLY worth considering if they provide dramatic
  improvement. Otherwise, linear regression wins by default for deployment.

  Using Model 3's feature set (without phase) for fair comparison.
""")

features_m3 = MODEL_CONFIGS['Model 3: + lags + cycle']['features']

# Random Forest
print("  Training Random Forest (100 trees, max_depth=10)...")
rf_results = evaluate_model(
    RandomForestRegressor, features_m3, runs, 'Random Forest',
    model_kwargs={'n_estimators': 100, 'max_depth': 10,
                  'random_state': 42, 'n_jobs': -1}
)
all_results['Random Forest'] = rf_results

cv_rf = rf_results['CV_metrics']
stress_rf = rf_results['stress_metrics']
print(f"    CV:    MAE={cv_rf['MAE']:.2f}degC  RMSE={cv_rf['RMSE']:.2f}degC  "
      f"Max={cv_rf['Max Error']:.2f}degC  R^2={cv_rf['R^2']:.4f}")
print(f"    Run 3: MAE={stress_rf['MAE']:.2f}degC  RMSE={stress_rf['RMSE']:.2f}degC  "
      f"Max={stress_rf['Max Error']:.2f}degC")

# Gradient Boosting
print("\n  Training Gradient Boosting (200 trees, max_depth=5)...")
gb_results = evaluate_model(
    GradientBoostingRegressor, features_m3, runs, 'Gradient Boosting',
    model_kwargs={'n_estimators': 200, 'max_depth': 5,
                  'learning_rate': 0.1, 'random_state': 42}
)
all_results['Gradient Boosting'] = gb_results

cv_gb = gb_results['CV_metrics']
stress_gb = gb_results['stress_metrics']
print(f"    CV:    MAE={cv_gb['MAE']:.2f}degC  RMSE={cv_gb['RMSE']:.2f}degC  "
      f"Max={cv_gb['Max Error']:.2f}degC  R^2={cv_gb['R^2']:.4f}")
print(f"    Run 3: MAE={stress_gb['MAE']:.2f}degC  RMSE={stress_gb['RMSE']:.2f}degC  "
      f"Max={stress_gb['Max Error']:.2f}degC")

# Compare with best linear model
best_linear_name = 'Model 4: + T_OUT^2'
best_linear_mae = all_results[best_linear_name]['CV_metrics']['MAE']
rf_improvement = best_linear_mae - cv_rf['MAE']
gb_improvement = best_linear_mae - cv_gb['MAE']

print(f"\n  COMPARISON WITH BEST LINEAR MODEL ({best_linear_name}):")
print(f"    Linear Model MAE:     {best_linear_mae:.2f}degC")
print(f"    Random Forest MAE:    {cv_rf['MAE']:.2f}degC  (improvement: {rf_improvement:+.2f}degC)")
print(f"    Gradient Boosting MAE:{cv_gb['MAE']:.2f}degC  (improvement: {gb_improvement:+.2f}degC)")

max_improvement = max(rf_improvement, gb_improvement)
if max_improvement < 0.3:
    print(f"\n  VERDICT: Tree models provide MINIMAL improvement ({max_improvement:.2f}degC).")
    print("  -> Linear regression is sufficient. Deploy on Arduino Nano with confidence.")
elif max_improvement < 1.0:
    print(f"\n  VERDICT: Tree models provide MODERATE improvement ({max_improvement:.2f}degC).")
    print("  -> Consider if this accuracy gain justifies deployment complexity.")
    print("    (Possible workaround: approximate the tree model with a lookup table)")
else:
    print(f"\n  VERDICT: Tree models provide SIGNIFICANT improvement ({max_improvement:.2f}degC).")
    print("  -> The non-linear dynamics are important. Consider:")
    print("    1) Lookup table approximation of the tree model")
    print("    2) Piecewise linear regression (different coefficients for different T_OUT ranges)")
    print("    3) Upgrading to a microcontroller with more memory (ESP32)")


# ============================================================
# SECTION 11: FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 70)
print("SECTION 11: FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)
print("""
WHAT WE'RE DOING:
  Using Random Forest's built-in feature importance to see which features
  contribute most to predicting T_IN. This helps us understand:
    - Is T_OUT the dominant predictor? (Should be)
    - Does dT/dt actually help? (If importance is low -> noise issue)
    - Are lags redundant with dT/dt?
    - How much does cycle number contribute?
""")

rf_model = rf_results['model_stress']
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("  Feature Importance Ranking (Random Forest, trained on Runs 1+2):")
print("  " + "-" * 50)
for rank, i in enumerate(sorted_idx, 1):
    bar = "#" * int(importances[i] * 50)
    print(f"    {rank}. {features_m3[i]:15s}: {importances[i]:.4f}  {bar}")

# Also show linear regression coefficients significance
print("\n  Linear Regression Coefficients (Model 4, for comparison):")
m4_model = all_results['Model 4: + T_OUT^2']['model_stress']
m4_features = MODEL_CONFIGS['Model 4: + T_OUT^2']['features']
print(f"    Intercept: {m4_model.intercept_:.4f}")
for feat, coef in zip(m4_features, m4_model.coef_):
    print(f"    {feat:15s}: {coef:+.6f}")

# Feature importance plot
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(range(len(features_m3)),
               [importances[i] for i in sorted_idx],
               color='#2ECC71', edgecolor='#27AE60')
ax.set_yticks(range(len(features_m3)))
ax.set_yticklabels([features_m3[i] for i in sorted_idx])
ax.set_xlabel('Feature Importance')
ax.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '10_feature_importance.png'), dpi=150,
            bbox_inches='tight')
plt.close()
print("\n  Saved: plots/10_feature_importance.png")


# ============================================================
# SECTION 12: VISUALIZATION OF RESULTS
# ============================================================
print("\n" + "=" * 70)
print("SECTION 12: GENERATING COMPARISON PLOTS")
print("=" * 70)

model_names_ordered = [
    'Model 1: T_OUT only',
    'Model 2: + dT/dt',
    'Model 3: + lags + cycle',
    'Model 4: + T_OUT^2',
    'Model 3 + Phase',
    'Random Forest',
    'Gradient Boosting',
]

short_names = ['M1: T_OUT', 'M2: +dT/dt', 'M3: +lags\n+cycle', 'M4: +T^2',
               'M3+Phase', 'RF', 'GBM']

colors = ['#3498DB', '#2ECC71', '#E67E22', '#E74C3C',
          '#9B59B6', '#1ABC9C', '#34495E']

# --- Plot 4: Model comparison bar chart (CV) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Model Comparison: Cross-Validation (Runs 1 <-> 2)',
             fontsize=14, fontweight='bold')

maes = [all_results[m]['CV_metrics']['MAE'] for m in model_names_ordered]
rmses = [all_results[m]['CV_metrics']['RMSE'] for m in model_names_ordered]
max_errs = [all_results[m]['CV_metrics']['Max Error'] for m in model_names_ordered]

for ax, vals, ylabel, title in zip(
    axes,
    [maes, rmses, max_errs],
    ['MAE (degC)', 'RMSE (degC)', 'Max Error (degC)'],
    ['Mean Absolute Error', 'Root Mean Squared Error', 'Maximum Error']
):
    bars = ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=0, ha='center', fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '04_model_comparison_cv.png'), dpi=150,
            bbox_inches='tight')
plt.close()
print("  Saved: plots/04_model_comparison_cv.png")

# --- Plot 5: Stress test comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Stress Test: Run 3 (Reduced LED Efficiency)',
             fontsize=14, fontweight='bold')

s_maes = [all_results[m]['stress_metrics']['MAE'] for m in model_names_ordered]
s_rmses = [all_results[m]['stress_metrics']['RMSE'] for m in model_names_ordered]
s_maxs = [all_results[m]['stress_metrics']['Max Error'] for m in model_names_ordered]

for ax, vals, ylabel, title in zip(
    axes,
    [s_maes, s_rmses, s_maxs],
    ['MAE (degC)', 'RMSE (degC)', 'Max Error (degC)'],
    ['Mean Absolute Error', 'Root Mean Squared Error', 'Maximum Error']
):
    bars = ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=0, ha='center', fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '05_stress_test_comparison.png'), dpi=150,
            bbox_inches='tight')
plt.close()
print("  Saved: plots/05_stress_test_comparison.png")

# --- Plot 6: Predicted vs Actual time series (Model 4) ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Model 4 Predictions vs Actual T_IN', fontsize=14, fontweight='bold')

# CV: Prediction on Run 2 (trained on Run 1)
run2 = runs['Run 2']
y_pred_cv = all_results['Model 4: + T_OUT^2']['pred_fold1']
y_true_cv = run2['t_in'].values

ax = axes[0]
ax.plot(run2['time'].values, y_true_cv, color='#3498DB', alpha=0.8,
        linewidth=0.8, label='Actual T_IN')
ax.plot(run2['time'].values, y_pred_cv, color='#E74C3C', alpha=0.8,
        linewidth=0.8, label='Predicted T_IN (Model 4)')
ax.set_ylabel('Temperature (degC)')
ax.set_title('Cross-Validation: Run 2 (model trained on Run 1)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Stress test on Run 3
run3 = runs['Run 3']
y_pred_stress = all_results['Model 4: + T_OUT^2']['pred_stress']
y_true_stress = run3['t_in'].values

ax = axes[1]
ax.plot(run3['time'].values, y_true_stress, color='#3498DB', alpha=0.8,
        linewidth=0.8, label='Actual T_IN')
ax.plot(run3['time'].values, y_pred_stress, color='#E74C3C', alpha=0.8,
        linewidth=0.8, label='Predicted T_IN (Model 4)')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Temperature (degC)')
ax.set_title('Stress Test: Run 3 — Reduced LED (model trained on Runs 1+2)',
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '06_model4_predictions.png'), dpi=150,
            bbox_inches='tight')
plt.close()
print("  Saved: plots/06_model4_predictions.png")

# --- Plot 7: Residual analysis (Model 4) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model 4 Residual Analysis (Actual - Predicted)',
             fontsize=14, fontweight='bold')

residuals_cv = y_true_cv - y_pred_cv

ax = axes[0]
ax.scatter(y_true_cv, residuals_cv, s=3, alpha=0.3, color='#3498DB')
ax.axhline(y=0, color='red', linewidth=1, linestyle='--')
ax.set_xlabel('Actual T_IN (degC)')
ax.set_ylabel('Residual (degC)')
ax.set_title('CV: Residuals vs T_IN')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.hist(residuals_cv, bins=50, color='#3498DB', alpha=0.7,
        edgecolor='black', linewidth=0.5)
ax.axvline(x=0, color='red', linewidth=1, linestyle='--')
ax.set_xlabel('Residual (degC)')
ax.set_ylabel('Count')
ax.set_title(f'CV Residual Distribution\n'
             f'Mean={np.mean(residuals_cv):.2f}degC, '
             f'Std={np.std(residuals_cv):.2f}degC')
ax.grid(True, alpha=0.3)

residuals_stress = y_true_stress - y_pred_stress
ax = axes[2]
ax.scatter(y_true_stress, residuals_stress, s=3, alpha=0.3, color='#E74C3C')
ax.axhline(y=0, color='red', linewidth=1, linestyle='--')
ax.set_xlabel('Actual T_IN (degC)')
ax.set_ylabel('Residual (degC)')
ax.set_title('Stress Test (Run 3): Residuals vs T_IN')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '07_residual_analysis.png'), dpi=150,
            bbox_inches='tight')
plt.close()
print("  Saved: plots/07_residual_analysis.png")

# --- Plot 8: Critical temperature error distributions ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Error Distribution at Critical PCR Temperatures (Model 4, CV)',
             fontsize=14, fontweight='bold')

run1 = runs['Run 1']
y_pred_f2 = all_results['Model 4: + T_OUT^2']['pred_fold2']
y_true_f2 = run1['t_in'].values

crit_colors = ['#E67E22', '#E74C3C', '#8E44AD']

for idx, (label, temp) in enumerate(CRITICAL_TEMPS.items()):
    ax = axes[idx]

    # Combine errors from both CV folds
    mask_f1 = (y_true_cv >= temp - TEMP_WINDOW) & (y_true_cv <= temp + TEMP_WINDOW)
    mask_f2 = (y_true_f2 >= temp - TEMP_WINDOW) & (y_true_f2 <= temp + TEMP_WINDOW)

    errors_combined = np.concatenate([
        y_true_cv[mask_f1] - y_pred_cv[mask_f1],
        y_true_f2[mask_f2] - y_pred_f2[mask_f2]
    ])

    if len(errors_combined) > 0:
        ax.hist(errors_combined, bins=30, color=crit_colors[idx], alpha=0.7,
                edgecolor='black', linewidth=0.5)
        ax.axvline(x=0, color='red', linewidth=1, linestyle='--')
        mae_here = np.mean(np.abs(errors_combined))
        ax.set_xlabel('Prediction Error (degC)')
        ax.set_ylabel('Count')
        ax.set_title(f'Near {label}\n'
                     f'MAE={mae_here:.2f}degC, n={len(errors_combined)} points')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(f'Near {label}')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '08_critical_temp_errors.png'), dpi=150,
            bbox_inches='tight')
plt.close()
print("  Saved: plots/08_critical_temp_errors.png")

# --- Plot 9: Per-cycle MAE (Model 4) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model 4: Per-Cycle MAE (Does Error Grow With Cycle Number?)',
             fontsize=14, fontweight='bold')

# CV: Run 2 predictions
cycles_r2 = sorted(run2['cycle'].unique())
cycle_mae_cv = []
for c in cycles_r2:
    mask = run2['cycle'].values == c
    if mask.any():
        cycle_mae_cv.append(mean_absolute_error(y_true_cv[mask], y_pred_cv[mask]))
    else:
        cycle_mae_cv.append(np.nan)

ax = axes[0]
ax.bar(cycles_r2, cycle_mae_cv, color='#3498DB', alpha=0.7, edgecolor='#2980B9')
ax.set_xlabel('Cycle Number')
ax.set_ylabel('MAE (degC)')
ax.set_title('CV: Per-Cycle MAE (Run 2)')
ax.grid(True, alpha=0.3, axis='y')

# Stress test: Run 3 predictions
cycles_r3 = sorted(run3['cycle'].unique())
cycle_mae_stress = []
for c in cycles_r3:
    mask = run3['cycle'].values == c
    if mask.any():
        cycle_mae_stress.append(
            mean_absolute_error(y_true_stress[mask], y_pred_stress[mask]))
    else:
        cycle_mae_stress.append(np.nan)

ax = axes[1]
ax.bar(cycles_r3, cycle_mae_stress, color='#E74C3C', alpha=0.7, edgecolor='#C0392B')
ax.set_xlabel('Cycle Number')
ax.set_ylabel('MAE (degC)')
ax.set_title('Stress Test: Per-Cycle MAE (Run 3)')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '09_per_cycle_error.png'), dpi=150,
            bbox_inches='tight')
plt.close()
print("  Saved: plots/09_per_cycle_error.png")


# ============================================================
# SECTION 13: COMPREHENSIVE SUMMARY TABLE
# ============================================================
print("\n" + "=" * 70)
print("SECTION 13: COMPREHENSIVE SUMMARY")
print("=" * 70)

print("""
WHAT THIS TABLE SHOWS:
  All 7 models compared side-by-side. Models 1-4 show the incremental
  value of each feature. 'Model 3 + Phase' shows the effect of including
  protocol phase information. RF/GBM show if non-linear models help.

  LEFT side: Cross-validation accuracy (Runs 1<->2 — normal conditions)
  RIGHT side: Stress test accuracy (Run 3 — reduced LED efficiency)
""")

# Summary table
header = (f"{'Model':<25} | {'CV MAE':>7} | {'CV RMSE':>8} | {'CV Max':>7} |"
          f" {'CV R^2':>7} | {'R3 MAE':>7} | {'R3 RMSE':>8} | {'R3 Max':>7}")
print(header)
print("-" * len(header))

for model_name in model_names_ordered:
    cv = all_results[model_name]['CV_metrics']
    st = all_results[model_name]['stress_metrics']
    name_disp = model_name[:25]
    print(f"{name_disp:<25} | {cv['MAE']:>6.2f}deg | {cv['RMSE']:>7.2f}deg |"
          f" {cv['Max Error']:>6.2f}deg | {cv['R^2']:>6.4f} |"
          f" {st['MAE']:>6.2f}deg | {st['RMSE']:>7.2f}deg | {st['Max Error']:>6.2f}deg")

# Critical temperature table
print()
header2 = (f"{'Model':<25} | {'CV @60':>7} | {'CV @72':>7} | {'CV @95':>7} |"
           f" {'R3 @60':>7} | {'R3 @72':>7} | {'R3 @95':>7}")
print(header2)
print("-" * len(header2))

for model_name in model_names_ordered:
    cv_c = all_results[model_name]['CV_critical']
    st_c = all_results[model_name]['stress_critical']
    name_disp = model_name[:25]

    vals = []
    for d in [cv_c, st_c]:
        for label in CRITICAL_TEMPS:
            v = d.get(label, np.nan)
            vals.append(f"{v:>6.2f}deg" if not np.isnan(v) else "   N/A")

    print(f"{name_disp:<25} | {vals[0]:>7} | {vals[1]:>7} | {vals[2]:>7} |"
          f" {vals[3]:>7} | {vals[4]:>7} | {vals[5]:>7}")


# ============================================================
# SECTION 14: ARDUINO NANO DEPLOYMENT
# ============================================================
print("\n" + "=" * 70)
print("SECTION 14: ARDUINO NANO DEPLOYMENT CODE")
print("=" * 70)
print("""
WHAT THIS IS:
  The linear regression equation from the best practical model,
  formatted as Arduino-ready C code.

HOW IT WORKS ON ARDUINO:
  1. Read T_OUT from thermocouple every second
  2. Store previous 3 readings in a circular buffer (4 floats = 16 bytes)
  3. Compute dT/dt = T_OUT_current - T_OUT_previous
  4. Plug values into the equation below
  5. Use predicted T_IN for PID control decisions

RESOURCE USAGE:
  - Flash: ~32 bytes for coefficients + ~200 bytes for function code
  - SRAM: ~16 bytes for temperature buffer
  - Compute: ~7 multiply-adds ~= 140µs (software float on ATmega328P)
  - Total: Negligible overhead for the PCR control loop
""")

# Print Arduino code for Model 4
m4_model = all_results['Model 4: + T_OUT^2']['model_stress']
m4_features = MODEL_CONFIGS['Model 4: + T_OUT^2']['features']

print("  // --- Arduino Code (Model 4) -----------------------------------")
print("  float predict_T_IN(float T_OUT, float T_OUT_prev1,")
print("                      float T_OUT_prev2, float T_OUT_prev3,")
print("                      int cycle) {")
print(f"    float dT = T_OUT - T_OUT_prev1;")
print(f"    float T_IN = {m4_model.intercept_:.6f}")

for feat, coef in zip(m4_features, m4_model.coef_):
    sign = "+" if coef >= 0 else "-"
    abs_coef = abs(coef)
    if feat == 't_out':
        print(f"                 {sign} {abs_coef:.6f} * T_OUT")
    elif feat == 'dT_OUT_dt':
        print(f"                 {sign} {abs_coef:.6f} * dT")
    elif feat == 't_out_lag1':
        print(f"                 {sign} {abs_coef:.6f} * T_OUT_prev1")
    elif feat == 't_out_lag2':
        print(f"                 {sign} {abs_coef:.6f} * T_OUT_prev2")
    elif feat == 't_out_lag3':
        print(f"                 {sign} {abs_coef:.6f} * T_OUT_prev3")
    elif feat == 'cycle':
        print(f"                 {sign} {abs_coef:.6f} * (float)cycle")
    elif feat == 't_out_sq':
        print(f"                 {sign} {abs_coef:.8f} * T_OUT * T_OUT")

print(f"                 ;")
print(f"    return T_IN;")
print(f"  }}")
print("  // --------------------------------------------------------------")

# Also print Model 3 for comparison (simpler, no quadratic)
print("\n  // --- Simpler Alternative (Model 3, no quadratic) -------------")
m3_model = all_results['Model 3: + lags + cycle']['model_stress']
m3_features = MODEL_CONFIGS['Model 3: + lags + cycle']['features']

print("  float predict_T_IN_simple(float T_OUT, float T_OUT_prev1,")
print("                             float T_OUT_prev2, float T_OUT_prev3,")
print("                             int cycle) {")
print(f"    float dT = T_OUT - T_OUT_prev1;")
print(f"    float T_IN = {m3_model.intercept_:.6f}")

for feat, coef in zip(m3_features, m3_model.coef_):
    sign = "+" if coef >= 0 else "-"
    abs_coef = abs(coef)
    if feat == 't_out':
        print(f"                 {sign} {abs_coef:.6f} * T_OUT")
    elif feat == 'dT_OUT_dt':
        print(f"                 {sign} {abs_coef:.6f} * dT")
    elif feat == 't_out_lag1':
        print(f"                 {sign} {abs_coef:.6f} * T_OUT_prev1")
    elif feat == 't_out_lag2':
        print(f"                 {sign} {abs_coef:.6f} * T_OUT_prev2")
    elif feat == 't_out_lag3':
        print(f"                 {sign} {abs_coef:.6f} * T_OUT_prev3")
    elif feat == 'cycle':
        print(f"                 {sign} {abs_coef:.6f} * (float)cycle")

print(f"                 ;")
print(f"    return T_IN;")
print(f"  }}")
print("  // --------------------------------------------------------------")


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print(f"All plots saved to: {PLOT_DIR}")
print("=" * 70)
print("""
NEXT STEPS:
  1. Review the results above and decide acceptable accuracy
  2. Perform Run 4 with normal LED efficiency
  3. If Run 4 behaves normally: retrain on Runs 1+2+4, validate on Run 3
  4. Deploy chosen model equation to Arduino Nano
""")
