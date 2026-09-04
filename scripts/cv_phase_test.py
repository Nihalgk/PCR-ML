import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, max_error

DATA_DIR = Path(__file__).resolve().parent
files = [
    '40 cycle(3step)_19.08.2026.txt',
    '40 cycle(3step)_19.08.2026_2.txt',
    '40 cycle(3step)_19.08.2026_3.txt',
    '40 cycle(30sHold)_20.08.2026.txt',
    '40 cycle(30sHold)_20.08.2026_2.txt'
]

def load_data():
    all_data = []
    for file in files:
        filepath = os.path.join(DATA_DIR, file)
        if not os.path.exists(filepath): continue
        with open(filepath, 'r') as f:
            for line in f:
                match = re.match(r'Time:([\d.]+),\s*Cycle:(\d+),\s*(\w+),\s*T_OUT:([\d.]+),\s*T_IN:([\d.]+)', line.strip())
                if match:
                    all_data.append({
                        'file': file,
                        'time': float(match.group(1)), 
                        'cycle': int(match.group(2)), 
                        'phase': match.group(3), 
                        't_out': float(match.group(4)), 
                        't_in': float(match.group(5))
                    })
    
    df = pd.DataFrame(all_data)
    
    df['t_out_lag1'] = df.groupby('file')['t_out'].shift(10)
    df['t_out_lag2'] = df.groupby('file')['t_out'].shift(20)
    df['t_out_lag3'] = df.groupby('file')['t_out'].shift(30)
    df['dT_OUT_dt'] = df['t_out'] - df['t_out_lag1']
    df = df.dropna()
    return df

df = load_data()
features = ['t_out', 'dT_OUT_dt', 't_out_lag1', 't_out_lag2', 't_out_lag3', 'cycle']

# Map phases
df['super_phase'] = df['phase']
df.loc[df['phase'] == 'INIT', 'super_phase'] = 'DENAT'
df.loc[df['phase'] == 'EXTEND', 'super_phase'] = 'EXTEND'
df.loc[df['phase'] == 'CYCLE_EXTENSION', 'super_phase'] = 'EXTEND'
df.loc[df['phase'] == 'FINAL_EXTENSION', 'super_phase'] = 'EXTEND'

# Helper to distinguish ramps vs holds
def is_hold(row):
    # simple heuristic: if t_in is near target
    if row['super_phase'] == 'DENAT':
        return row['t_in'] >= 94.0
    elif row['super_phase'] == 'ANNEAL':
        return row['t_in'] <= 61.0
    elif row['super_phase'] == 'EXTEND':
        return row['t_in'] >= 71.0
    return False

df['is_hold'] = df.apply(is_hold, axis=1)

print(f"Data points: {len(df)}")
print("-" * 50)

predictions_global = []
predictions_phase = []

# Leave-One-Out CV
for file_out in files:
    train_df = df[df['file'] != file_out]
    test_df = df[df['file'] == file_out].copy()
    
    if len(test_df) == 0: continue
    
    # 1. Global Model Training
    lr_global = LinearRegression().fit(train_df[features], train_df['t_in'])
    test_df['pred_global'] = lr_global.predict(test_df[features])
    
    # 2. Phase-Specific Models Training
    test_df['pred_phase'] = np.nan
    for sp in ['DENAT', 'ANNEAL', 'EXTEND']:
        mask_train = train_df['super_phase'] == sp
        mask_test = test_df['super_phase'] == sp
        if mask_train.sum() == 0 or mask_test.sum() == 0: continue
        
        lr_phase = LinearRegression().fit(train_df.loc[mask_train, features], train_df.loc[mask_train, 't_in'])
        test_df.loc[mask_test, 'pred_phase'] = lr_phase.predict(test_df.loc[mask_test, features])
        
    predictions_global.append(test_df)

result_df = pd.concat(predictions_global)

# Generate final coefficients across ALL data for Arduino
final_models = {}
for sp in ['DENAT', 'ANNEAL', 'EXTEND']:
    mask = df['super_phase'] == sp
    if mask.sum() > 0:
        lr = LinearRegression().fit(df.loc[mask, features], df.loc[mask, 't_in'])
        final_models[sp] = lr

def report(name, true_y, pred_y):
    mae = mean_absolute_error(true_y, pred_y)
    mx = max_error(true_y, pred_y)
    return f"MAE: {mae:.2f}°C (Max: {mx:.2f}°C)"

print("=== LEAVE-ONE-RUN-OUT CV RESULTS ===")
print("GLOBAL MODEL:")
print(f"  Overall : {report('Global', result_df['t_in'], result_df['pred_global'])}")
print(f"  DENAT   : {report('Global', result_df[result_df['super_phase']=='DENAT']['t_in'], result_df[result_df['super_phase']=='DENAT']['pred_global'])}")
print(f"  ANNEAL  : {report('Global', result_df[result_df['super_phase']=='ANNEAL']['t_in'], result_df[result_df['super_phase']=='ANNEAL']['pred_global'])}")
print(f"  EXTEND  : {report('Global', result_df[result_df['super_phase']=='EXTEND']['t_in'], result_df[result_df['super_phase']=='EXTEND']['pred_global'])}")

result_df = result_df.dropna(subset=['pred_phase'])

print("\nPHASE-SPECIFIC MODELS:")
print(f"  Overall : {report('Phase', result_df['t_in'], result_df['pred_phase'])}")
print(f"  DENAT   : {report('Phase', result_df[result_df['super_phase']=='DENAT']['t_in'], result_df[result_df['super_phase']=='DENAT']['pred_phase'])}")
print(f"  ANNEAL  : {report('Phase', result_df[result_df['super_phase']=='ANNEAL']['t_in'], result_df[result_df['super_phase']=='ANNEAL']['pred_phase'])}")
print(f"  EXTEND  : {report('Phase', result_df[result_df['super_phase']=='EXTEND']['t_in'], result_df[result_df['super_phase']=='EXTEND']['pred_phase'])}")

print("\n--- ERROR BY PHASE TYPE (Phase-Specific) ---")
print(f"  Ramps   : {report('Phase', result_df[~result_df['is_hold']]['t_in'], result_df[~result_df['is_hold']]['pred_phase'])}")
print(f"  Holds   : {report('Phase', result_df[result_df['is_hold']]['t_in'], result_df[result_df['is_hold']]['pred_phase'])}")

print("\n=== FINAL ARDUINO COEFFICIENTS (ALL DATA) ===")
for sp in ['DENAT', 'ANNEAL', 'EXTEND']:
    if sp not in final_models: continue
    print(f"\n{sp} MODEL:")
    print(f"  float raw_estimate = {final_models[sp].intercept_:.4f}f")
    print(f"                       + ({final_models[sp].coef_[0]:.4f}f * s1)")
    print(f"                       + ({final_models[sp].coef_[1]:.4f}f * dT_dt)")
    print(f"                       + ({final_models[sp].coef_[2]:.4f}f * lag1)")
    print(f"                       + ({final_models[sp].coef_[3]:.4f}f * lag2)")
    print(f"                       + ({final_models[sp].coef_[4]:.4f}f * lag3)")
    print(f"                       + ({final_models[sp].coef_[5]:.4f}f * (float)currentCycle);")
