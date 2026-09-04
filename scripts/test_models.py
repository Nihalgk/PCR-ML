import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import re
from pathlib import Path
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

DATA_DIR = Path(__file__).resolve().parent
FILES = {
    '19.08_1': DATA_DIR / '40 cycle(3step)_19.08.2026.txt',
    '19.08_2': DATA_DIR / '40 cycle(3step)_19.08.2026_2.txt',
    '19.08_3': DATA_DIR / '40 cycle(3step)_19.08.2026_3.txt',
    '20.08_1': DATA_DIR / '40 cycle(30sHold)_20.08.2026.txt',
    '20.08_2': DATA_DIR / '40 cycle(30sHold)_20.08.2026_2.txt',
}

def parse_pcr_file(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('=') or line.startswith('PCR:'):
                continue
            match = re.match(
                r'Time:([\d.]+),\s*Cycle:(\d+),\s*(\w+),\s*'
                r'T_OUT:([\d.]+),\s*T_IN:([\d.]+),\s*'
                r'F T_IN:([\d.]+)?(?:,\s*)?(.*)',
                line
            )
            if not match:
                match = re.match(
                    r'Time:([\d.]+),\s*Cycle:(\d+),\s*(\w+),\s*'
                    r'T_OUT:([\d.]+),\s*T_IN:([\d.]+),\s*(.*)',
                    line
                )
                if match:
                    time_s, cycle, phase, t_out, t_in, mode = match.groups()
                    f_t_in = t_in
                else:
                    continue
            else:
                time_s, cycle, phase, t_out, t_in, f_t_in, mode = match.groups()
                
            data.append({
                'time': float(time_s),
                'cycle': int(cycle),
                'phase': phase,
                't_out': float(t_out),
                't_in': float(t_in),
            })
    return pd.DataFrame(data)

def engineer_features(df):
    df = df.copy()
    df['dT_OUT_dt'] = df['t_out'].diff()
    df['t_out_lag1'] = df['t_out'].shift(1)
    df['t_out_lag2'] = df['t_out'].shift(2)
    df['t_out_lag3'] = df['t_out'].shift(3)
    df['t_out_sq'] = df['t_out'] ** 2
    df = df.dropna()
    return df

runs = {}
for name, path in FILES.items():
    df = parse_pcr_file(path)
    runs[name] = engineer_features(df)

# Train on 19.08 (3s hold), Test on 20.08 (30s hold)
train_dfs = [runs['19.08_1'], runs['19.08_2'], runs['19.08_3']]
test_dfs = [runs['20.08_1'], runs['20.08_2']]

df_train = pd.concat(train_dfs)
df_test = pd.concat(test_dfs)

MODEL_CONFIGS = {
    'M1_T_OUT_only': ['t_out'],
    'M2_dT_dt': ['t_out', 'dT_OUT_dt'],
    'M3_lags_cycle': ['t_out', 'dT_OUT_dt', 't_out_lag1', 't_out_lag2', 't_out_lag3', 'cycle'],
    'M4_quadratic': ['t_out', 'dT_OUT_dt', 't_out_lag1', 't_out_lag2', 't_out_lag3', 'cycle', 't_out_sq']
}

print("=== EVALUATING MODELS ===")
print("Trained on 19.08 (3s hold) -> Tested on 20.08 (30s hold)")
print("-" * 50)

for name, features in MODEL_CONFIGS.items():
    model = LinearRegression()
    model.fit(df_train[features], df_train['t_in'])
    pred = model.predict(df_test[features])
    mae = mean_absolute_error(df_test['t_in'], pred)
    print(f"Linear {name:15s}: MAE = {mae:.2f} degC")

print("-" * 50)
features = MODEL_CONFIGS['M3_lags_cycle']
rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(df_train[features], df_train['t_in'])
pred_rf = rf.predict(df_test[features])
mae_rf = mean_absolute_error(df_test['t_in'], pred_rf)
print(f"Random Forest M3        : MAE = {mae_rf:.2f} degC")

gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
gb.fit(df_train[features], df_train['t_in'])
pred_gb = gb.predict(df_test[features])
mae_gb = mean_absolute_error(df_test['t_in'], pred_gb)
print(f"Gradient Boosting M3    : MAE = {mae_gb:.2f} degC")
