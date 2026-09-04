import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

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
            match = re.match(r'Time:([\d.]+),\s*Cycle:(\d+),\s*(\w+),\s*T_OUT:([\d.]+),\s*T_IN:([\d.]+)', line)
            if match:
                time_s, cycle, phase, t_out, t_in = match.groups()
                data.append({
                    'cycle': int(cycle),
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

features = ['t_out', 'dT_OUT_dt', 't_out_lag1', 't_out_lag2', 't_out_lag3', 'cycle']

print("=== LEAVE-ONE-OUT CROSS VALIDATION ===")
all_maes_lr = []
all_maes_rf = []

for test_name in runs.keys():
    train_dfs = [runs[n] for n in runs.keys() if n != test_name]
    df_train = pd.concat(train_dfs)
    df_test = runs[test_name]
    
    # Linear Regression M3
    lr = LinearRegression()
    lr.fit(df_train[features], df_train['t_in'])
    pred_lr = lr.predict(df_test[features])
    mae_lr = mean_absolute_error(df_test['t_in'], pred_lr)
    all_maes_lr.append(mae_lr)
    
    # RF
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(df_train[features], df_train['t_in'])
    pred_rf = rf.predict(df_test[features])
    mae_rf = mean_absolute_error(df_test['t_in'], pred_rf)
    all_maes_rf.append(mae_rf)

    print(f"Test on {test_name:10s} | Linear M3 MAE: {mae_lr:.2f} | RF M3 MAE: {mae_rf:.2f}")

print("-" * 50)
print(f"AVERAGE CV MAE | Linear M3: {np.mean(all_maes_lr):.2f} | RF M3: {np.mean(all_maes_rf):.2f}")
