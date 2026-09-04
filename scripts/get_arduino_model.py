#!/usr/bin/env python3
import os
import re
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data' if (BASE_DIR / 'data').exists() else BASE_DIR
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
    df = df.dropna()
    return df

dfs = []
for path in FILES.values():
    dfs.append(engineer_features(parse_pcr_file(path)))

df_all = pd.concat(dfs)

features = ['t_out', 'dT_OUT_dt', 't_out_lag1', 't_out_lag2', 't_out_lag3', 'cycle']
lr = LinearRegression()
lr.fit(df_all[features], df_all['t_in'])

pred = lr.predict(df_all[features])
mae = mean_absolute_error(df_all['t_in'], pred)

print(f"Linear M3 Training MAE: {mae:.2f}")
print("Coefficients:")
print(f"Intercept: {lr.intercept_}")
for f, c in zip(features, lr.coef_):
    print(f"{f}: {c}")
