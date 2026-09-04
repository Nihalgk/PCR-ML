import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent
PLOT_DIR = DATA_DIR / 'soak_plots'
os.makedirs(PLOT_DIR, exist_ok=True)

FILES = {
    '19.08_3step_2': DATA_DIR / '40 cycle(3step)_19.08.2026_2.txt',
    '19.08_3step_3': DATA_DIR / '40 cycle(3step)_19.08.2026_3.txt',
    '20.08_30sHold': DATA_DIR / '40 cycle(30sHold)_20.08.2026.txt',
    '20.08_30sHold_2': DATA_DIR / '40 cycle(30sHold)_20.08.2026_2.txt',
    '19.08_3step_1': DATA_DIR / '40 cycle(3step)_19.08.2026.txt',
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
            # fallback if F T_IN is not present or something
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
                'mode': mode.strip() if mode else ''
            })
    return pd.DataFrame(data)

def analyze_thermal_soak(df, name):
    phases = ['DENAT', 'ANNEAL', 'EXTEND']
    results = []
    
    for cycle in df['cycle'].unique():
        cycle_data = df[df['cycle'] == cycle]
        for phase in phases:
            phase_data = cycle_data[cycle_data['phase'] == phase]
            if len(phase_data) == 0:
                continue
            
            end_samples = phase_data.tail(3)
            
            t_out_end = end_samples['t_out'].mean()
            t_in_end = end_samples['t_in'].mean()
            
            results.append({
                'cycle': cycle,
                'phase': phase,
                't_out': t_out_end,
                't_in': t_in_end,
                'diff': t_out_end - t_in_end
            })
            
    return pd.DataFrame(results)

plt.figure(figsize=(15, 10))

for idx, (name, path) in enumerate(FILES.items()):
    df = parse_pcr_file(path)
    soak_df = analyze_thermal_soak(df, name)
    
    print(f"--- {name} ---")
    for phase in ['DENAT', 'ANNEAL', 'EXTEND']:
        phase_df = soak_df[soak_df['phase'] == phase]
        if len(phase_df) == 0:
            continue
            
        early_diff = phase_df[phase_df['cycle'] <= 3]['diff'].mean()
        late_diff = phase_df[phase_df['cycle'] >= 30]['diff'].mean()
        
        print(f"Phase {phase}:")
        print(f"  Early (cycles 1-3) diff: {early_diff:.2f}")
        print(f"  Late (cycles 30+) diff: {late_diff:.2f}")
        print(f"  Drift (Late - Early): {late_diff - early_diff:.2f}")
        
    plt.subplot(3, 2, idx + 1)
    for phase in ['DENAT', 'ANNEAL', 'EXTEND']:
        phase_df = soak_df[soak_df['phase'] == phase]
        plt.plot(phase_df['cycle'], phase_df['diff'], marker='o', label=phase)
    
    plt.title(name)
    plt.xlabel('Cycle')
    plt.ylabel('T_OUT - T_IN (degC)')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'thermal_soak_analysis.png'))
print("Analysis complete. Saved plot to", os.path.join(PLOT_DIR, 'thermal_soak_analysis.png'))
