"""
Log parsing module for PCR thermocycler serial telemetry.
Handles variable formats, comments, missing fields, and multiple protocol types.
"""

from pathlib import Path
import re
from typing import Dict, List, Optional, Union
import pandas as pd


# Regex pattern to match PCR log lines
# Example: Time:1.0, Cycle:1, INIT, T_OUT:23.00, T_IN:23.50, LED PWM:255, HEATING_FULL
# Example with F T_IN: Time:1.0, Cycle:1, INIT, T_OUT:25.00, T_IN:25.75, F T_IN:25.75, HEATING_FULL
PCR_LOG_PATTERN = re.compile(
    r"Time:\s*(?P<time>[\d.]+),\s*"
    r"Cycle:\s*(?P<cycle>\d+),\s*"
    r"(?P<phase>[A-Za-z0-9_]+),\s*"
    r"T_OUT:\s*(?P<t_out>[\d.]+),\s*"
    r"T_IN:\s*(?P<t_in>[\d.]+)"
    r"(?:,\s*F\s*T_IN:\s*(?P<f_t_in>[\d.]+))?"
    r"(?:,\s*LED\s*PWM:\s*(?P<led_pwm>\d+))?"
    r"(?:,\s*(?P<status>[A-Za-z0-9_ -]+))?",
    re.IGNORECASE
)


def parse_pcr_file(
    filepath: Union[str, Path],
    run_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Parse a PCR raw log file into a clean pandas DataFrame.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the raw serial output .txt log file.
    run_name : Optional[str]
        Identifier for the run. If None, derived from filename.

    Returns
    -------
    pd.DataFrame
        Cleaned telemetry DataFrame with columns:
        ['time', 'cycle', 'phase', 't_out', 't_in', 'run_name', ...]
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PCR log file not found at: {filepath}")

    if run_name is None:
        run_name = filepath.stem

    records: List[Dict[str, Union[float, int, str]]] = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, headers, comments
            if not line or line.startswith("=") or line.startswith("PCR:") or line.startswith("#"):
                continue

            match = PCR_LOG_PATTERN.search(line)
            if match:
                groups = match.groupdict()
                record: Dict[str, Union[float, int, str]] = {
                    "time": float(groups["time"]),
                    "cycle": int(groups["cycle"]),
                    "phase": str(groups["phase"]).strip().upper(),
                    "t_out": float(groups["t_out"]),
                    "t_in": float(groups["t_in"]),
                    "run_name": run_name,
                }
                if groups.get("f_t_in") is not None:
                    try:
                        record["f_t_in"] = float(groups["f_t_in"])
                    except (ValueError, TypeError):
                        pass
                if groups.get("led_pwm") is not None:
                    try:
                        record["led_pwm"] = int(groups["led_pwm"])
                    except (ValueError, TypeError):
                        pass
                if groups.get("status") is not None:
                    record["status"] = str(groups["status"]).strip()

                records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No valid PCR telemetry lines could be parsed from: {filepath}")

    return df


def load_all_runs(
    base_dir: Optional[Union[str, Path]] = None,
    pattern: str = "*.txt"
) -> Dict[str, pd.DataFrame]:
    """
    Find and parse all PCR log files in the given directory.

    Parameters
    ----------
    base_dir : Optional[Union[str, Path]]
        Directory to search for log files. Defaults to current workspace.
    pattern : str
        Glob pattern for log files (default: '*.txt').

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping run names to parsed DataFrames.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2]
    else:
        base_dir = Path(base_dir)

    runs: Dict[str, pd.DataFrame] = {}
    
    # Check root and data/ directory if exists
    search_paths = [base_dir]
    if (base_dir / "data").exists():
        search_paths.append(base_dir / "data")

    for search_path in search_paths:
        for file in sorted(search_path.glob(pattern)):
            if "cycle" in file.name.lower():
                name = file.stem
                try:
                    df = parse_pcr_file(file, run_name=name)
                    runs[name] = df
                except Exception as e:
                    print(f"Warning: Failed to parse {file.name}: {e}")

    return runs
