import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = BASE_DIR / "logs" / "predictions.log"
BASELINE_FILE = BASE_DIR / "outputs" / "feature_baseline.json"


def calculate_psi(expected: List[float], actual: List[float], eps: float = 1e-6) -> float:
    """
    Calculate Population Stability Index (PSI)
    """
    psi = 0.0

    for e, a in zip(expected, actual):
        e = max(e, eps)
        a = max(a, eps)
        psi += (a - e) * np.log(a / e)

    return round(float(psi), 4)


def interpret_psi(psi: float) -> str:
    """
    Interpret PSI thresholds
    """
    if psi < 0.1:
        return "No drift"
    elif psi < 0.2:
        return "Moderate drift"
    else:
        return "Significant drift - retrain recommended"


def load_baseline() -> Dict[str, Any]:
    """
    Load training-time baseline distribution
    """
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(f"Baseline file not found at: {BASELINE_FILE}")

    with open(BASELINE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prediction_logs() -> List[Dict[str, Any]]:
    """
    Load production logs from predictions.log
    Each line is expected to be a JSON object
    """
    records = []

    if not LOG_FILE.exists():
        return records

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                if "input_data" in entry:
                    records.append(entry["input_data"])
            except json.JSONDecodeError:
                continue

    return records


def build_actual_distribution(values: List[float], bin_edges: List[float]) -> List[float]:
    """
    Build actual production distribution using the same bin edges as training
    """
    if not values:
        raise ValueError("No production values available for distribution calculation")

    counts, _ = np.histogram(values, bins=bin_edges)
    total = counts.sum()

    if total == 0:
        raise ValueError("Histogram count is zero. Cannot compute actual distribution")

    distribution = (counts / total).round(6).tolist()
    return distribution


def run_psi_monitor(feature_name: str = "length_of_stay_hours") -> Dict[str, Any]:
    """
    Run PSI monitor for a specific feature
    """
    baseline = load_baseline()

    if feature_name not in baseline:
        return {
            "error": f"Feature '{feature_name}' not found in baseline file"
        }

    feature_baseline = baseline[feature_name]

    expected_distribution = feature_baseline.get("distribution")
    bin_edges = feature_baseline.get("bin_edges")

    if not expected_distribution or not bin_edges:
        return {
            "error": f"Baseline data for feature '{feature_name}' is incomplete"
        }

    log_records = load_prediction_logs()

    if not log_records:
        return {
            "error": "No production prediction logs found yet"
        }

    values = []
    for record in log_records:
        value = record.get(feature_name)
        if value is None:
            continue

        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue

    if len(values) < 10:
        return {
            "error": f"Not enough production data for PSI on '{feature_name}'. Minimum 10 records required.",
            "records_found": len(values)
        }

    actual_distribution = build_actual_distribution(values, bin_edges)
    psi_value = calculate_psi(expected_distribution, actual_distribution)
    status = interpret_psi(psi_value)

    return {
        "feature": feature_name,
        "records_used": len(values),
        "expected_distribution": expected_distribution,
        "actual_distribution": actual_distribution,
        "bin_edges": bin_edges,
        "psi": psi_value,
        "status": status,
        "thresholds": {
            "no_drift_below": 0.1,
            "moderate_drift_below": 0.2,
            "significant_drift_at_or_above": 0.2
        }
    }