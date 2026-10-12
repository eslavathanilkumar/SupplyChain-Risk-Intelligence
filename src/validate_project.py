from pathlib import Path
import json
import sys

import pandas as pd
import joblib
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FILES = [
    "reports/ml_model_comparison.csv",
    "reports/country_network_metrics.csv",
    "reports/risk_intelligence_report.txt",
    "reports/risk_intelligence_report.json",
    "reports/network_analysis_report.txt",
    "reports/genai_risk_briefing.md",
    "dashboard/app.py",
    "src/api.py",
]

def check(condition, message):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    return bool(condition)

def main():
    print("=" * 60)
    print("SUPPLY CHAIN RISK INTELLIGENCE - VALIDATION")
    print("=" * 60)

    failures = 0

    # 1. Required files
    for relative_path in REQUIRED_FILES:
        path = PROJECT_ROOT / relative_path
        if not check(path.is_file(), f"File exists: {relative_path}"):
            failures += 1

    # Stop data checks if required inputs are missing
    model_csv = PROJECT_ROOT / "reports/ml_model_comparison.csv"
    network_csv = PROJECT_ROOT / "reports/country_network_metrics.csv"
    risk_json = PROJECT_ROOT / "reports/risk_intelligence_report.json"
    briefing = PROJECT_ROOT / "reports/genai_risk_briefing.md"

    if model_csv.is_file():
        try:
            models = pd.read_csv(model_csv)
            valid = not models.empty and {
                "Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"
            }.issubset(models.columns)
            if not check(valid, "Model comparison CSV has expected columns and rows"):
                failures += 1
            else:
                print("\nModel comparison:")
                print(models.to_string(index=False))
        except Exception as exc:
            print(f"[FAIL] Cannot read model comparison CSV: {exc}")
            failures += 1

    if network_csv.is_file():
        try:
            network = pd.read_csv(network_csv)
            valid = not network.empty
            if not check(valid, "Network metrics CSV contains rows"):
                failures += 1
            else:
                print(f"Network metric rows: {len(network):,}")
                print(f"Network columns: {', '.join(network.columns)}")
        except Exception as exc:
            print(f"[FAIL] Cannot read network metrics CSV: {exc}")
            failures += 1

    if risk_json.is_file():
        try:
            with risk_json.open("r", encoding="utf-8") as file:
                json.load(file)
            check(True, "Risk intelligence JSON is valid")
        except Exception as exc:
            print(f"[FAIL] Risk intelligence JSON is invalid: {exc}")
            failures += 1

    if briefing.is_file():
        try:
            text = briefing.read_text(encoding="utf-8").strip()
            if not check(len(text) > 100, "GenAI briefing is non-empty and substantial"):
                failures += 1
        except Exception as exc:
            print(f"[FAIL] Cannot read GenAI briefing: {exc}")
            failures += 1

    # 2. Check saved ML model artifacts
    model_dir = PROJECT_ROOT / "models"
    model_files = list(model_dir.glob("*.joblib")) if model_dir.exists() else []
    if check(len(model_files) > 0, "Saved ML model artifacts exist"):
        for model_file in model_files:
            try:
                joblib.load(model_file)
                check(True, f"Model loads: {model_file.name}")
            except Exception as exc:
                print(f"[FAIL] Model failed to load: {model_file.name}: {exc}")
                failures += 1

    # 3. Check local API endpoints
    endpoints = [
        "/health",
        "/api/v1/model-comparison",
        "/api/v1/risk-report",
        "/api/v1/network/summary",
        "/api/v1/network/top-countries",
        "/api/v1/genai-briefing",
    ]

    print("\nAPI endpoint checks:")
    for endpoint in endpoints:
        try:
            response = requests.get(
                f"http://127.0.0.1:8000{endpoint}",
                timeout=10,
            )
            passed = response.status_code == 200
            if not check(passed, f"{endpoint} -> HTTP {response.status_code}"):
                failures += 1
        except requests.RequestException as exc:
            print(f"[FAIL] {endpoint} -> {exc}")
            failures += 1

    print("\n" + "=" * 60)
    if failures:
        print(f"VALIDATION FINISHED WITH {failures} FAILURE(S)")
        print("Review the failed checks before publishing.")
        sys.exit(1)
    else:
        print("ALL VALIDATION CHECKS PASSED")
        print("Review data limitations before calling this production-ready.")
    print("=" * 60)

if __name__ == "__main__":
    main()