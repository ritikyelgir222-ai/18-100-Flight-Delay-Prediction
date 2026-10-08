"""
Phase 13: Monitoring & Maintenance
--------------------------------------
WHY THIS SCRIPT'S "NEW BATCH" IS GENUINELY FUTURE DATA (like the sales-
forecasting and e-commerce-demand projects, unlike the cross-sectional
projects in this series): the test set is the actual Nov-Dec period,
chronologically after training (Jan-Aug) and validation (Sep-Oct) --
so drift found here reflects a real seasonal shift within the year, not
a same-snapshot artifact.
"""

import json

import numpy as np
import joblib

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns
from split import split_data
from train_model import recall_at_top_quintile

RECALL_DROP_THRESHOLD = 0.10
DRIFT_FEATURES = ["hour", "temp", "precip", "distance"]


def population_stability_index(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def run_monitoring_check():
    model = joblib.load("outputs/delay_model.joblib")
    feature_cols = joblib.load("outputs/feature_columns.joblib")
    train_medians = joblib.load("outputs/train_medians.joblib")

    flights, weather, airlines, planes = load_raw_data()
    cleaned = clean_data(flights, weather, planes)
    train_raw, val_raw, test_raw = split_data(cleaned)

    train_eng, _ = engineer_features(train_raw)
    test_eng, _ = engineer_features(test_raw, train_medians=train_medians)

    X_train = train_eng.reindex(columns=feature_cols, fill_value=0)
    X_test = test_eng.reindex(columns=feature_cols, fill_value=0)
    y_test = test_eng["is_delayed"]

    print("=== Feature Drift (PSI): train (Jan-Aug) vs. test (Nov-Dec) ===")
    print("PSI < 0.1: stable | 0.1-0.25: moderate | > 0.25: significant drift\n")
    for feat in DRIFT_FEATURES:
        psi = population_stability_index(X_train[feat], X_test[feat])
        flag = "SIGNIFICANT DRIFT" if psi > 0.25 else ("moderate" if psi > 0.1 else "ok")
        print(f"  {feat}: PSI={psi:.4f} [{flag}]")
    # WHY temp DRIFT IS EXPECTED, NOT A PROBLEM: Nov-Dec is genuinely
    # colder than the Jan-Aug training window's average (which includes
    # summer) -- the same "expected seasonal coverage difference" found
    # in the sales-forecasting and e-commerce-demand projects' monitors.

    scores = model.predict_proba(X_test)[:, 1]
    recall = recall_at_top_quintile(y_test, scores)

    print(f"\n=== Performance on held-out batch (genuinely future data) ===")
    print(f"Recall at top 20%: {recall:.4f}")

    with open("outputs/business_validation.json") as f:
        baseline_recall = json.load(f)["test_recall_at_top20pct"]

    drop = baseline_recall - recall
    if drop > RECALL_DROP_THRESHOLD:
        print(f"\n⚠️  RETRAIN TRIGGERED: recall dropped by {drop:.3f} (threshold: {RECALL_DROP_THRESHOLD})")
    else:
        print(f"\n✅ No retrain needed (recall change: {drop:.3f}, threshold: {RECALL_DROP_THRESHOLD})")

    print("\nNOTE: this run scores the model against the SAME Nov-Dec test set used at")
    print("training time (recall matches business_validation.json exactly) -- in production,")
    print("this should run monthly against the actual next month's flights.")


if __name__ == "__main__":
    run_monitoring_check()
