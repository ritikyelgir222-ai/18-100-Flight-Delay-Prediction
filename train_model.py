"""
Phase 8: Model Development & Phase 9: Evaluation & Business Validation
---------------------------------------------------------------------------
Every modeling choice below has a WHY comment.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns
from split import split_data


def recall_at_top_quintile(y_true, y_score, quintile=0.20):
    """
    BUSINESS-RELEVANT METRIC.
    WHY: an operations control center can only proactively act on
    (pre-position a relief crew, hold a gate, warn downstream
    connections) a limited share of that day's flights -- not every
    scheduled departure. Asking "of the riskiest 20% of flights we
    flag, what share of the ones that actually get delayed did we
    catch" is the operationally realistic question, the same
    capacity-constrained framing used for churn, attrition, and
    diabetes-screening in this series.
    """
    y_true = np.asarray(y_true)
    n_top = max(1, int(len(y_score) * quintile))
    top_idx = np.argsort(y_score)[-n_top:]
    caught = y_true[top_idx].sum()
    total_positive = y_true.sum()
    return caught / total_positive if total_positive > 0 else 0.0


def train_and_evaluate():
    flights, weather, airlines, planes = load_raw_data()
    cleaned = clean_data(flights, weather, planes)

    train_raw, val_raw, test_raw = split_data(cleaned)
    print(f"Train: {len(train_raw)} flights (Jan-Aug)")
    print(f"Val:   {len(val_raw)} flights (Sep-Oct)")
    print(f"Test:  {len(test_raw)} flights (Nov-Dec)")

    # WHY engineer_features IS CALLED SEPARATELY PER SPLIT, REUSING
    # TRAIN-ONLY MEDIANS: the same leakage-safe pattern as the
    # diabetes-screening project -- pressure/plane_age imputation must
    # be computed from training data only.
    train_eng, train_medians = engineer_features(train_raw)
    val_eng, _ = engineer_features(val_raw, train_medians=train_medians)
    test_eng, _ = engineer_features(test_raw, train_medians=train_medians)

    feature_cols = get_feature_columns(train_eng)
    X_train = train_eng.reindex(columns=feature_cols, fill_value=0)
    X_val = val_eng.reindex(columns=feature_cols, fill_value=0)
    X_test = test_eng.reindex(columns=feature_cols, fill_value=0)
    y_train, y_val, y_test = train_eng["is_delayed"], val_eng["is_delayed"], test_eng["is_delayed"]

    experiment_log = []

    # -----------------------------------------------------------------
    # Baseline: Logistic Regression
    # WHY class_weight="balanced": ~22% positive rate.
    # WHY SCALE FEATURES HERE: coefficients sensitive to feature scale
    # (distance ranges into the thousands, day_of_week is 0-6).
    # -----------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_scaled, y_train)
    logreg_val_scores = logreg.predict_proba(X_val_scaled)[:, 1]
    experiment_log.append({
        "model": "logistic_regression (baseline)",
        "val_auc": round(roc_auc_score(y_val, logreg_val_scores), 4),
        "val_recall_at_top20pct": round(recall_at_top_quintile(y_val, logreg_val_scores), 4),
    })

    # -----------------------------------------------------------------
    # Candidate: Random Forest
    # WHY A REDUCED FOREST (30 trees, depth 10): ~230K training rows on
    # a single CPU core -- the same compute-budget tradeoff made
    # explicit in the sales-forecasting and fraud-detection projects.
    # -----------------------------------------------------------------
    rf = RandomForestClassifier(n_estimators=30, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_val_scores = rf.predict_proba(X_val)[:, 1]
    experiment_log.append({
        "model": "random_forest",
        "val_auc": round(roc_auc_score(y_val, rf_val_scores), 4),
        "val_recall_at_top20pct": round(recall_at_top_quintile(y_val, rf_val_scores), 4),
    })

    # -----------------------------------------------------------------
    # Final candidate: XGBoost
    # WHY tree_method="hist": dramatically faster at this row count with
    # negligible accuracy cost, the same choice made in the sales-
    # forecasting and fraud-detection projects.
    # SELECTION CRITERION: chosen by val_recall_at_top20pct, decided
    # PROGRAMMATICALLY (comparing raw, unrounded floats -- see the
    # rounding-bug lesson from the diabetes-screening project).
    # -----------------------------------------------------------------
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="auc",
        tree_method="hist", random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_val_scores = xgb.predict_proba(X_val)[:, 1]
    xgb_val_auc = roc_auc_score(y_val, xgb_val_scores)
    xgb_val_recall = recall_at_top_quintile(y_val, xgb_val_scores)
    experiment_log.append({
        "model": "xgboost",
        "val_auc": round(xgb_val_auc, 4),
        "val_recall_at_top20pct": round(xgb_val_recall, 4),
    })

    print("\n=== Experiment Log (validation set: Sep-Oct) ===")
    log_df = pd.DataFrame(experiment_log)
    print(log_df.to_string(index=False))
    log_df.to_csv("outputs/experiment_log.csv", index=False)

    rf_val_recall_raw = recall_at_top_quintile(y_val, rf_val_scores)  # raw float, not the rounded logged value
    if rf_val_recall_raw >= xgb_val_recall:
        final_model, final_name = rf, "random_forest"
        final_val_recall = rf_val_recall_raw
    else:
        final_model, final_name = xgb, "xgboost"
        final_val_recall = xgb_val_recall
    print(f"\n=== Selected final model: {final_name} (higher val_recall_at_top20pct) ===")

    # -----------------------------------------------------------------
    # Phase 9: Final evaluation on the held-out TEST set (Nov-Dec --
    # a different, higher-delay seasonal regime than most of training)
    # -----------------------------------------------------------------
    test_scores = final_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_scores)
    test_recall_top20 = recall_at_top_quintile(y_test, test_scores)

    threshold = np.percentile(test_scores, 80)
    test_preds = (test_scores >= threshold).astype(int)
    test_precision = precision_score(y_test, test_preds)
    test_recall_at_threshold = recall_score(y_test, test_preds)

    business_summary = {
        "final_model": final_name,
        "test_auc": round(test_auc, 4),
        "test_recall_at_top20pct": round(test_recall_top20, 4),
        "test_precision_at_top20pct_threshold": round(test_precision, 4),
        "test_recall_at_top20pct_threshold": round(test_recall_at_threshold, 4),
        "n_test_flights": len(y_test),
        "n_test_delayed": int(y_test.sum()),
        "test_period": "November-December (includes the December holiday-travel delay spike)",
        "note": (
            "Test period is a genuinely different, higher-delay season than most of "
            "training (Jan-Aug) -- this is a harder, more honest evaluation than a "
            "same-season random split would give. No dollar-value business case is "
            "calculated (no per-flight cost/cascading-delay-cost data available); "
            "the catch-rate at a realistic ops-capacity threshold (top 20% "
            "highest-risk flights) is the honest, directly-measured metric here."
        ),
    }

    print("\n=== Business Validation Summary (Phase 9) ===")
    for k, v in business_summary.items():
        print(f"{k}: {v}")
    with open("outputs/business_validation.json", "w") as f:
        json.dump(business_summary, f, indent=2)

    if hasattr(final_model, "feature_importances_"):
        importances = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("\n=== Top 10 Feature Importances (final model) ===")
        print(importances.head(10).round(4).to_string())
        importances.to_csv("outputs/feature_importance.csv", header=["importance"])

    joblib.dump(final_model, "outputs/delay_model.joblib")
    joblib.dump(feature_cols, "outputs/feature_columns.joblib")
    joblib.dump(train_medians, "outputs/train_medians.joblib")

    with open("outputs/model_card.json", "w") as f:
        json.dump({
            "model_type": "RandomForestClassifier" if final_name == "random_forest" else "XGBClassifier",
            "data_source": "nycflights13 (real 2013 NYC departures + weather + plane metadata)",
            "n_features": len(feature_cols),
            "features": feature_cols,
            "training_rows": len(X_train),
            "split_strategy": "chronological (train Jan-Aug, val Sep-Oct, test Nov-Dec), not random",
            "validation_recall_at_top20pct": round(final_val_recall, 4),
            "test_auc": round(test_auc, 4),
            "test_recall_at_top20pct": round(test_recall_top20, 4),
            "intended_use": "Flag flights at elevated departure-delay risk to support proactive crew/gate planning. Cancelled flights are out of scope (see clean_and_engineer.py).",
            "known_limitations": (
                "Trained on a single year (2013) of data from 3 NYC airports only -- "
                "does not generalize to other airports/years without revalidation. "
                "wind_gust was dropped entirely (79.6% missing) rather than imputed. "
                "17.2% of flights have no matching plane record and rely on an "
                "imputed plane_age plus a missingness flag."
            ),
        }, f, indent=2)

    print("\nSaved model -> outputs/delay_model.joblib")
    print("Saved model card -> outputs/model_card.json")


if __name__ == "__main__":
    train_and_evaluate()
