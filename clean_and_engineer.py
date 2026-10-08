"""
Phases 5 (data quality fixes) & 7 (feature engineering)
------------------------------------------------------------
Every transformation and engineered feature below has a one-line WHY
comment next to it. FINAL FEATURE LIST is built at the bottom as
FEATURE_COLUMNS.
"""

import numpy as np
import pandas as pd

DELAY_THRESHOLD_MIN = 15
WEATHER_FEATURES = ["temp", "dewp", "humid", "wind_speed", "precip", "pressure", "visib"]
ONE_HOT_COLUMNS = ["carrier", "origin"]


def clean_data(flights: pd.DataFrame, weather: pd.DataFrame, planes: pd.DataFrame):
    df = flights.copy()

    # -----------------------------------------------------------------
    # Fix 1: exclude cancelled flights (no dep_time recorded).
    # WHY: a cancelled flight has no departure delay by definition --
    # including these rows with some placeholder value would either
    # bias the target or require a fundamentally different model
    # (predicting cancellation, not delay severity). Scoped out
    # explicitly, verified in eda.py to be 2.45% of all flights.
    # -----------------------------------------------------------------
    df = df[df["dep_time"].notna()].copy()

    # -----------------------------------------------------------------
    # Fix 2: build the binary target using the FAA/DOT standard
    # definition (>= 15 minutes late).
    # -----------------------------------------------------------------
    df["is_delayed"] = (df["dep_delay"] >= DELAY_THRESHOLD_MIN).astype(int)

    # -----------------------------------------------------------------
    # Fix 3: join weather on (origin, year, month, day, hour) -- the
    # natural shared key between the two tables.
    # -----------------------------------------------------------------
    weather_cols = ["origin", "year", "month", "day", "hour"] + WEATHER_FEATURES
    df = df.merge(weather[weather_cols], on=["origin", "year", "month", "day", "hour"], how="left")

    # WHY wind_gust IS DROPPED ENTIRELY, NOT IMPUTED (a different call
    # than every other project in this series, which imputed missing
    # values): EDA found wind_gust missing in 79.6% of weather records
    # -- imputing a column that's four-fifths absent would mean
    # inventing the large majority of its values from a handful of
    # real ones. At that missingness level, dropping the column
    # entirely is the more honest choice than pretending imputation
    # recovers real signal. pressure (10.4% missing) is a genuinely
    # different case -- imputed below, the same way Day 16's insulin/
    # skin-thickness columns were.
    if "wind_gust" in df.columns:
        df = df.drop(columns=["wind_gust"])

    # -----------------------------------------------------------------
    # Fix 4: join plane manufacture year (from planes.csv, keyed by
    # tailnum) to engineer plane age below.
    # WHY LEFT JOIN, NOT INNER: only 82.8% of flights match a plane
    # record (some tail numbers are missing from the planes table, a
    # real data-completeness gap, not something to silently drop rows
    # over) -- a missingness flag captures this explicitly rather than
    # losing 17.2% of flights entirely.
    # -----------------------------------------------------------------
    df = df.merge(planes[["tailnum", "year"]].rename(columns={"year": "plane_year"}), on="tailnum", how="left")
    df["plane_age"] = df["year"] - df["plane_year"]
    df["plane_age_missing"] = df["plane_age"].isnull().astype(int)

    return df


def engineer_features(clean_df: pd.DataFrame, train_medians: dict = None):
    """
    WHY train_medians IS A PARAMETER: the same leakage-safe pattern used
    in the diabetes-screening project -- pressure and plane_age
    imputation medians must come from the TRAINING split only, computed
    once and reused for validation, test, and live serving data.
    """
    df = clean_df.copy()

    impute_cols = ["pressure", "plane_age"]
    if train_medians is None:
        train_medians = {col: df[col].median() for col in impute_cols}
    for col in impute_cols:
        df[col] = df[col].fillna(train_medians[col])

    # WHY fillna(median) IS SAFE for the other weather columns (unlike
    # pressure/plane_age): temp/dewp/humid/wind_speed/precip/visib were
    # missing only 0-4 rows total across the whole weather table (see
    # EDA) -- a handful of genuinely missing hourly readings, not a
    # systematic gap, so no dedicated missingness flag is needed for
    # such a tiny number of rows.
    for col in ["temp", "dewp", "humid", "wind_speed", "precip", "visib"]:
        if col not in train_medians:
            train_medians[col] = df[col].median()
        df[col] = df[col].fillna(train_medians[col])

    # -----------------------------------------------------------------
    # Engineered feature: day_of_week
    # BUSINESS LOGIC: weekday travel patterns (business travel Mon/Fri
    # peaks) differ from weekend leisure patterns -- a real signal
    # distinct from month-level seasonality already captured.
    # -----------------------------------------------------------------
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df["day_of_week"] = df["date"].dt.dayofweek

    df = pd.get_dummies(df, columns=ONE_HOT_COLUMNS, prefix=ONE_HOT_COLUMNS)
    dummy_cols = [c for c in df.columns if any(c.startswith(p + "_") for p in ONE_HOT_COLUMNS)]
    df[dummy_cols] = df[dummy_cols].astype(int)

    return df, train_medians


NUMERIC_MODEL_FEATURES = [
    "hour", "month", "day_of_week", "distance",
    "temp", "dewp", "humid", "wind_speed", "precip", "pressure", "visib",
    "plane_age", "plane_age_missing",
]


def get_feature_columns(engineered_df: pd.DataFrame) -> list:
    dummy_cols = [c for c in engineered_df.columns if any(c.startswith(p + "_") for p in ONE_HOT_COLUMNS)]
    return NUMERIC_MODEL_FEATURES + dummy_cols


if __name__ == "__main__":
    from data_loader import load_raw_data

    flights, weather, airlines, planes = load_raw_data()
    cleaned = clean_data(flights, weather, planes)
    print(f"Flights after excluding cancellations: {len(cleaned)}")

    engineered, train_medians = engineer_features(cleaned)
    feature_cols = get_feature_columns(engineered)
    print(f"Final feature columns: {len(feature_cols)}")
    print("Imputation medians (from this full-data run; train_model.py recomputes from train split only):")
    print(train_medians)

    engineered[feature_cols + ["is_delayed"]].to_csv("outputs/engineered_data_sample.csv", index=False)
    print("\nSaved a feature-columns-only sample -> outputs/engineered_data_sample.csv")
