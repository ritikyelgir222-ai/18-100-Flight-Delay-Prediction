# Project Documentation: Flight Delay Prediction
### Technical Documentation & Handover — Phase 12 of the SDLC

This document is the technical documentation and user manual a real handover
package would include. Every number below came from actually running the
code in this repository — none are illustrative.

---

## 1. Project Summary

| | |
|---|---|
| **Objective** | Flag flights at elevated departure-delay risk to support proactive crew/gate planning |
| **Client context** | Airline operations control center |
| **Data source** | nycflights13 — flights + weather + planes + airlines, real 2013 data |
| **Dataset size** | 328,521 flights (after excluding cancellations), 32 features |
| **Final model** | XGBoost Classifier |
| **Test AUC** | 0.6967 (test period: Nov-Dec, a different season than training) |
| **Business framing** | Catch delayed flights within an ops center's realistic proactive-intervention capacity (top 20% highest-risk) |

---

## 2. Architecture

```
data/flights.csv, weather.csv, planes.csv, airlines.csv
        │
        ▼
data_loader.py ──────► loads all 4 tables
        │
        ▼
clean_and_engineer.py ─► excludes cancelled flights, builds the target,
        │                joins weather (origin+date+hour) and plane
        │                metadata (tailnum), DROPS wind_gust (79.6%
        │                missing), leakage-safe median imputation
        ▼
   split.py ───────────► CHRONOLOGICAL split: train Jan-Aug,
        │                val Sep-Oct, test Nov-Dec
        ▼
train_model.py ───────► baseline, Random Forest, XGBoost;
        │                PROGRAMMATIC selection; business validation
        ▼
outputs/delay_model.joblib, feature_columns.joblib, train_medians.joblib
        │
        ├──────────────► app.py ─── FastAPI /predict endpoint
        │
        └──────────────► monitor.py ─ PSI drift check on genuinely
                                       future data (Nov-Dec)
```

---

## 3. Data Preparation

| Step | Detail |
|---|---|
| Exclude cancellations | 8,255 of 336,776 flights (2.45%) have no `dep_time` — never departed, out of scope for delay prediction |
| Build target | `is_delayed = dep_delay >= 15` (the FAA/DOT standard definition) |
| Join weather | On `(origin, year, month, day, hour)` — the natural shared key |
| **Drop `wind_gust`** | 79.6% missing across the weather table — imputing would mean inventing 4/5 of the column; dropped entirely instead |
| Join plane metadata | On `tailnum`, left join (82.8% match rate) → engineer `plane_age`; 17.2% unmatched flights get an imputed value plus a `plane_age_missing` flag |
| Impute `pressure`, `plane_age`, and the 6 near-complete weather columns | Median, computed from the TRAINING split only, reused for val/test/serving |

---

## 4. Key EDA Findings

| Finding | Detail |
|---|---|
| Overall delay rate | 22.2% (>=15 min, cancellations excluded) |
| **The delay cascade (the central finding)** | Delay rate climbs from ~6.3% at 5am to ~37.8% at 9pm, near-monotonically across every hour in between |
| Carrier variance | Ranges widely across carriers — carrier identity is a huge standalone signal |
| Origin airport | EWR highest, LGA lowest |
| Month (seasonality) | Summer (Jun-Jul) and December both spike; Sep-Nov lowest |
| Weather usability | `wind_gust` 79.6% missing (dropped); `pressure` 10.4% missing (imputed) |

---

## 5. Modeling Results

### Experiment log (validation set: Sep-Oct)

| Model | Val AUC | Val Recall @ Top 20% |
|---|---|---|
| Logistic Regression (baseline) | 0.6986 | 40.32% |
| Random Forest (30 trees, depth 10) | 0.6954 | 38.98% |
| **XGBoost** | **0.7086** | **42.28%** |

**XGBoost wins outright this time** — unlike Day 10 (demand forecasting)
and Day 15 (salary benchmarking), where Random Forest won, here XGBoost
beats both the baseline and Random Forest on both metrics, not just a
narrow tie-break. Selected programmatically, comparing raw unrounded
floats (per the rounding-bug lesson from the diabetes-screening project).

### Held-out test set (Nov-Dec — a genuinely different season)

| Metric | Value |
|---|---|
| Test AUC | 0.6967 |
| Test recall @ top 20% | 37.84% |
| Test precision @ top-20% threshold | 41.29% |
| N test flights | 54,145 |
| N test delayed | 11,815 |

**This AUC (~0.70) is more modest than most classification projects in
this series**, and that's an honest reflection of the problem, not a
weak model. Real-time operational factors that materially affect
whether a specific flight gets delayed — live air-traffic-control
decisions, the status of the SPECIFIC aircraft's prior leg that day,
gate availability, ground crew scheduling — aren't in this dataset at
all. The model does meaningfully better than random (AUC 0.70 vs. 0.50)
using only schedule, weather, and static aircraft-age information, which
is a fair statement of what's achievable from this feature set alone.

### What drives the model (feature importance)

Top 5 by importance:

1. `hour` (0.1470) — the delay-cascade effect
2. `carrier_EV` (0.1431) — ExpressJet, the highest-delay carrier in EDA
3. `carrier_US` (0.0612)
4. `carrier_DL` (0.0477)
5. `dewp` (0.0344)

`hour` and the top carrier dummy alone account for nearly 29% of the
model's total decision weight — directly confirming the two strongest
EDA findings (the delay cascade and carrier variance) rather than the
model discovering something EDA missed.

### Business validation

No dollar-value business case is calculated — a deliberate choice, the
same reasoning as the diabetes-screening project: this dataset has no
per-flight cost or cascading-delay-cost data, and fabricating one would
be misleading rather than illustrative. The honest, measured metric: at
a realistic ops-capacity threshold (the highest-risk 20% of flights),
the model catches **37.84%** of flights that actually get delayed, on a
test period from a different, higher-delay season than most of training
— a genuinely harder bar than a same-season evaluation would set.

---

## 6. API Reference (Phase 10)

**Base URL (local):** `http://127.0.0.1:8000`

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Test form (weather/plane-age optional, falls back to training medians) |
| `/docs` | GET | Swagger UI |
| `/health` | GET | Health check |
| `/predict` | POST | Score a single scheduled flight |

**Response (verified, two contrasting real profiles):**
```json
{
  "delay_risk_score": 0.8119,
  "risk_tier": "High",
  "top_factors": ["hour", "carrier_EV", "carrier_US"]
}
```
A Hawaiian Airlines 6am October flight from JFK scored **0.1635** on the
same model — confirming the API responds to genuinely different inputs.

---

## 7. Monitoring & Maintenance Plan (Phase 13)

- **Feature drift (PSI)** on `hour`, `temp`, `precip`, `distance`:
  `temp` shows significant drift (PSI 2.60) between train (Jan-Aug,
  includes summer) and test (Nov-Dec, genuinely colder) — **expected
  seasonal coverage difference, not a data problem**, the same
  interpretation applied to analogous findings in the sales-forecasting
  and e-commerce-demand projects. `hour`, `precip`, and `distance` all
  stayed stable.
- **Performance decay check**: no retrain triggered (same test set as
  training-time evaluation, so recall matches exactly).

**In production**, this should run monthly against the actual next
month's flights — this dataset's chronological structure means the
monitoring script's "new batch" here genuinely is future data, unlike
the cross-sectional projects in this series.

---

## 8. Known Limitations (stated for the handover record)

1. **Single year (2013), single origin metro (3 NYC airports)** — needs
   revalidation before deploying on other airports or more recent years.
2. **`wind_gust` dropped entirely** — a real information loss, traded
   against not fabricating 79.6% of a column's values.
3. **17.2% of flights lack a matched plane record** — imputed plane age
   plus an explicit missingness flag.
4. **No dollar-value business case** — see Section 5.
5. **Modest AUC (~0.70)**, an honest ceiling given the feature set —
   real-time operational data (live ATC decisions, specific aircraft's
   same-day prior-leg status) would likely improve this substantially
   but isn't available here.

---

## 9. File Map (for quick reference)

| File | Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Load all 4 tables, confirm cancellation/missingness patterns |
| `eda.py` | 6 | The delay-cascade finding, carrier/origin/month cuts |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Multi-table joins, the wind_gust drop, leakage-safe imputation |
| `split.py` | 7 | Chronological (month-based) split |
| `train_model.py` | 8-9 | Model training, programmatic selection, business validation |
| `app.py` | 10 | FastAPI delay-risk service |
| `monitor.py` | 13 | Drift detection on genuinely future data |
| `README.md` | 12 | Setup and run instructions |
| `PROJECT_DOCUMENTATION.md` (this file) | 12 | Technical documentation and handover |
