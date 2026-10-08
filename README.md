# Flight Delay Prediction — Full Project Code (Real Dataset)

Companion code to Day 18 of the 100-day LinkedIn series. Uses the real
nycflights13 dataset (336,776 real 2013 flights from NYC's three
airports, joined with real hourly weather and aircraft metadata), run
end to end.

## Data source

**nycflights13** — flights, weather, airlines, and planes tables, all
real US Bureau of Transportation Statistics-derived data. All 5 CSVs are
included at `data/`.

## The key finding: the delay cascade

The single strongest, most visually obvious pattern in this data: delay
rate climbs **near-monotonically from ~6% at 5am to ~38% at 9pm**. This
is the well-documented "delay cascade" — the same aircraft and crew fly
multiple legs per day, so a delay earlier in the day propagates and
compounds by evening. `hour` alone is one of the two strongest features
in the final model (alongside carrier identity).

## What's genuinely different about this project

1. **A real multi-table join**, not a single pre-packaged CSV: flights
   + weather (hourly, keyed by origin/date/hour) + plane metadata (keyed
   by tail number, 82.8% match rate) + carrier names.
2. **A column dropped entirely rather than imputed** — `wind_gust` is
   79.6% missing across the weather table. Every other project in this
   series that hit missing data imputed it (with a flag); here, that
   much missingness means imputation would be inventing four-fifths of
   the column, so it's dropped instead — a genuinely different call,
   explained in `clean_and_engineer.py`.
3. **A third split methodology for this series**: chronological by
   month (train Jan-Aug, validate Sep-Oct, test **Nov-Dec** — a
   genuinely different, higher-delay season than most of training).
   This is a harder, more honest test than a random split would give.
4. **XGBoost wins outright this time** — not a near-tie, not Random
   Forest (which has won twice elsewhere in this series). See
   `train_model.py`'s experiment log.

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python data_loader.py         # Phase 5 — loads all 5 tables, confirms cancellation/missingness patterns
python eda.py                  # Phase 6 — the delay-cascade finding, carrier/month/origin cuts + outputs/eda_summary.png
python clean_and_engineer.py   # Phase 5 (fixes) + 7 — multi-table joins, the wind_gust drop, leakage-safe imputation
python train_model.py          # Phase 8-9 — baseline + final model (chronological split), business validation
python monitor.py              # Phase 13 — drift check on genuinely future data (Nov-Dec)
```

## Serve the model (Phase 10)

```bash
uvicorn app:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"carrier": "EV", "origin": "EWR", "hour": 21, "month": 7, "day_of_week": 4, "distance": 500}'
```

Expected: a risk score around **0.81** and `"risk_tier": "High"` — this
profile (the highest-delay carrier, the peak-delay hour, peak summer
month) matches exactly what EDA identifies as highest-risk.

## File map

| File | SDLC Phase | What it does |
|---|---|---|
| `data_loader.py` | 5 | Loads all 5 tables, confirms cancellation and weather-missingness patterns |
| `eda.py` | 6 | The delay-cascade finding, carrier/origin/month cuts |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Multi-table joins, **the wind_gust drop**, leakage-safe median imputation |
| `split.py` | 7 | Chronological (month-based) train/val/test split |
| `train_model.py` | 8-9 | Baseline → Random Forest → XGBoost, programmatic selection, business validation |
| `app.py` | 10 | FastAPI delay-risk service |
| `monitor.py` | 13 | PSI drift check on genuinely future data |

## Outputs produced (in `outputs/`)

- `eda_summary.png`, `experiment_log.csv`
- `business_validation.json` — AUC, top-20%-recall, no fabricated dollar figure
- `feature_importance.csv`
- `delay_model.joblib`, `feature_columns.joblib`, `train_medians.joblib`, `model_card.json`

## Known limitations (stated honestly, not hidden)

- **Single year, single origin metro (3 NYC airports)** — does not
  generalize to other airports/years without revalidation.
- **`wind_gust` dropped entirely** (79.6% missing) — a real loss of
  potentially useful weather signal, traded for not fabricating
  four-fifths of a column.
- **17.2% of flights have no matching plane record** — `plane_age` is
  imputed with a training-median for these, flagged via
  `plane_age_missing`.
- **No dollar-value business case** — no per-flight cost or cascading-
  delay-cost data exists in this dataset; the catch-rate at a realistic
  ops-capacity threshold is the honest, measured metric instead.
- **Modest test AUC** (~0.70) — an honest reflection that real-time
  operational factors (live air-traffic-control decisions, upstream
  aircraft rotation status, gate availability) aren't captured in this
  dataset at all; flight delay is a genuinely harder prediction problem
  than several other classification projects in this series.
