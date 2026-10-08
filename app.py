"""
Phase 10: MLOps & Deployment
--------------------------------
A minimal FastAPI service that scores a single scheduled flight's
departure-delay risk. Reuses clean_and_engineer.py's imputation medians
(saved at training time) so training and serving logic never drift apart.

Run with:  uvicorn app:app --reload
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Flight Delay Risk API", version="1.0")

MODEL = joblib.load("outputs/delay_model.joblib")
FEATURE_COLUMNS = joblib.load("outputs/feature_columns.joblib")
TRAIN_MEDIANS = joblib.load("outputs/train_medians.joblib")

CARRIERS = sorted([c.replace("carrier_", "") for c in FEATURE_COLUMNS if c.startswith("carrier_")])
ORIGINS = sorted([c.replace("origin_", "") for c in FEATURE_COLUMNS if c.startswith("origin_")])


class FlightRecord(BaseModel):
    carrier: str
    origin: str  # "EWR" | "JFK" | "LGA"
    hour: int  # scheduled departure hour, 0-23
    month: int
    day_of_week: int  # 0=Monday .. 6=Sunday
    distance: int
    temp: Optional[float] = None
    dewp: Optional[float] = None
    humid: Optional[float] = None
    wind_speed: Optional[float] = None
    precip: Optional[float] = None
    pressure: Optional[float] = None
    visib: Optional[float] = None
    plane_age: Optional[float] = None  # leave blank if unknown -- will be imputed the same way training data was


class DelayRiskResponse(BaseModel):
    delay_risk_score: float
    risk_tier: str
    top_factors: list[str]


def tier_from_score(score: float) -> str:
    if score >= 0.5:
        return "High"
    if score >= 0.25:
        return "Medium"
    return "Low"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    carrier_opts = "".join(f"<option>{c}</option>" for c in CARRIERS)
    origin_opts = "".join(f"<option>{o}</option>" for o in ORIGINS)
    return f"""
    <html>
    <head><title>Flight Delay Risk</title></head>
    <body style="font-family: sans-serif; max-width: 640px; margin: 40px auto;">
        <h2>Flight Delay Risk — Test Form</h2>
        <p>Leave weather/plane-age fields blank to use training-time medians
           (as if forecast/plane data weren't available yet). Full API docs
           at <a href="/docs">/docs</a>.</p>
        <form id="riskForm" style="display:grid; grid-template-columns: 1fr 1fr; gap: 10px 20px;">
            <label>Carrier<br><select name="carrier" required>{carrier_opts}</select></label>
            <label>Origin<br><select name="origin" required>{origin_opts}</select></label>
            <label>Scheduled hour (0-23)<br><input name="hour" type="number" value="20" required></label>
            <label>Month (1-12)<br><input name="month" type="number" value="7" required></label>
            <label>Day of week (0=Mon)<br><input name="day_of_week" type="number" value="4" required></label>
            <label>Distance (miles)<br><input name="distance" type="number" value="1000" required></label>
            <button type="submit" style="grid-column: 1 / -1; margin-top: 10px;">Check risk</button>
        </form>
        <h3 id="result"></h3>
        <script>
        document.getElementById("riskForm").addEventListener("submit", async function(e) {{
            e.preventDefault();
            const form = new FormData(e.target);
            const payload = {{
                carrier: form.get("carrier"), origin: form.get("origin"),
                hour: parseInt(form.get("hour"), 10), month: parseInt(form.get("month"), 10),
                day_of_week: parseInt(form.get("day_of_week"), 10),
                distance: parseInt(form.get("distance"), 10)
            }};
            const res = await fetch("/predict", {{
                method: "POST",
                headers: {{"Content-Type": "application/json"}},
                body: JSON.stringify(payload)
            }});
            if (!res.ok) {{
                document.getElementById("result").innerText = "Error " + res.status + ": " + await res.text();
                return;
            }}
            const data = await res.json();
            document.getElementById("result").innerText =
                "Risk score: " + data.delay_risk_score + " | Tier: " + data.risk_tier +
                " | Top factors: " + data.top_factors.join(", ");
        }});
        </script>
    </body>
    </html>
    """


@app.post("/predict", response_model=DelayRiskResponse)
def predict_delay(record: FlightRecord):
    raw = record.model_dump()

    row = {
        "hour": raw["hour"], "month": raw["month"], "day_of_week": raw["day_of_week"],
        "distance": raw["distance"],
        "temp": raw["temp"] if raw["temp"] is not None else TRAIN_MEDIANS["temp"],
        "dewp": raw["dewp"] if raw["dewp"] is not None else TRAIN_MEDIANS["dewp"],
        "humid": raw["humid"] if raw["humid"] is not None else TRAIN_MEDIANS["humid"],
        "wind_speed": raw["wind_speed"] if raw["wind_speed"] is not None else TRAIN_MEDIANS["wind_speed"],
        "precip": raw["precip"] if raw["precip"] is not None else TRAIN_MEDIANS["precip"],
        "pressure": raw["pressure"] if raw["pressure"] is not None else TRAIN_MEDIANS["pressure"],
        "visib": raw["visib"] if raw["visib"] is not None else TRAIN_MEDIANS["visib"],
        "plane_age": raw["plane_age"] if raw["plane_age"] is not None else TRAIN_MEDIANS["plane_age"],
        "plane_age_missing": 1 if raw["plane_age"] is None else 0,
    }
    for c in CARRIERS:
        row[f"carrier_{c}"] = 1 if c == raw["carrier"] else 0
    for o in ORIGINS:
        row[f"origin_{o}"] = 1 if o == raw["origin"] else 0

    X = pd.DataFrame([row]).reindex(columns=FEATURE_COLUMNS, fill_value=0)

    proba = float(MODEL.predict_proba(X)[0, 1])
    tier = tier_from_score(proba)

    if hasattr(MODEL, "feature_importances_"):
        importances = pd.Series(MODEL.feature_importances_, index=FEATURE_COLUMNS)
        top_factors = importances.sort_values(ascending=False).head(3).index.tolist()
    else:
        top_factors = ["hour", "carrier", "month"]

    return DelayRiskResponse(
        delay_risk_score=round(proba, 4),
        risk_tier=tier,
        top_factors=top_factors,
    )
