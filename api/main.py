from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
model = joblib.load(ROOT / "models" / "xgb_price_model.pkl")
df = pd.read_csv(ROOT / "data" / "cleaned_flights_analysis.csv")

FEATURES = ['Airline','Source','Destination','Total_Stops','Distance_km',
            'Travel_Class','Days_Before_Departure','Season','Weekday',
            'Aircraft_Type','Booking_Channel','Passenger_Count','Duration_mins']

app = FastAPI(title="AeroLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],   # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/options")
def options():
    """Dropdown values for the frontend."""
    return {
        "sources": sorted(df['Source'].unique().tolist()),
        "destinations": sorted(df['Destination'].unique().tolist()),
        "classes": ['Economy', 'Premium Economy', 'Business', 'First'],
        "airlines": sorted(df['Airline'].unique().tolist()),
    }


def _template(source, destination, travel_class):
    """Borrow a real flight so feature combinations stay plausible."""
    pool = df[(df['Source'] == source) &
              (df['Destination'] == destination) &
              (df['Travel_Class'] == travel_class)]
    if pool.empty:
        raise HTTPException(404, f"No flights found: {source} -> {destination} ({travel_class})")
    return pool


class PredictRequest(BaseModel):
    source: str
    destination: str
    travel_class: str
    days_before: int


@app.post("/api/predict")
def predict(req: PredictRequest):
    pool = _template(req.source, req.destination, req.travel_class)
    sample = pool.sample(min(50, len(pool)), random_state=42)

    rows = []
    for _, r in sample.iterrows():
        f = {c: r[c] for c in FEATURES}
        f['Days_Before_Departure'] = req.days_before
        rows.append(f)

    preds = model.predict(pd.DataFrame(rows)[FEATURES])
    return {
        "predicted_price": round(float(np.mean(preds)), 2),
        "low": round(float(np.percentile(preds, 25)), 2),
        "high": round(float(np.percentile(preds, 75)), 2),
        "distance_km": round(float(sample['Distance_km'].median()), 1),
        "n_reference_flights": int(len(pool)),
    }


@app.post("/api/forecast")
def forecast(req: PredictRequest):
    """Price curve across every booking lead time, 0-120 days."""
    pool = _template(req.source, req.destination, req.travel_class)
    sample = pool.sample(min(100, len(pool)), random_state=42)

    curves = []
    for _, r in sample.iterrows():
        base = {c: r[c] for c in FEATURES}
        grid = []
        for d in range(121):
            f = base.copy()
            f['Days_Before_Departure'] = d
            grid.append(f)
        curves.append(model.predict(pd.DataFrame(grid)[FEATURES]))

    avg = np.mean(curves, axis=0)
    smoothed = pd.Series(avg).rolling(7, center=True, min_periods=1).mean()

    floor = float(smoothed[90:].mean())
    threshold = int(smoothed[smoothed <= floor * 1.05].index.min())
    early = float(smoothed[60:121].mean())
    late = float(smoothed[0:4].mean())

    return {
        "curve": [{"days": d, "price": round(float(p), 2)} for d, p in enumerate(smoothed)],
        "book_by_days": threshold,
        "early_price": round(early, 2),
        "late_price": round(late, 2),
        "premium_pct": round((late / early - 1) * 100, 1),
    }


class CompareRequest(BaseModel):
    source: str
    destination: str


@app.post("/api/compare")
def compare(req: CompareRequest):
    """Forecast curves for every cabin class available on a route."""
    out = []
    for cls in ['Economy', 'Premium Economy', 'Business', 'First']:
        pool = df[(df['Source'] == req.source) &
                  (df['Destination'] == req.destination) &
                  (df['Travel_Class'] == cls)]
        if len(pool) < 10:
            continue

        sample = pool.sample(min(60, len(pool)), random_state=42)
        curves = []
        for _, r in sample.iterrows():
            base = {c: r[c] for c in FEATURES}
            grid = [{**base, 'Days_Before_Departure': d} for d in range(121)]
            curves.append(model.predict(pd.DataFrame(grid)[FEATURES]))

        avg = pd.Series(np.mean(curves, axis=0)).rolling(7, center=True, min_periods=1).mean()
        floor = float(avg[90:].mean())
        early = float(avg[60:121].mean())
        late = float(avg[0:4].mean())

        out.append({
            "travel_class": cls,
            "n_flights": int(len(pool)),
            "premium_pct": round((late / early - 1) * 100, 1),
            "book_by_days": int(avg[avg <= floor * 1.05].index.min()),
            "indexed": [{"days": d, "index": round(float(p / floor * 100), 1)}
                        for d, p in enumerate(avg)],
        })

    if not out:
        raise HTTPException(404, "No cabin classes with sufficient data on this route")
    return {"classes": out}
