from fastapi import APIRouter, Request
from pydantic import BaseModel
import pandas as pd
import h3
from datetime import datetime

router = APIRouter()

class RiskRequest(BaseModel):
    lat: float
    lon: float
    datetime_str: str = None   # optional, defaults to now

@router.post("/predict/risk")
async def predict_risk(req: RiskRequest, request: Request):

    # Use current time if no datetime provided
    if req.datetime_str:
        dt = datetime.fromisoformat(req.datetime_str)
    else:
        dt = datetime.now()

    # Get H3 cell for this location
    cell_id = h3.latlng_to_cell(req.lat, req.lon, 8)

    # Look up static cell features
    cell_df = request.app.state.cell_features
    cell_row = cell_df[cell_df['h3_cell'] == cell_id]

    if len(cell_row) > 0:
        total_crimes     = float(cell_row['total_crimes_in_cell'].values[0])
        cell_violent_rate= float(cell_row['cell_violent_rate'].values[0])
        neighbour_avg    = float(cell_row['neighbour_avg_crimes'].values[0])
        cell_danger_rank = float(cell_row['total_crimes_in_cell'].rank(pct=True).values[0])
    else:
        # Cell not seen in training — use city averages
        total_crimes      = float(cell_df['total_crimes_in_cell'].mean())
        cell_violent_rate = float(cell_df['cell_violent_rate'].mean())
        neighbour_avg     = float(cell_df['neighbour_avg_crimes'].mean())
        cell_danger_rank  = 0.5

    # Look up rolling features
    daily_df = request.app.state.daily_features
    daily_df['date_only'] = pd.to_datetime(daily_df['date_only'])
    today = pd.Timestamp(dt.date())

    recent = daily_df[
        (daily_df['h3_cell'] == cell_id) &
        (daily_df['date_only'] <= today)
    ].sort_values('date_only').tail(1)

    if len(recent) > 0:
        crimes_last_7d  = float(recent['crimes_last_7d'].values[0])
        crimes_last_30d = float(recent['crimes_last_30d'].values[0])
        violent_last_7d = float(recent['violent_last_7d'].values[0])
    else:
        crimes_last_7d  = 0.0
        crimes_last_30d = 0.0
        violent_last_7d = 0.0

    # Hour and day risk — simple lookup from known patterns
    hour_risk_map = {
        0:0.28, 1:0.30, 2:0.32, 3:0.31, 4:0.27, 5:0.22,
        6:0.18, 7:0.16, 8:0.15, 9:0.14, 10:0.15, 11:0.16,
        12:0.18, 13:0.18, 14:0.19, 15:0.20, 16:0.21, 17:0.22,
        18:0.23, 19:0.24, 20:0.26, 21:0.28, 22:0.29, 23:0.29
    }
    dow_risk_map = {0:0.20, 1:0.19, 2:0.20, 3:0.20, 4:0.22, 5:0.24, 6:0.23}

    # Build feature row — must exactly match FEATURES_V3 order
    features = pd.DataFrame([{
        'hour':                 dt.hour,
        'day_of_week':          dt.weekday(),
        'month':                dt.month,
        'is_weekend':           int(dt.weekday() >= 5),
        'is_night':             int(dt.hour >= 20 or dt.hour <= 5),
        'district_encoded':     0,
        'location_encoded':     0,
        'total_crimes_in_cell': total_crimes,
        'cell_violent_rate':    cell_violent_rate,
        'neighbour_avg_crimes': neighbour_avg,
        'cell_danger_rank':     cell_danger_rank,
        'crimes_last_7d':       crimes_last_7d,
        'crimes_last_30d':      crimes_last_30d,
        'violent_last_7d':      violent_last_7d,
        'hour_risk':            hour_risk_map.get(dt.hour, 0.22),
        'dow_risk':             dow_risk_map.get(dt.weekday(), 0.21),
    }])

    # Predict
    model = request.app.state.model
    crime_prob = float(model.predict_proba(features)[0][1])

    # Composite score (just crime for now, expand later)
    composite = round(crime_prob, 4)

    return {
        "h3_cell_id":      cell_id,
        "lat":             req.lat,
        "lon":             req.lon,
        "crime_risk":      round(crime_prob, 4),
        "accident_risk":   0.0,
        "env_risk":        0.0,
        "composite_risk":  composite,
        "risk_level":      "HIGH" if composite > 0.6 else "MEDIUM" if composite > 0.35 else "LOW",
        "datetime":        dt.isoformat()
    }