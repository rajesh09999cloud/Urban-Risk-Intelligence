from fastapi import APIRouter
import pandas as pd

router = APIRouter()

# Load incidents once at module level
try:
    _df = pd.read_csv("chicago_crimes_2023.csv", low_memory=False)
    _df.columns = [c.strip().lower().replace(" ","_") for c in _df.columns]
    _df = _df.rename(columns={
        "primary_type": "Primary Type",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "date": "Date"
    })
    _df['Latitude']  = pd.to_numeric(_df['Latitude'],  errors='coerce')
    _df['Longitude'] = pd.to_numeric(_df['Longitude'], errors='coerce')
    _df = _df.dropna(subset=['Latitude','Longitude'])
except Exception as e:
    _df = pd.DataFrame()
    print(f"Warning: could not load incidents CSV: {e}")

@router.get("/incidents")
def get_incidents(
    lat: float = 41.8781,
    lon: float = -87.6298,
    radius_km: float = 1.0,
    limit: int = 100
):
    if _df.empty:
        return {"incidents": [], "total": 0}

    # Simple bounding box filter (fast approximation)
    deg = radius_km / 111.0
    nearby = _df[
        (_df['Latitude']  >= lat - deg) & (_df['Latitude']  <= lat + deg) &
        (_df['Longitude'] >= lon - deg) & (_df['Longitude'] <= lon + deg)
    ].head(limit)

    records = []
    for _, row in nearby.iterrows():
        records.append({
            "type":      str(row.get('Primary Type', 'UNKNOWN')),
            "date":      str(row.get('Date', '')),
            "lat":       float(row['Latitude']),
            "lon":       float(row['Longitude']),
            "arrest":    str(row.get('arrest', 'false')),
        })

    return {"incidents": records, "total": len(records)}