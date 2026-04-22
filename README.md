# Urban Risk Intelligence System

A real-time machine learning system that predicts crime risk across Chicago using geospatial analysis, XGBoost modeling, and interactive map visualization.

**Live Demo:** [https://urban-risk-intelligence.netlify.app](https://urban-risk-intelligence.netlify.app)  
**API Docs:** [https://urban-risk-intelligence.onrender.com/docs](https://urban-risk-intelligence.onrender.com/docs)

---

## What It Does

This system analyzes 260,000+ Chicago crime incidents from 2023, assigns each incident to an H3 hexagonal grid cell, trains an XGBoost classifier to predict violent crime risk, and serves predictions through a live REST API visualized on an interactive map.

Users can:
- View color-coded risk hexagons covering all of Chicago
- Click any hexagon to see its risk score, crime count, and violent rate
- Toggle a crime heatmap with adjustable radius, blur, and opacity
- Switch to predict mode and click anywhere on the map for a live risk prediction
- View incident markers for individual crimes with type, date, and arrest info
- See a list of nearby incidents within 0.5km of any selected zone

---

## Architecture

```
Chicago Open Data API
        ↓
Data Ingestion (Python / Pandas)
        ↓
H3 Spatial Grid Assignment (Resolution 8)
        ↓
Feature Engineering
  - Time features (hour, day, month, is_weekend, is_night)
  - Spatial features (cell crime count, violent rate, neighbour avg)
  - Rolling windows (7-day, 30-day crime counts per cell)
  - Interaction features (hour_risk, dow_risk, cell_danger_rank)
        ↓
XGBoost Classifier (ROC-AUC: 0.75+)
        ↓
FastAPI Backend (5 endpoints)
        ↓
Leaflet.js Frontend Map
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data processing | Python, Pandas, GeoPandas |
| Spatial indexing | H3 (Uber Hexagonal Grid) |
| Machine learning | XGBoost, Scikit-learn |
| Backend API | FastAPI, Uvicorn |
| Database | MySQL (schema designed) |
| Frontend | HTML, CSS, JavaScript, Leaflet.js |
| Model storage | Google Drive + gdown |
| Backend deployment | Render.com (free tier) |
| Frontend deployment | Netlify (free tier) |

---

## ML Model Details

**Model:** XGBoost Classifier (V3 — with spatial features)

**Target variable:** `is_violent` — whether a crime incident is violent (battery, assault, robbery, homicide, kidnapping)

**Features used:**

| Feature | Description |
|---|---|
| `hour` | Hour of day (0-23) |
| `day_of_week` | Day of week (0=Mon, 6=Sun) |
| `month` | Month of year |
| `is_weekend` | 1 if Saturday or Sunday |
| `is_night` | 1 if hour >= 20 or hour <= 5 |
| `district_encoded` | Label-encoded police district |
| `location_encoded` | Label-encoded location description |
| `total_crimes_in_cell` | Total crimes in H3 cell (historical) |
| `cell_violent_rate` | % of crimes that are violent in this cell |
| `neighbour_avg_crimes` | Average crime count of 6 neighboring cells |
| `cell_danger_rank` | Percentile rank of cell by crime count |
| `crimes_last_7d` | Rolling 7-day crime count for this cell |
| `crimes_last_30d` | Rolling 30-day crime count for this cell |
| `violent_last_7d` | Rolling 7-day violent crime count |
| `hour_risk` | Historical violent rate for this hour |
| `dow_risk` | Historical violent rate for this day of week |

**Validation:** Temporal walk-forward cross-validation (no data leakage)

**Results:**

| Model | Features | ROC-AUC |
|---|---|---|
| V1 | Time features only | 0.7112 |
| V2 | Time + static spatial | 0.7174 |
| V3 | Time + spatial + rolling windows | 0.7199 |

---

## API Endpoints

Base URL: `https://urban-risk-intelligence.onrender.com`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | API status |
| POST | `/api/v1/predict/risk` | Predict crime risk for a lat/lon point |
| GET | `/api/v1/heatmap` | Get GeoJSON risk scores for all H3 cells |
| GET | `/api/v1/incidents` | Get nearby incidents for a location |

**Example predict request:**
```bash
curl -X POST "https://urban-risk-intelligence.onrender.com/api/v1/predict/risk" \
  -H "Content-Type: application/json" \
  -d '{"lat": 41.8781, "lon": -87.6298, "datetime_str": "2023-08-15 23:00:00"}'
```

**Example response:**
```json
{
  "h3_cell_id": "8828308281fffff",
  "crime_risk": 0.7823,
  "composite_risk": 0.7823,
  "risk_level": "HIGH",
  "datetime": "2023-08-15T23:00:00"
}
```

---

## Project Structure

```
Urban-Risk-Intelligence/
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entry point
│   └── routers/
│       ├── __init__.py
│       ├── predict.py           # /predict/risk endpoint
│       ├── heatmap.py           # /heatmap endpoint
│       └── incidents.py         # /incidents endpoint
├── frontend/
│   └── index.html               # Complete single-file frontend
├── models/                      # Downloaded at runtime from Google Drive
│   ├── crime_model_v3.pkl
│   ├── crime_features_v3.pkl
│   ├── cell_features.csv
│   └── daily_cell_features.csv
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Local Setup

### Prerequisites
- Python 3.10+
- Anaconda (recommended)
- Git

### Steps

**1. Clone the repository:**
```bash
git clone https://github.com/rajesh09999cloud/Urban-Risk-Intelligence.git
cd Urban-Risk-Intelligence
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the API server:**
```bash
python -m uvicorn api.main:app --reload --port 8000
```

The server will automatically download the model files from Google Drive on first startup.

**4. Open the frontend:**

Open `frontend/index.html` in your browser. Update the `API_URL` in the file to `http://localhost:8000` for local development.

**5. API docs available at:**
```
http://localhost:8000/docs
```

---

## Data Sources

| Dataset | Source | Records |
|---|---|---|
| Chicago Crime Data 2023 | [Chicago Open Data Portal](https://data.cityofchicago.org/resource/ijzp-q8t2.json) | 260,000+ |

Data is fetched directly via the Socrata API — no manual download required.

---

## Key Design Decisions

**Why XGBoost over deep learning?**  
XGBoost consistently outperforms neural networks on tabular geospatial data at this scale. The spatial and temporal features matter more than sequence modeling.

**Why H3 hexagonal grid?**  
H3 provides uniform cell sizes (unlike census tracts which vary wildly), making spatial aggregation consistent and allowing neighbor-based features through `grid_disk()`.

**Why temporal validation instead of random split?**  
Random splits leak future data into training for time-series problems. Temporal walk-forward validation mirrors real deployment conditions.

**Why spatial lag features?**  
Crime spillover is a real criminological phenomenon — high-crime cells make neighboring cells riskier too. The `neighbour_avg_crimes` feature captures this directly.

---

## Deployment Architecture

```
GitHub Repository
      ↓
Render.com (Backend)          Netlify (Frontend)
FastAPI + Uvicorn              Static HTML/JS
Auto-deploys on push           Auto-deploys on push
Model files from Google Drive  Calls Render API
      ↓                              ↓
https://urban-risk-intelligence.onrender.com
https://urban-risk-intelligence.netlify.app
```

---

## Future Improvements

- Add accident risk model using NHTSA crash data
- Add environmental/AQI risk model using EPA data
- Composite risk score combining all three models
- MySQL database for storing predictions and user sessions
- Time filter on frontend (filter hexagons by hour/day of week)
- Alert system for high-risk zone notifications

---

## Author

**Rajesh Vinjam**  
Real-Time Urban Risk Intelligence System  
Built with Python · XGBoost · FastAPI · H3 Grid · Leaflet.js

---

## License

This project is licensed under the MIT License.
