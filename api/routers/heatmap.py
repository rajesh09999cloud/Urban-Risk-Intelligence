from fastapi import APIRouter, Request
import pandas as pd
import h3

router = APIRouter()

@router.get("/heatmap")
async def get_heatmap(request: Request, risk_type: str = "composite"):

    cell_df = request.app.state.cell_features.copy()

    # Normalize risk score to 0-1
    max_val = cell_df['total_crimes_in_cell'].max()
    cell_df['risk_score'] = (
        cell_df['cell_violent_rate'] * 0.6 +
        (cell_df['total_crimes_in_cell'] / max_val) * 0.4
    )
    cell_df['risk_score'] = (
        cell_df['risk_score'] / cell_df['risk_score'].max()
    ).round(4)

    features = []
    for _, row in cell_df.iterrows():
        try:
            boundary = h3.cell_to_boundary(row['h3_cell'])
            # GeoJSON needs [lon, lat] order
            coords = [[lon, lat] for lat, lon in boundary]
            coords.append(coords[0])  # close the polygon

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                },
                "properties": {
                    "cell_id":       row['h3_cell'],
                    "risk_score":    float(row['risk_score']),
                    "total_crimes":  int(row['total_crimes_in_cell']),
                    "violent_rate":  round(float(row['cell_violent_rate']), 4),
                    "risk_level":    "HIGH" if row['risk_score'] > 0.6
                                     else "MEDIUM" if row['risk_score'] > 0.35
                                     else "LOW"
                }
            })
        except Exception:
            continue

    return {
        "type": "FeatureCollection",
        "features": features,
        "total_cells": len(features)
    }