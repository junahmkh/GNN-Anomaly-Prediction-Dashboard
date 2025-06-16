import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import requests

# ------------------ Constants ------------------
fw_ids = [f"FW_{i}" for i in [4,6,12,24,32,64,96,192,288]]
node_ids = [f"node_{i:02d}" for i in range(1, 21)]
threshold = 0.1
num_points = 30  # how many time points to plot
backend_url = "http://backend:8001"  # change to your backend URL

# ------------------ Fetch data from backend ------------------
def fetch_predictions_for_rack(rack: int):
    try:
        response = requests.get(f"{backend_url}/results/{rack}", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data["predictions"]
    except Exception as e:
        st.warning(f"Could not fetch predictions for rack {rack}: {e}")
        return []

# ------------------ Mock Data Generator ------------------
def generate_mock_rack_data(num_points=30):
    from random import uniform
    racks = []
    now = datetime.utcnow()
    for i in range(3, 50):  # start from rack 3, since 0,1,2 are actual data
        rack_id = f"{i:02d}"
        history = {}
        for fw in [4,6,12,24,32,64,96,192,288]:
            fw_id = f"FW_{fw}"
            history[fw_id] = {
                f"node_{n:02d}": [
                    {
                        "timestamp": (now - timedelta(minutes=5 * j)).isoformat(),
                        "score": round(uniform(0, 1), 2)
                    }
                    for j in range(num_points)
                ]
                for n in range(1, 21)
            }
        racks.append({"id": rack_id, "history": history})
    return racks

# ------------------ Build rack data with actual + mock ------------------
rack_data = []

# Add real data racks 0,1,2
for rack_id in [0,1,2]:
    predictions = fetch_predictions_for_rack(rack_id)
    # Convert predictions list (dicts with timestamp, fw, prediction) to the "history" structure expected below
    # We'll build a dict: {fw_id: {node_id: list of {timestamp, score}}}
    history = {}
    for fw in [4,6,12,24,32,64,96,192,288]:
        fw_id = f"FW_{fw}"
        # For each fw, collect predictions for this rack and fw
        relevant_preds = [p for p in predictions if p["fw"] == fw]
        # We don't have node-wise split in your current model output, so we'll simulate node scores equally:
        # Use the prediction value for all nodes, or simulate variation if desired.
        node_scores = {}
        for node_idx in range(1, 21):
            node_id = f"node_{node_idx:02d}"
            # Each prediction corresponds to a timestamp, but we may have multiple timestamps.
            # So collect scores per timestamp:
            scores = []
            for p in relevant_preds[-num_points:]:  # last N predictions
                # Assuming p["prediction"] is a list of floats per node? If not, use average or single value:
                if isinstance(p["prediction"], list):
                    score_val = p["prediction"][node_idx % len(p["prediction"])]
                else:
                    score_val = float(p["prediction"])  # fallback to single value
                scores.append({"timestamp": p["timestamp"], "score": score_val})
            node_scores[node_id] = scores
        history[fw_id] = node_scores

    rack_data.append({"id": str(rack_id), "history": history})

# Add mock racks 3..49
rack_data.extend(generate_mock_rack_data(num_points=num_points))

rack_ids = [rack["id"] for rack in rack_data]

# ------------------ Calculate anomaly counts ------------------
anomaly_counts = np.zeros((len(rack_ids), len(fw_ids)), dtype=int)

for rack_idx, rack in enumerate(rack_data):
    for fw_idx, fw_id in enumerate(fw_ids):
        count = 0
        for node_id in node_ids:
            latest_score = rack["history"][fw_id][node_id][0]["score"]
            if latest_score < threshold:
                count += 1
        anomaly_counts[rack_idx, fw_idx] = count

# ------------------ Plot ------------------
fig = go.Figure(data=go.Heatmap(
    z=anomaly_counts,
    x=fw_ids,
    y=rack_ids,
    colorscale='Reds',
    colorbar=dict(title='Anomaly Count'),
    hovertemplate='Rack: %{y}<br>FW: %{x}<br>Anomalies: %{z}<extra></extra>'
))
fig.update_layout(
    height=700,
    yaxis_autorange='reversed',
    title="Anomaly Counts Heatmap (Racks vs Forecast Windows)",
    xaxis_title="Forecast Window",
    yaxis_title="Rack"
)

st.plotly_chart(fig, use_container_width=True)
