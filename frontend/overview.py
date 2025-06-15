import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from random import uniform

# ------------------ Mock Data Generator ------------------
def generate_mock_rack_data(num_points=30):
    racks = []
    now = datetime.utcnow()
    for i in range(1, 50):
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

# ------------------ Setup ------------------
rack_data = generate_mock_rack_data()
rack_ids = [rack["id"] for rack in rack_data]
fw_ids = [f"FW_{i}" for i in [4,6,12,24,32,64,96,192,288]]
node_ids = [f"node_{i:02d}" for i in range(1, 21)]
threshold = 0.1

st.set_page_config(layout="wide", page_title="GNN Overview: Anomaly Summary")

st.title("Overview: All Racks Anomaly Counts")

# Calculate anomaly counts per rack and forecast window
anomaly_counts = np.zeros((len(rack_ids), len(fw_ids)), dtype=int)

for rack_idx, rack in enumerate(rack_data):
    for fw_idx, fw_id in enumerate(fw_ids):
        count = 0
        for node_id in node_ids:
            latest_score = rack["history"][fw_id][node_id][0]["score"]
            if latest_score < threshold:
                count += 1
        anomaly_counts[rack_idx, fw_idx] = count

# Heatmap for anomaly counts (racks vs forecast windows)
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
