import streamlit as st
import plotly.graph_objects as go
import numpy as np
import requests
import time

# ------------------ Constants ------------------
fw_values = [4, 6, 12, 24, 32, 64, 96, 192, 288]
fw_ids = [f"FW_{fw}" for fw in fw_values]
rack_ids = ["0", "2", "8"]
#rack_ids = [0, 2, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]

#threshold = 0.1
thresholds_fw = {
    "FW_4": 0.133077,
    "FW_6": 0.111795,
    "FW_12": 0.078974,
    "FW_24": 0.060513,
    "FW_32": 0.067949,
    "FW_64": 0.061026,
    "FW_96": 0.068462,
    "FW_192": 0.068205,
    "FW_288": 0.074615,
} 

backend_url = "http://backend:8001"

# ------------------ Auto Refresh ------------------
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=5000, key="auto-refresh")

st.set_page_config(layout="wide", page_title="GNN Inference: Anomaly Prediction on M100 Data")

st.title("System Status: Overview")

# ------------------ Fetch + Wait for Data ------------------
def fetch_predictions_for_rack(rack_id):
    try:
        response = requests.get(f"{backend_url}/results/{rack_id}", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f"Failed to fetch rack {rack_id}: {e}")
        return []

def fetch_predictions_for_rack_with_wait(rack_id, max_wait=60, interval=2):
    """Try fetching predictions repeatedly until non-empty or max_wait seconds elapsed."""
    start = time.time()
    while True:
        data = fetch_predictions_for_rack(rack_id)
        if data and data != []:
            # Also check if 'predictions' key exists and is non-empty list if data is dict
            if isinstance(data, dict) and data.get("predictions"):
                return data
            # Or if data is already a list with items
            if isinstance(data, list) and len(data) > 0:
                return data
        if time.time() - start > max_wait:
            return data
        time.sleep(interval)

# ------------------ Build Anomaly Matrix ------------------
anomaly_counts = np.zeros((len(rack_ids), len(fw_ids)), dtype=int)
all_latest_timestamps = []

for rack_idx, rack_id in enumerate(rack_ids):
    with st.spinner(f"Fetching data for Rack {rack_id}..."):
        predictions = fetch_predictions_for_rack_with_wait(rack_id)

    latest_by_fw = {}

    if isinstance(predictions, list):
        predictions_dict = {"predictions": predictions}
    else:
        predictions_dict = predictions

    for pred in predictions_dict.get("predictions", []):
        fw = pred["fw"]
        ts = pred["timestamp"]
        if fw not in latest_by_fw or ts > latest_by_fw[fw]["timestamp"]:
            latest_by_fw[fw] = pred

    if latest_by_fw:
        rack_latest = max(pred["timestamp"] for pred in latest_by_fw.values())
        all_latest_timestamps.append(rack_latest)
    else:
        all_latest_timestamps.append(None)

    for fw in fw_values:
        fw_key = f"FW_{fw}"
        threshold = thresholds_fw.get(fw_key) 
        if fw in latest_by_fw:
            pred_obj = latest_by_fw[fw]["prediction"]
            if isinstance(pred_obj, dict) and "prediction" in pred_obj:
                preds = pred_obj["prediction"]
                for score in preds:
                    if score > threshold:
                        anomaly_counts[rack_idx, fw_ids.index(fw_key)] += 1

# Compute overall latest timestamp
valid_timestamps = [ts for ts in all_latest_timestamps if ts is not None]
if valid_timestamps:
    overall_latest = max(valid_timestamps)
    st.write(f"#### Current Timestamp: {overall_latest}")
else:
    st.write("#### No predictions available yet.")

# ------------------ Plot Heatmap ------------------
fig = go.Figure(data=go.Heatmap(
    z=anomaly_counts,
    x=fw_ids,
    y=rack_ids,
    colorscale=[
        [0.0, 'green'],   # 0 anomalies
        [0.5, 'yellow'],  # medium
        [1.0, 'red']      # max anomalies
    ],
    colorbar=dict(title='Count'),
    hovertemplate='Rack: %{y}<br>FW: %{x}<br>Anomalies: %{z}<extra></extra>'
))

fig.update_layout(
    height=600,
    title="Anomaly Heatmap",
    xaxis_title="Future Window",
    yaxis_title="Rack",
    yaxis=dict(
        tickmode='array',
        tickvals=rack_ids,
        ticktext=rack_ids
    )
)

st.plotly_chart(fig, use_container_width=True)
