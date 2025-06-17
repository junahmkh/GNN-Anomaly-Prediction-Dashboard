import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide", page_title="GNN Inference: Anomaly Prediction on M100 Data")

# ------------------ Constants ------------------
backend_url = "http://backend:8001"
rack_ids = ["0", "2", "8"]
fw_values = [4,6,12,24,32,64,96,192,288]
fw_ids = [f"FW_{fw}" for fw in fw_values]
threshold = 0.1

# ------------------ Auto Refresh ------------------
st_autorefresh(interval=5000, key="auto-refresh")

# ------------------ Helper: Fetch predictions ------------------
@st.cache_data(ttl=30)
def fetch_predictions(rack_id):
    try:
        response = requests.get(f"{backend_url}/results/{rack_id}", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f"Failed to fetch predictions for rack {rack_id}: {e}")
        return []

# ------------------ Helper: Infer node IDs from prediction array ------------------
def extract_node_ids(predictions):
    for entry in predictions:
        prediction_array = entry.get("prediction", {}).get("prediction", [])
        if isinstance(prediction_array, list) and prediction_array:
            return [f"node_{i:02d}" for i in range(len(prediction_array))]
    return []

# ------------------ Prepare parsed prediction history ------------------
def parse_predictions(raw_data, node_ids):
    history = {fw_id: {node_id: [] for node_id in node_ids} for fw_id in fw_ids}

    if not raw_data or not isinstance(raw_data, list):
        return history

    for pred in raw_data:
        fw = pred.get("fw")
        timestamp = pred.get("timestamp")
        fw_id = f"FW_{fw}"
        if fw_id not in history:
            continue
        scores = pred.get("prediction", {}).get("prediction", [])
        if len(scores) != len(node_ids):
            continue

        for i, node_id in enumerate(node_ids):
            history[fw_id][node_id].append({
                "timestamp": timestamp,
                "score": scores[i]
            })

    for fw_data in history.values():
        for series in fw_data.values():
            series.sort(key=lambda x: x["timestamp"], reverse=True)

    return history

# ------------------ Sidebar & Layout ------------------
st.sidebar.title("Dashboard: Rack")
selected_rack_id = st.sidebar.selectbox("Select Rack", rack_ids)
st.title(f"Rack: {selected_rack_id} - GNN Anomaly & Timing Dashboard")

tab1, tab2 = st.tabs(["🔍 Anomaly Visualization", "⏱️ GNN Timing Analysis"])

# ------------------ Fetch & Prepare Data ------------------
raw_predictions = fetch_predictions(selected_rack_id)
prediction_entries = raw_predictions.get("predictions", [])
node_ids = extract_node_ids(prediction_entries)
history = parse_predictions(prediction_entries, node_ids)

# ------------------ Tab 1: Anomaly Visualization ------------------
with tab1:
    st.subheader("Latest Prediction Scores Heatmap")

    latest_scores = np.zeros((len(node_ids), len(fw_ids)), dtype=float)
    latest_anomaly = np.zeros((len(node_ids), len(fw_ids)), dtype=int)

    all_timestamps = []
    for fw_id in fw_ids:
        for node_id in node_ids:
            if history[fw_id][node_id]:
                all_timestamps.append(history[fw_id][node_id][0]["timestamp"])

    if all_timestamps:
        latest_overall_timestamp = max(all_timestamps)
        latest_overall_datetime = datetime.fromisoformat(latest_overall_timestamp)
        st.markdown(f"Latest Timestamp Across Rack {selected_rack_id}: **{latest_overall_datetime} UTC**")
    else:
        st.warning("No prediction data available yet.")

    for fw_idx, fw_id in enumerate(fw_ids):
        for node_idx, node_id in enumerate(node_ids):
            time_series = history[fw_id][node_id]
            if time_series:
                score = time_series[0]["score"]
                latest_scores[node_idx, fw_idx] = score
                # Flipped anomaly logic: anomaly if score > threshold
                latest_anomaly[node_idx, fw_idx] = int(score > threshold)
            else:
                latest_scores[node_idx, fw_idx] = np.nan

    fig = go.Figure(data=go.Heatmap(
        z=latest_scores,
        x=fw_ids,
        y=node_ids,
        colorscale='RdYlGn_r',
        colorbar=dict(title='Latest Score (Lower = More Anomalous)'),
        hovertemplate='Node: %{y}<br>FW: %{x}<br>Score: %{z}<extra></extra>'
    ))
    fig.update_layout(
        height=600,
        yaxis_autorange='reversed',
        title="Latest Prediction Scores Heatmap (Nodes vs Forecast Windows)",
        xaxis_title="Forecast Window",
        yaxis_title="Node"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Anomaly threshold:** {threshold} (Scores above are anomalous)")

    # Time Series View
    st.subheader("View Detailed Time Series for a Node and Forecast Window")
    selected_fw = st.selectbox("Forecast Window", fw_ids)
    selected_node = st.selectbox("Node", node_ids)

    df = pd.DataFrame(history[selected_fw][selected_node])
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Flipped anomaly logic here as well
        df['anomaly'] = df['score'] > threshold

        fig2 = px.line(df, x='timestamp', y='score', title=f"{selected_fw} - {selected_node} Time Series",
                       markers=True, labels={'score': 'Model Score'})
        fig2.update_layout(height=400, yaxis=dict(range=[0, 1]))
        fig2.add_trace(go.Scatter(
            x=df.loc[df['anomaly'], 'timestamp'],
            y=df.loc[df['anomaly'], 'score'],
            mode='markers',
            marker=dict(color='red', size=8),
            name='Anomalies'
        ))
        fig2.add_hline(y=threshold, line_dash="dash", line_color="red",
                       annotation_text="Anomaly Threshold", annotation_position="bottom right")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info(f"No time series data for {selected_fw} / {selected_node}.")

# ------------------ Tab 2: GNN Timing Analysis ------------------
with tab2:
    st.subheader(f"GNN Execution Times per Step for Rack {selected_rack_id}")

    # Dummy/mock timing data — replace with backend timings if available
    timing_data = []
    for fw in fw_ids:
        timing_data.append({
            "FW": fw,
            "Data Fetch (ms)": np.random.randint(50, 150),
            "Preprocessing (ms)": np.random.randint(100, 250),
            "Inference (ms)": np.random.randint(80, 200)
        })
    df_timing = pd.DataFrame(timing_data)
    df_melted = df_timing.melt(id_vars=["FW"], var_name="Step", value_name="Time (ms)")

    fig_timing = px.bar(df_melted, x="FW", y="Time (ms)", color="Step", barmode="group",
                        title="Execution Time by Step (per Forecast Window)")
    fig_timing.update_layout(height=500)
    st.plotly_chart(fig_timing, use_container_width=True)

    # Hardware metrics mock
    st.subheader("Hardware Metrics During Inference")
    hw_data = pd.DataFrame({
        "FW": fw_ids,
        "GPU Power (W)": np.random.uniform(60, 120, len(fw_ids)).round(1),
        "GPU Utilization (%)": np.random.uniform(20, 95, len(fw_ids)).round(1),
        "CPU Usage (%)": np.random.uniform(15, 85, len(fw_ids)).round(1),
        "RAM Usage (MB)": np.random.uniform(2000, 8000, len(fw_ids)).round(0),
        "GPU Temperature (°C)": np.random.uniform(40, 85, len(fw_ids)).round(1),
    })

    def make_bar_plot(y_column, title):
        fig = px.bar(hw_data, x="FW", y=y_column, title=title)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    make_bar_plot("GPU Power (W)", "GPU Power Consumption")
    make_bar_plot("GPU Utilization (%)", "GPU Utilization")
    make_bar_plot("CPU Usage (%)", "CPU Usage")
    make_bar_plot("RAM Usage (MB)", "RAM Usage")
    make_bar_plot("GPU Temperature (°C)", "GPU Temperature")
