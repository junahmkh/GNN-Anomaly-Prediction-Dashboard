import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
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

# ------------------ UI ------------------
st.set_page_config(layout="wide", page_title="GNN Inference: Anomaly Prediction on M100 data")
st.sidebar.title("Dashboard: Rack")
selected_rack_id = st.sidebar.selectbox("Select Rack", rack_ids)
selected_rack = next(r for r in rack_data if r["id"] == selected_rack_id)

st.title(f"Rack: {selected_rack_id} - GNN Anomaly & Timing Dashboard")

tab1, tab2 = st.tabs(["🔍 Anomaly Visualization", "⏱️ GNN Timing Analysis"])

# ------------------ Tab 1: Anomaly Visualization ------------------
with tab1:
    st.subheader("Latest Prediction Scores Heatmap")

    # Create matrix for latest scores and binary anomaly flags
    latest_scores = np.zeros((len(node_ids), len(fw_ids)), dtype=float)
    latest_anomaly = np.zeros((len(node_ids), len(fw_ids)), dtype=int)

    for fw_idx, fw_id in enumerate(fw_ids):
        for node_idx, node_id in enumerate(node_ids):
            time_series = selected_rack["history"][fw_id][node_id]
            latest_point = time_series[0]
            score = latest_point["score"]
            latest_scores[node_idx, fw_idx] = score
            latest_anomaly[node_idx, fw_idx] = int(score < threshold)

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
    st.markdown(f"**Anomaly threshold:** {threshold} (Scores below are anomalous)")

    # Time series viewer
    st.subheader("View detailed time series for a node and forecast window")
    selected_fw = st.selectbox("Forecast Window", fw_ids)
    selected_node = st.selectbox("Node", node_ids)

    df = pd.DataFrame(selected_rack["history"][selected_fw][selected_node])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['anomaly'] = df['score'] < threshold

    fig2 = px.line(df, x='timestamp', y='score', title=f"{selected_fw} - {selected_node} Time Series",
                   markers=True, labels={'score': 'Model Score'})
    fig2.update_layout(height=400, yaxis=dict(range=[0, 0.3]))

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
    
# ------------------ Tab 2: GNN Timing Analysis ------------------
with tab2:
    st.subheader(f"GNN Execution Times per Step for Rack {selected_rack_id}")

    # ------------------ Timing Mock Data ------------------
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

    # ------------------ Hardware Metrics Mock Data ------------------
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
