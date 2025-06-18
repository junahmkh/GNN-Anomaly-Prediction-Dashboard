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
#rack_ids = [0, 2, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]

fw_values = [4,6,12,24,32,64,96,192,288]
fw_ids = [f"FW_{fw}" for fw in fw_values]

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
st.title(f"Dashboard : Rack {selected_rack_id}")

tab1, tab2 = st.tabs(["🔍 Anomaly Visualization", "⏱️ Timing Analysis"])

# ------------------ Fetch & Prepare Data ------------------
raw_predictions = fetch_predictions(selected_rack_id)
prediction_entries = raw_predictions.get("predictions", [])
node_ids = extract_node_ids(prediction_entries)
history = parse_predictions(prediction_entries, node_ids)

# ------------------ Tab 1: Anomaly Visualization ------------------
with tab1:
    st.subheader("Anomaly Heatmap")

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
        st.markdown(f"Latest Timestamp **{latest_overall_datetime} UTC**")
    else:
        st.warning("No prediction data available yet.")

    for fw_idx, fw_id in enumerate(fw_ids):
        for node_idx, node_id in enumerate(node_ids):
            time_series = history[fw_id][node_id]
            if time_series:
                score = time_series[0]["score"]
                latest_scores[node_idx, fw_idx] = score
                threshold = thresholds_fw[fw_id]
                latest_anomaly[node_idx, fw_idx] = int(score > threshold)
                #latest_anomaly[node_idx, fw_idx] = int(score > threshold)
            else:
                latest_scores[node_idx, fw_idx] = np.nan

    # fig = go.Figure(data=go.Heatmap(
    #     z=latest_anomaly, #check if to change back to later_scores
    #     x=fw_ids,
    #     y=node_ids,
    #     colorscale='RdYlGn_r',
    #     colorbar=dict(title='Latest Score (Lower = More Anomalous)'),
    #     hovertemplate='Node: %{y}<br>FW: %{x}<br>Score: %{z}<extra></extra>'
    # ))
    # fig.update_layout(
    #     height=600,
    #     yaxis_autorange='reversed',
    #     title="Latest Prediction Scores Heatmap (Nodes vs Future Windows)",
    #     xaxis_title="Future Window",
    #     yaxis_title="Node"
    # )

    fig = go.Figure(data=go.Heatmap(
        z=latest_anomaly,
        x=fw_ids,
        y=node_ids,
        colorscale=[[0.0, 'green'], [1.0, 'red']],
        zmin=0, zmax=1,
        showscale=False,
        hovertemplate='Node: %{y}<br>FW: %{x}<br>Score: %{z}<extra></extra>'
    ))

    # Custom legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Anomalous (1)'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='green'),
        name='Normal (0)'
    ))

    fig.update_layout(
        height=600,
        yaxis_autorange='reversed',
        #title="Binary Anomaly Heatmap (Nodes vs Future Windows)",
        xaxis_title="Future Window",
        yaxis_title="Node",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    # st.markdown(f"**Anomaly threshold:** {threshold} (Scores above are anomalous)")

    # df_thresh = pd.DataFrame({
    #     "Future Window": fw_ids,
    #     "Anomaly Threshold": [thresholds_fw[fw] for fw in fw_ids]
    # })

    # df_thresh_reset = df_thresh.reset_index(drop=True)

    # styled_df = df_thresh_reset.style.format({"Anomaly Threshold": "{:.6f}"}) \
    #     .set_properties(subset=["Future Window"], **{'text-align': 'left'}) \
    #     .set_properties(subset=["Anomaly Threshold"], **{'text-align': 'right'})

    # st.table(styled_df)

    # Time Series View
    st.subheader("View Detailed Time Series for a Node and Future Window")
    selected_fw = st.selectbox("Future Window", fw_ids)
    selected_node = st.selectbox("Node", node_ids)

    df = pd.DataFrame(history[selected_fw][selected_node])
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        threshold = thresholds_fw[selected_fw]
        df['anomaly'] = df['score'] > threshold

        fig2 = px.line(df, x='timestamp', y='score', title=f"Time-series - FW: {selected_fw} | Node: {selected_node}",
                       markers=True, labels={'score': 'Anomaly Score'})
        fig2.update_layout(height=400, yaxis=dict(range=[0, 1]))
        fig2.add_trace(go.Scatter(
            x=df.loc[df['anomaly'], 'timestamp'],
            y=df.loc[df['anomaly'], 'score'],
            mode='markers',
            marker=dict(color='red', size=8),
            name='Anomalies'
        ))
        fig2.add_hline(y=threshold, line_dash="dash", line_color="red",
               annotation_text=f"Threshold: {threshold:.4f}", annotation_position="bottom right")
        # fig2.add_hline(y=threshold, line_dash="dash", line_color="red",
        #                annotation_text="Anomaly Threshold", annotation_position="bottom right")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info(f"No time series data for {selected_fw} / {selected_node}.")

# ------------------ Tab 2: GNN Timing Analysis ------------------
with tab2:
    try:
         # Hardware metrics mock
        st.subheader("Time Metrics During Inference - Per Future Window")

        url = backend_url + f"/timings/{selected_rack_id}/latest"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        timing_data = data["timings"]

        if not timing_data:
            st.warning("No timing data available for the selected rack.")
        else:
            # Prepare DataFrame
            df_timing = pd.DataFrame(timing_data)
            df_timing["FW"] = df_timing["FW"].astype(int)
            df_timing = df_timing.sort_values("FW")

            # Create FW labels like "FW_4"
            df_timing["FW"] = df_timing["FW"].map(lambda fw: f"FW_{fw}")
            fw_ids = df_timing["FW"].tolist()

            df_timing = df_timing.drop(columns=["timestamp"])

            # Melt for long format
            df_melted = df_timing.melt(id_vars=["FW"], var_name="Step", value_name="Time (ms)")

            # Show each step as a separate chart
            for step in df_melted["Step"].unique():
                step_df = df_melted[df_melted["Step"] == step]
                fig = px.bar(
                    step_df,
                    x="FW",
                    y="Time (ms)",
                    title=f"Time: {step}",
                    category_orders={"FW": fw_ids}
                )
                fig.update_layout(
                    height=400,
                    xaxis_title="Future Window",
                    yaxis_title="Execution Time [ms]",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

    except requests.RequestException as e:
        st.error(f"Failed to fetch timing data: {e}")

    # st.subheader("Hardware Metrics During Inference")

    # try:
    #     # Fetch resource usage
    #     url_resources = backend_url + f"/resources/{selected_rack_id}/latest"
    #     response_resources = requests.get(url_resources)
    #     response_resources.raise_for_status()
    #     data_resources = response_resources.json()
    #     resource_data = data_resources["resources"]

    #     if not resource_data:
    #         st.warning("No resource usage data available for the selected rack.")
    #     else:
    #         fw_map = dict(zip(fw_values, fw_ids))

    #         df_resources = pd.DataFrame(resource_data)
    #         df_resources["FW"] = df_resources["FW"].astype(int)
    #         df_resources["FW_label"] = df_resources["FW"].map(fw_map)
    #         df_resources = df_resources.sort_values("FW")

    #         # CPU Usage plot
    #         fig_cpu = px.bar(
    #             df_resources,
    #             x="FW_label",
    #             y="CPU (%)",
    #             title=f"CPU Usage per Forecast Window for Rack {selected_rack_id} at {data_resources['timestamp']}"
    #         )
    #         fig_cpu.update_layout(height=400, xaxis_title="Forecast Window (FW)", yaxis_title="CPU Usage (%)")
    #         st.plotly_chart(fig_cpu, use_container_width=True)

    #         # RAM Usage plot
    #         fig_ram = px.bar(
    #             df_resources,
    #             x="FW_label",
    #             y="RAM (MB)",
    #             title=f"RAM Usage per Forecast Window for Rack {selected_rack_id} at {data_resources['timestamp']}"
    #         )
    #         fig_ram.update_layout(height=400, xaxis_title="Forecast Window (FW)", yaxis_title="RAM Usage (MB)")
    #         st.plotly_chart(fig_ram, use_container_width=True)

    # except requests.RequestException as e:
    #     st.error(f"Failed to fetch resource usage data: {e}")