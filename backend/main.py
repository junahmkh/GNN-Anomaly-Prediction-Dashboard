import os
import pickle
import logging
import requests
from fastapi import FastAPI, HTTPException
import time
from apscheduler.schedulers.background import BackgroundScheduler

from data_fetch import data_fetch
from data_preprocessing import pre_process

app = FastAPI()

rack_ids = [0, 2, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
log_dir = "/app/logs"
data_dir = "/app/storage"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "prediction_scheduler.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load timestamps list
with open("common_ts.pickle", "rb") as f:
    timestamps = pickle.load(f)

timestamps = sorted(timestamps)

index = 0  # pointer to next timestamp

# File path to store latest predictions persistently
PREDICTIONS_PATH = os.path.join(data_dir, "latest_predictions.pickle")

# Try to load previous predictions from disk
if os.path.exists(PREDICTIONS_PATH):
    with open(PREDICTIONS_PATH, "rb") as f:
        latest_predictions = pickle.load(f)
    logger.info(f"Loaded existing latest_predictions from {PREDICTIONS_PATH}")
else:
    latest_predictions = {}

def save_predictions():
    try:
        with open(PREDICTIONS_PATH, "wb") as f:
            pickle.dump(latest_predictions, f)
        logger.info("✅ Saved latest_predictions to disk.")
    except Exception as e:
        logger.error(f"Failed to save latest_predictions: {e}")

# This dict can be exposed via an API or returned somehow
latest_timings = {}  # key: f"{ts}|{fw}|{rack}", value: dict of timings

def run_scheduled_prediction():
    global index
    if index >= len(timestamps):
        logger.info("✅ All timestamps processed, resetting index to 0")
        index = 0

    ts = timestamps[index]
    logger.info(f"Processing telemetry data for timestamp: {ts}")

    for rack in rack_ids:
        try:
            # --- Data Fetch Timing ---
            start_fetch = time.perf_counter()
            fetched_df = data_fetch(rack, ts)
            fetch_time = (time.perf_counter() - start_fetch) * 1000  # ms

            # --- Preprocessing Timing ---
            start_preprocess = time.perf_counter()
            graph_payload = pre_process(fetched_df)
            preprocess_time = (time.perf_counter() - start_preprocess) * 1000  # ms

            for fw in [4, 6, 12, 24, 32, 64, 96, 192, 288]:
                try:
                    # --- Inference Timing ---
                    start_inference = time.perf_counter()
                    url = f"http://gnn_inference:10000/predict/{fw}/{rack}"
                    response = requests.post(url, json=graph_payload, timeout=10)
                    response.raise_for_status()
                    inference_time = (time.perf_counter() - start_inference) * 1000  # ms

                    # Store prediction and timings
                    prediction = response.json()
                    key = f"{ts}|{fw}|{rack}"
                    latest_predictions[key] = prediction

                    latest_timings[key] = {
                        "FW": fw,
                        "Data Fetch (ms)": round(fetch_time),
                        "Preprocessing (ms)": round(preprocess_time),
                        "Inference (ms)": round(inference_time)
                    }

                    logger.info(f"Prediction successful for ts={ts}, fw={fw}, rack={rack}")

                except Exception as e:
                    logger.error(f"Inference error for ts={ts}, fw={fw}, rack={rack}: {e}")

        except Exception as e:
            logger.error(f"Error during prediction for ts={ts}, rack={rack}: {e}")

    save_predictions()
    index += 1

@app.on_event("startup")
def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_scheduled_prediction, "interval", minutes=1)
    scheduler.start()
    logger.info("Scheduler started — running prediction every 1 minute")


@app.get("/results/{rack}")
def get_latest_predictions(rack: int):
    rack_predictions = []
    for key, pred in latest_predictions.items():
        ts, fw_str, rack_str = key.split("|")
        fw = int(fw_str)
        r = int(rack_str)
        if r == rack:
            rack_predictions.append({"timestamp": ts, "fw": fw, "prediction": pred})

    if not rack_predictions:
        raise HTTPException(status_code=404, detail=f"No predictions found for rack {rack}")

    rack_predictions.sort(key=lambda x: (x["timestamp"], x["fw"]))

    return {"rack": rack, "predictions": rack_predictions}

@app.get("/timings/{rack}/latest")
def get_latest_timings_for_rack(rack: int):
    # Filter timings for the rack
    rack_timings = []
    for key, timing in latest_timings.items():
        ts, fw_str, rack_str = key.split("|")
        if int(rack_str) == rack:
            rack_timings.append((ts, timing, int(fw_str)))

    if not rack_timings:
        raise HTTPException(status_code=404, detail=f"No timing data found for rack {rack}")

    # Find the latest timestamp
    latest_ts = max(ts for ts, _, _ in rack_timings)

    # Return timings only for the latest timestamp
    latest_data = []
    for ts, timing, fw in rack_timings:
        if ts == latest_ts:
            entry = {"FW": fw, **timing, "timestamp": ts}
            latest_data.append(entry)

    # Sort by FW for consistent chart ordering
    latest_data.sort(key=lambda x: x["FW"])

    return {"rack": rack, "timestamp": latest_ts, "timings": latest_data}