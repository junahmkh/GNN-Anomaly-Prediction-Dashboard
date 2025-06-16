import os
import pickle
import logging
import requests
from fastapi import FastAPI, HTTPException
from apscheduler.schedulers.background import BackgroundScheduler

from .data_fetch import data_fetch
from .data_preprocessing import pre_process

app = FastAPI()

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

def run_scheduled_prediction():
    global index
    if index >= len(timestamps):
        logger.info("✅ All timestamps processed, resetting index to 0")
        index = 0

    ts = timestamps[index]
    logger.info(f"Processing telemetry data for timestamp: {ts}")

    for rack in [0, 1, 2]:
        try:
            fetched_df = data_fetch(rack, ts)
            graph_payload = pre_process(fetched_df)

            for fw in [4, 6, 12, 24, 32, 64, 96, 192, 288]:
                url = f"http://gnn_inference:10000/predict/{fw}/{rack}"
                response = requests.post(url, json=graph_payload, timeout=10)
                response.raise_for_status()

                prediction = response.json()
                # Use a string key for pickling-friendly storage
                key = f"{ts}|{fw}|{rack}"
                latest_predictions[key] = prediction

                logger.info(f"Prediction successful for ts={ts}, fw={fw}, rack={rack}")

        except Exception as e:
            logger.error(f"Error during prediction for ts={ts}, rack={rack}: {e}")

    # Save after every run to disk
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
