import os
import torch
from model import anomaly_anticipation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rack_ids = [0, 2, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]

def load_all_models(base_dir = "/app/GNN_models"):
    models = {}
    for fw in [4,6,12,24,32,64,96,192,288]:
        for i in rack_ids:
            model_path = os.path.join(base_dir, f"FW_{fw}/{i}_{fw}.pth")
            if os.path.exists(model_path):
                model = anomaly_anticipation(417, 16)
                model.load_state_dict(torch.load(model_path,map_location=device))
                model.to(device)
                model.eval()
                models[f"{fw}/rack_{i}"] = model
    return models, device