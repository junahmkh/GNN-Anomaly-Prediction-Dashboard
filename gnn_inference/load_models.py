import os
import torch
from model import anomaly_anticipation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_all_models(base_dir="GNN_models", in_channels=32, out_channels=16):
    models = {}
    for fw in [4,6,12,24,32,64,96,192,288]:
        for i in range(3):
            model_path = os.path.join(base_dir, f"FW_{fw}/{i}_{fw}.pth")
            if os.path.exists(model_path):
                model = anomaly_anticipation(in_channels, out_channels)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
                models[model_path] = model
    return models, device