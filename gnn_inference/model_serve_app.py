from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from torch_geometric.data import Data
from load_models import load_all_models

models, device = load_all_models(in_channels=417, out_channels=16)
app = FastAPI()

class GraphInput(BaseModel):
    x: list[list[float]]
    edge_index: list[list[int]]

@app.post("/predict/{fw}/{model_id}")
def predict(fw: int, model_id: int, graph_input: GraphInput):
    key = f"{fw}/rack_{model_id}"
    model = models.get(key)

    if not model:
        raise HTTPException(status_code=404, detail=f"Model {key} not found")

    try:
        # Move data to the right device
        x = torch.tensor(graph_input.x, dtype=torch.float).to(device)
        edge_index = torch.tensor(graph_input.edge_index, dtype=torch.long).to(device)
        data = Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = torch.sigmoid(out)

        return {"prediction": pred.squeeze().tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
