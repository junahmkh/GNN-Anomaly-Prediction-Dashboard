---

# GNN Anomaly Prediction Dashboard

A dashboard application for anomaly detection using Graph Neural Networks (GNNs).
This project is containerized with **Docker Compose**, making it easy to run locally or deploy.

This application provides anomaly prediction for the **Marconi100 supercomputer** using the **GRAAFE models** proposed by *Molan et al.*  
👉 Full article available at: [GRAAFE](https://www.sciencedirect.com/science/article/abs/pii/S0167739X24003327)

---

## 📊 Model Overview
- The models are **rack-wise Graph Neural Networks (GNNs)** trained on telemetry data from the Marconi100 data center at **Cineca**.  
- **Marconi100** consists of **49 racks** with a total of **980 compute nodes**.  
- Each rack has **9 trained model variants**, corresponding to **future windows**:  [4, 6, 12, 24, 32, 64, 96, 192, 288]

These range from **1 hour ahead** up to **72 hours ahead**, with each step = 15 minutes.  
- Each model outputs an **anomaly probability per compute node**, which is then classified as anomalous or not based on thresholds tuned in experiments.  
- As expected, prediction accuracy decreases for longer forecast windows.  
---
## 🏗️ Application Architecture
The system is composed of **three services**, deployed with Docker:

### 1. Backend Service (FastAPI)  
- Acts as the orchestrator for all services.  
- Schedules inference runs every **15 minutes** across all GNN models and forecast windows.  

### 2. GNN Inference Service  
- Hosts the trained GNN models.  
- Executes anomaly predictions at **15-minute intervals** for each rack and future window.  

### 3. Frontend Service  
Provides a **web interface** with two pages:  

#### 🔹 Overview Page
- Displays a **heatmap** (`racks × future windows`).  
- Cell color intensity represents the **count of anomalies**:  
- **Green** → few anomalies  
- **Red** → many anomalies  
- Enables system-wide anomaly inspection.  

#### 🔹 Dashboard Page
- Provides a **rack-level deep dive**.  
- Heatmap shows **nodes × future windows** for a selected rack.  
- A **time-series plot** displays anomaly probabilities per node with the threshold line.  
- Includes a tab for monitoring **inference runtime performance**.  

---

## ✨ Key Features
- **Rack-wise anomaly prediction** with GNN models.  
- **Multi-horizon forecasting** (1h–72h ahead).  
- **Automated inference scheduling** every 15 minutes.  
- **Interactive visualization** at both rack and node levels.  
- **Inference runtime monitoring** for performance tracking.  

---

## 🚀 Features

* **GNN-based anomaly prediction** with pre-trained models
* **Interactive dashboard** to visualize anomalies
* **Modular architecture** with separate services for:

  * `frontend` (UI)
  * `backend` (API)
  * `gnn_inference` (model inference service)
  * `storage` and `data` support

---

## 📦 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/junahmkh/GNN-Anomaly-Prediction-Dashboard.git
cd GNN-Anomaly-Prediction-Dashboard
```

### 2. Install Requirements

Make sure you have the following installed:

* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/)

### 3. Start the Application

```bash
docker-compose up --build
```

This will:

* Build images for the backend, frontend, and inference service
* Start all containers defined in `docker-compose.yml`

### 4. Access the Dashboard

Once services are up, open your browser at:

```
http://localhost:<port>
```

> 🔍 Check the `docker-compose.yml` file for the exact host port mapping of each service (e.g., frontend) and update accordingly to avoid any conflict with existing services.

---

## 📊 Dataset

The dashboard uses the **M100 dataset**.
You can download it from [Zenodo](https://zenodo.org/records/7541722).
Place the dataset under the `data/m100_aggregated/` directory before running the app.

---

## 🛠️ Useful Commands

| Action           | Command                                                    |
| ---------------- | ---------------------------------------------------------- |
| Start containers | `docker-compose up --build`                                |
| Stop containers  | `docker-compose down`                                      |
| View logs        | `docker-compose logs -f`                                   |
| Rebuild services | `docker-compose up --build --force-recreate`               |
| Full clean-up    | `docker-compose down --rmi all --volumes --remove-orphans` |

---

## 📂 Project Structure

```
GNN-Anomaly-Prediction-Dashboard/
│── backend/          # API service
│── frontend/         # Dashboard UI
│── gnn_inference/    # GNN model inference service
│── data/             # Dataset (M100)
│── storage/          # Persistent storage
│── docker-compose.yml
│── README.md
```

---

## 📜 License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for details.

---
