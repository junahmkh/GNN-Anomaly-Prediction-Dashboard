---

# GNN Anomaly Prediction Dashboard

A dashboard application for anomaly detection using Graph Neural Networks (GNNs).
This project is containerized with **Docker Compose**, making it easy to run locally or deploy.

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

> 🔍 Check the `docker-compose.yml` file for the exact frontend port mapping.

---

## 📊 Dataset

The dashboard uses the **M100 dataset**.
You can download it from [Zenodo]([https://zenodo.org/](https://zenodo.org/records/7541722)) (link provided in the repo).
Place the dataset under the `data/` directory before running the app.

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

👉 Do you want me to also include **example screenshots of the dashboard** (with placeholders) in the README so it looks more polished?
