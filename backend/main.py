import requests

response = requests.post("http://inference:9000/predict/FW_12/1", json={"features": [...]})
result = response.json()