import requests
import random
import time

server_url = "http://127.0.0.1:5000"

# ===== Get current session info from server =====
try:
    resp = requests.get(f"{server_url}/current_session")
    session = resp.json()
    hand = session["hand"]
    handwriting_type = session["type"]
    filename = session["filename"]
    print(f"Using session: {filename}, Hand: {hand}, Type: {handwriting_type}")
except Exception as e:
    print("Error getting session info:", e)
   
# ===== Send simulated data =====
url = f"{server_url}/data"
sampling_interval = 0.05  # ~20 Hz

while True:
    data = {
        "ax": round(random.uniform(-1, 1), 2),
        "ay": round(random.uniform(-1, 1), 2),
        "az": round(random.uniform(9.7, 9.9), 2),  # include gravity
        "gx": round(random.uniform(-10, 10), 2),
        "gy": round(random.uniform(-10, 10), 2),
        "gz": round(random.uniform(-10, 10), 2),
        
        
    }

    try:
        response = requests.post(url, json=data)
        print(f"Sent data: {data}, Server response: {response.status_code}")
    except Exception as e:
        print("Error:", e)

    time.sleep(sampling_interval)
