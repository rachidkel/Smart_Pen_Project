from flask import Flask, request, jsonify, render_template, redirect, url_for
import csv
import os
from datetime import datetime

app = Flask(__name__)

BASE_DATA_FOLDER = "data"

is_recording = False
current_word = ""
current_hand = ""
current_quality = ""
file_path = None

@app.route("/")
def index():
    return render_template("index.html", recording=is_recording)

@app.route("/start", methods=["POST"])
def start():
    global is_recording, current_word, current_hand, current_quality, file_path

    current_word = request.form.get("arabic_word")
    current_hand = request.form.get("hand")
    current_quality = request.form.get("quality")
    custom_name = request.form.get("filename")

    folder_path = os.path.join(
        BASE_DATA_FOLDER,
        f"{current_hand}_hand",
        current_quality
    )
    os.makedirs(folder_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{custom_name}_{timestamp}.csv"
    file_path = os.path.join(folder_path, filename)

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "ax", "ay", "az",
            "gx", "gy", "gz",
            "hand",
            "quality",
            "arabic_word"
        ])

    is_recording = True
    return redirect(url_for("index"))

@app.route("/stop", methods=["POST"])
def stop():
    global is_recording
    is_recording = False
    return redirect(url_for("index"))

@app.route("/data", methods=["POST"])
def receive_data():
    global is_recording, file_path

    if not is_recording or not file_path:
        return jsonify({"status": "ignored"}), 200

    data = request.get_json()

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            data["ax"], data["ay"], data["az"],
            data["gx"], data["gy"], data["gz"],
            current_hand,
            current_quality,
            current_word
        ])

    return jsonify({"status": "saved"}), 200

if __name__ == "__main__":
    print("🚀 Open http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
