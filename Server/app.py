from flask import Flask, request, jsonify, render_template, redirect, url_for
import csv
import datetime
import os

app = Flask(__name__)

# ===== Global State =====
is_recording = False
current_file = None
current_path = None

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# ===== Routes =====

@app.route('/')
def index():
    return render_template(
        "index.html",
        recording=is_recording,
        filename=current_file
    )


@app.route('/start', methods=['POST'])
def start_recording():
    global is_recording, current_file, current_path

    session_name = request.form.get("filename", "session").strip()
    hand = request.form.get("hand")
    quality = request.form.get("quality")
    arabic_word = request.form.get("arabic_word", "").strip()

    # ---- Validation ----
    if not hand or not quality:
        return "❌ Hand and Quality are required", 400

    hand = hand.lower()
    quality = quality.lower()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    folder_path = os.path.join(
        DATA_FOLDER,
        f"{hand}_hand",
        quality
    )
    os.makedirs(folder_path, exist_ok=True)

    current_file = f"{session_name}_{hand}_{quality}_{timestamp}.csv"
    current_path = os.path.join(folder_path, current_file)

    with open(current_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "ax", "ay", "az",
            "gx", "gy", "gz",
            "hand",
            "quality",
            "arabic_word"
        ])

    app.config["HAND"] = hand
    app.config["QUALITY"] = quality
    app.config["ARABIC_WORD"] = arabic_word

    is_recording = True
    print(f"✅ Recording started: {current_path}")

    return redirect(url_for("index"))


@app.route('/stop', methods=['POST'])
def stop_recording():
    global is_recording
    is_recording = False
    print("🛑 Recording stopped")
    return redirect(url_for("index"))


@app.route('/data', methods=['POST'])
def receive_data():
    global is_recording, current_path

    if not is_recording or not current_path:
        return jsonify({"status": "ignored"}), 200

    data = request.get_json(force=True)

    with open(current_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.datetime.now().isoformat(),
            data.get("ax"),
            data.get("ay"),
            data.get("az"),
            data.get("gx"),
            data.get("gy"),
            data.get("gz"),
            app.config.get("HAND"),
            app.config.get("QUALITY"),
            app.config.get("ARABIC_WORD")
        ])

    return jsonify({"status": "ok"}), 200


@app.route('/status')
def status():
    return jsonify({
        "recording": is_recording,
        "file": current_file
    })


if __name__ == "__main__":
    print("🚀 Server running on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
