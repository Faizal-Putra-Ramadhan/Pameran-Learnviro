import os
import time

from flask import Flask, Response, jsonify, render_template

from detector import EmotionDetector
from sensors import SensorHub

app = Flask(__name__)
detector = EmotionDetector(
    model_path=os.environ.get("MODEL_PATH", "emotion_model.pkl"),
    camera_index=int(os.environ.get("CAMERA_INDEX", 1)),
)
sensors = SensorHub(interval=2.0)
detector.start()


@app.route("/")
def index():
    return render_template("index.html")


def mjpeg():
    while True:
        frame = detector.get_jpeg()
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.04)  # ~25 fps maks


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def status():
    return jsonify({**detector.get_state(), "sensor": sensors.get()})


@app.route("/api/history")
def history():
    return jsonify(sensors.get_history())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
