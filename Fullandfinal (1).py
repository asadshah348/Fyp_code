from flask import Flask, request, render_template_string, jsonify, Response
from inference_sdk import InferenceHTTPClient
import os
import base64
import time
import random
import threading
import hashlib
import io
import wave
import subprocess
import signal
import json
import webbrowser
# ============================================================
# AWS S3 IMPORTS
# ============================================================
import boto3
from botocore.exceptions import ClientError

# ============================================================
# AUDIO IMPORTS - Real BeeCNN Model
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import tempfile

# ============================================================
# JETSON CAMERA & AUDIO IMPORTS
# ============================================================
import cv2

from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

# Try to import Jetson-specific GPIO
try:
    import Jetson.GPIO as GPIO
    JETSON_GPIO_AVAILABLE = True
except ImportError:
    JETSON_GPIO_AVAILABLE = False
    print("WARNING: Jetson.GPIO not available. Running in PC fallback mode.")

# Try pyaudio for audio capture
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("WARNING: PyAudio not available. Audio capture will use arecord fallback.")

app = Flask(__name__)

# Folders
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
AUDIO_UPLOAD_FOLDER = "audio_uploads"
VIDEO_UPLOAD_FOLDER = "video_uploads"
CAPTURE_FOLDER = "captures"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(AUDIO_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

# History data is stored on AWS S3 (not locally)
HISTORY_S3_KEY = "data/detection_history.json"
AUDIO_HISTORY_S3_KEY = "data/audio_history.json"
ALERTS_S3_KEY = "data/alerts.json"
DASHBOARD_STATS_S3_KEY = "data/dashboard_stats.json"

# Roboflow Clients
# `client` (workspace: asad-fnvcs) — used for IMAGE uploads only
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="cSTL1ItPm98da1Y3USmT"
)

# `realtime_client` (workspace: team-yolo) — used for LIVE webcam polling and
# video-frame inference via HTTP run_workflow.
realtime_client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="OkvGCd6FNLzoqkyEJ29k"
)

# ============================================================
# AWS S3 CONFIGURATION
# ============================================================
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "honeybee-fyp-2026")

if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    print("WARNING: AWS credentials not found in environment variables!")

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# ============================================================
# HISTORY & ALERT HELPERS (AWS S3 backed)
# ============================================================
def load_json_from_s3(s3_key, default=None):
    """Load JSON data from S3. Returns default if key is missing."""
    if default is None:
        default = []
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except ClientError as e:
        code = e.response.get('Error', {}).get('Code', '')
        if code in ('NoSuchKey', '404'):
            return default
        print(f"S3 load error ({s3_key}): {e}")
        return default
    except Exception as e:
        print(f"S3 load failed ({s3_key}): {e}")
        return default

def save_json_to_s3(s3_key, data):
    """Persist JSON data to S3."""
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=json.dumps(data, indent=2).encode('utf-8'),
            ContentType='application/json'
        )
        return True
    except Exception as e:
        print(f"S3 save failed ({s3_key}): {e}")
        return False

def add_alert(title, message, severity="info"):
    alerts = load_json_from_s3(ALERTS_S3_KEY)
    alerts.insert(0, {
        "timestamp": datetime.now().isoformat(),
        "title": title,
        "message": message,
        "severity": severity
    })
    alerts = alerts[:200]
    save_json_to_s3(ALERTS_S3_KEY, alerts)

def log_cv_detection(bee_count, source="camera"):
    history = load_json_from_s3(HISTORY_S3_KEY)
    history.append({
        "timestamp": datetime.now().isoformat(),
        "bee_count": int(bee_count),
        "source": source
    })
    history = history[-1000:]
    save_json_to_s3(HISTORY_S3_KEY, history)

    if bee_count > 250:
        add_alert("High Bee Activity", f"{bee_count} bees detected via {source}", "warning")
    elif bee_count == 0:
        add_alert("No Bees Detected", f"Zero bees detected from {source}", "info")

def log_audio_analysis(result):
    history = load_json_from_s3(AUDIO_HISTORY_S3_KEY)
    history.append({
        "timestamp": datetime.now().isoformat(),
        "estimated_count": int(result.get("estimated_count", 0)),
        "level": result.get("level", "Unknown"),
        "swarming_probability": int(result.get("swarming_probability", 0)),
        "anomaly_detected": bool(result.get("anomaly_detected", False))
    })
    history = history[-1000:]
    save_json_to_s3(AUDIO_HISTORY_S3_KEY, history)

    if result.get("anomaly_detected"):
        add_alert("Audio Anomaly", f"Anomaly detected: {result.get('level')}", "critical")
    if result.get("swarming_probability", 0) > 50:
        add_alert("Swarm Risk", f"Swarming probability {result.get('swarming_probability')}%", "critical")
    if "Extreme" in result.get("level", "") or "Very High" in result.get("level", ""):
        add_alert("Extreme Population", f"Level: {result.get('level')}", "warning")

DEFAULT_DASHBOARD_STATS = {
    "captures": 0,
    "audio_samples": 0,
    "videos_processed": 0,
    "total_detections": 0
}

_stats_lock = threading.Lock()

def load_dashboard_stats():
    """Load dashboard counters from S3 (with defaults)."""
    stats = load_json_from_s3(DASHBOARD_STATS_S3_KEY, default=dict(DEFAULT_DASHBOARD_STATS))
    # Ensure all expected keys exist
    for k, v in DEFAULT_DASHBOARD_STATS.items():
        stats.setdefault(k, v)
    return stats

def increment_dashboard_stats(captures=0, audio_samples=0, videos_processed=0, total_detections=0):
    """Atomically increment dashboard counters on S3."""
    with _stats_lock:
        stats = load_dashboard_stats()
        stats["captures"] = int(stats.get("captures", 0)) + int(captures)
        stats["audio_samples"] = int(stats.get("audio_samples", 0)) + int(audio_samples)
        stats["videos_processed"] = int(stats.get("videos_processed", 0)) + int(videos_processed)
        stats["total_detections"] = int(stats.get("total_detections", 0)) + int(total_detections)
        save_json_to_s3(DASHBOARD_STATS_S3_KEY, stats)
        return stats

# Known dummy alerts from an earlier build — purged from S3 on startup.
_DUMMY_ALERT_MESSAGES = {
    "Node Topi-01 deployed in Topi, KPK",
    "Node Ghazi-02 deployed in Ghazi, KPK",
    "Swarm event detected on Node Topi-01",
    "Low battery warning on Node Ghazi-02",
}

def init_test_data():
    """Clean any pre-existing dummy alerts from S3 and log a real device-online
    alert for this startup."""
    alerts = load_json_from_s3(ALERTS_S3_KEY)
    cleaned = [a for a in alerts if a.get("message") not in _DUMMY_ALERT_MESSAGES]
    if len(cleaned) != len(alerts):
        save_json_to_s3(ALERTS_S3_KEY, cleaned)

    add_alert(
        "Device Online",
        f"BeeDetect AI started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "info",
    )

# ============================================================
# JETSON HARDWARE CONFIGURATION
# ============================================================
# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAPTURE_FPS = 30

# Audio settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_DURATION = 3            # default duration for legacy callers
AUDIO_MAX_DURATION = 20       # hard cap for user-controlled recording

# Jetson CSI Camera GStreamer pipeline
def get_jetson_gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0
):
    """Return GStreamer pipeline for Jetson CSI camera"""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, "
        f"format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )

# Global camera state
camera = None
camera_lock = threading.Lock()
is_capturing = False
current_frame = None
audio_recording_process = None

# Live WebRTC detection state (server-side)
live_session = None                # {"active": bool} sentinel
live_capture_thread = None         # thread that keeps pulling raw frames from the camera
live_inference_thread = None       # thread that sends frames to Roboflow and publishes annotations
latest_raw_frame = None            # np.ndarray of the most recent camera frame (unannotated)
latest_annotated_frame = None      # np.ndarray of the latest Roboflow-annotated frame
latest_live_count = 0              # latest bee count from Roboflow
live_session_lock = threading.Lock()

# ============================================================
# REAL AUDIO MODEL - BeeCNN
# ============================================================
class BeeCNN(nn.Module):
    def __init__(self):
        super(BeeCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_model = BeeCNN().to(device)

MODEL_PATH = "bee_audio_model.pth"
if os.path.exists(MODEL_PATH):
    audio_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    audio_model.eval()
    AUDIO_MODEL_LOADED = True
else:
    AUDIO_MODEL_LOADED = False
    print(f"WARNING: {MODEL_PATH} not found. Audio analysis will use fallback.")

# Audio settings
SAMPLE_RATE = 16000
DURATION = 3
FIXED_LENGTH = SAMPLE_RATE * DURATION
N_MELS = 128


def extract_mel(file_path):
    """Extract mel spectrogram from audio file"""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(audio) < FIXED_LENGTH:
        audio = np.pad(audio, (0, FIXED_LENGTH - len(audio)))
    else:
        audio = audio[:FIXED_LENGTH]
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float()


def get_bee_level_and_range(probs):
    """6-level classification from probabilities"""
    low, med, high = probs
    score = (low * 1) + (med * 2) + (high * 3)
    if score < 1.3:
        return "Very Low", "0 - 30"
    elif score < 1.7:
        return "Low", "30 - 100"
    elif score < 2.1:
        return "Medium", "100 - 300"
    elif score < 2.4:
        return "High", "300 - 600"
    elif score < 2.7:
        return "Very High", "600 - 1000"
    else:
        return "Extreme Swarm", "1000+"


def predict_audio(file_path):
    """Run real audio prediction with BeeCNN. Deterministic: same audio → same output."""
    if not AUDIO_MODEL_LOADED:
        return generate_fallback_audio_analysis(file_path)

    x = extract_mel(file_path).to(device)
    with torch.no_grad():
        output = audio_model(x)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]

    level, bee_range = get_bee_level_and_range(probs)

    activity_intensity = int((probs[1] * 0.5 + probs[2] * 1.0) * 100)
    stress_level = "High" if probs[2] > 0.5 else "Moderate" if probs[1] > 0.5 else "Low"
    swarming_probability = int(probs[2] * 100)
    anomaly_detected = bool(probs[2] > 0.6)  # ← Fixed: numpy.bool_ → Python bool

    # Deterministic estimated_count: interpolate within the predicted range using probs[2].
    # Same audio → same probs → same count (no np.random.uniform).
    if '-' in bee_range:
        parts = bee_range.split('-')
        r_start = int(parts[0].strip())
        r_end = int(parts[1].strip().replace('+', ''))
        estimated_count = int(round(r_start + (r_end - r_start) * float(probs[2])))
    else:
        estimated_count = int(bee_range.replace('+', '').strip())

    frequency_data = []
    for i in range(20):
        freq_val = 20 + int(80 * (probs[i % 3] * (1 - i/40) + 0.1))
        frequency_data.append(freq_val)

    return {
        "level": level,
        "bee_range": bee_range,
        "low_prob": float(probs[0]),
        "med_prob": float(probs[1]),
        "high_prob": float(probs[2]),
        "activity_intensity": activity_intensity,
        "stress_level": stress_level,
        "swarming_probability": swarming_probability,
        "anomaly_detected": anomaly_detected,
        "estimated_count": estimated_count,
        "frequency_data": frequency_data,
        "model_loaded": True
    }


def _hash_audio_identifier(path_or_name):
    """Build a stable identifier from the audio content so re-uploads of the
    same file — regardless of the server-side timestamped filename — produce
    the same fallback result. Falls back to hashing the name string if the
    path doesn't exist or can't be read."""
    if os.path.exists(path_or_name) and os.path.isfile(path_or_name):
        try:
            h = hashlib.md5()
            with open(path_or_name, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception as e:
            print(f"audio content hash failed ({e}); falling back to name hash")
    return hashlib.md5(str(path_or_name).encode()).hexdigest()


def generate_fallback_audio_analysis(path_or_name):
    """Fallback deterministic audio analysis when model not loaded.
    Uses a hash derived from the audio *content* so identical audio — uploaded
    any number of times with any filename — always produces the same result.
    Also keeps estimated_count inside the reported bee_range."""
    identifier = _hash_audio_identifier(path_or_name)
    get_val = lambda key, min_v, max_v: min_v + (int(hashlib.md5((identifier + key).encode()).hexdigest(), 16) % (max_v - min_v + 1))

    activity = get_val("_activity", 40, 95)
    classifications = ['Normal Hive', 'Swarming Risk', 'Queenless']
    class_index = get_val("_class", 0, 2)
    stress_levels = ['Low', 'Moderate', 'High']
    stress_index = get_val("_stress", 0, 2)
    swarming_prob = get_val("_swarm", 5, 85)
    anomaly_detected = bool(get_val("_anomaly", 0, 1) == 1)

    frequency_data = []
    for i in range(20):
        freq_val = get_val(f"_freq_{i}", 20, 100)
        frequency_data.append(freq_val)

    level = classifications[class_index]
    if level == 'Normal Hive':
        bee_range = "100 - 300"
    elif level == 'Swarming Risk':
        bee_range = "600 - 1000"
    else:
        bee_range = "30 - 100"

    # Deterministic estimated_count inside the predicted range — keep it
    # between 20% and 80% of the span so it doesn't hug the bounds.
    if '-' in bee_range:
        parts = bee_range.split('-')
        r_start = int(parts[0].strip())
        r_end = int(parts[1].strip().replace('+', ''))
        position = get_val("_pos", 20, 80) / 100.0
        estimated_count = int(round(r_start + (r_end - r_start) * position))
    else:
        estimated_count = int(bee_range.replace('+', '').strip())

    return {
        "level": level,
        "bee_range": bee_range,
        "low_prob": 0.33,
        "med_prob": 0.33,
        "high_prob": 0.34,
        "activity_intensity": activity,
        "stress_level": stress_levels[stress_index],
        "swarming_probability": swarming_prob,
        "anomaly_detected": anomaly_detected,
        "estimated_count": estimated_count,
        "frequency_data": frequency_data,
        "model_loaded": False
    }


def upload_to_s3(file_path, filename, folder="images"):
    """Upload a file to AWS S3 bucket"""
    try:
        s3_key = f"{folder}/{int(time.time())}_{filename}"
        s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
        s3_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        print(f"Uploaded to S3: {s3_url}")
        return s3_url
    except ClientError as e:
        print(f"S3 upload error: {e}")
        return None
    except Exception as e:
        print(f"S3 upload failed: {e}")
        return None


# ============================================================
# JETSON CAMERA FUNCTIONS
# ============================================================
def init_camera(camera_type="csi", device_id=0):
    """Initialize camera - supports CSI and USB cameras"""
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None

        if camera_type == "csi" and JETSON_GPIO_AVAILABLE:
            # Try CSI camera with GStreamer
            pipeline = get_jetson_gstreamer_pipeline(
                sensor_id=device_id,
                capture_width=CAMERA_WIDTH,
                capture_height=CAMERA_HEIGHT,
                framerate=CAPTURE_FPS
            )
            camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

            if not camera.isOpened():
                print("CSI camera failed, falling back to USB camera...")
                camera = cv2.VideoCapture(device_id)
        else:
            # USB camera or PC webcam
            camera = cv2.VideoCapture(device_id)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if camera.isOpened():
            print(f"Camera initialized: {camera_type} (device {device_id})")
            return True
        else:
            print("ERROR: Failed to open camera")
            camera = None
            return False


def release_camera():
    """Release camera resource"""
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            print("Camera released")


def capture_frame():
    """Capture a single frame from camera"""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            return None
        ret, frame = camera.read()
        if ret:
            return frame
        return None


def save_captured_frame(frame, filename=None):
    """Save captured frame to file and return path"""
    if filename is None:
        filename = f"capture_{int(time.time())}.jpg"
    filepath = os.path.join(CAPTURE_FOLDER, filename)
    cv2.imwrite(filepath, frame)
    return filepath


def frame_to_base64(frame):
    """Convert OpenCV frame to base64 string"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def generate_camera_feed():
    """Generate MJPEG stream from camera for live preview"""
    global camera
    while True:
        frame = capture_frame()
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)


# ============================================================
# JETSON AUDIO FUNCTIONS
# ============================================================

# Shared state for user-stoppable recording. The background thread writes
# into this dict; the HTTP handlers start/stop and read `path` after.
recording_state = {
    "active": False,
    "thread": None,
    "path": None,
    "process": None,         # arecord subprocess (when PyAudio is unavailable)
}
recording_lock = threading.Lock()


def _recording_worker(output_path, max_duration):
    """Background worker: records audio frames until `active` goes False OR
    max_duration is reached, whichever comes first. Always writes a valid WAV
    to `output_path` on exit."""
    if PYAUDIO_AVAILABLE:
        p = pyaudio.PyAudio()
        frames = []
        stream = None
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=AUDIO_CHANNELS,
                rate=AUDIO_SAMPLE_RATE,
                input=True,
                frames_per_buffer=1024,
            )
            max_chunks = int(AUDIO_SAMPLE_RATE / 1024 * max_duration)
            read_chunks = 0
            print(f"Recording up to {max_duration} seconds (stoppable)...")
            while recording_state["active"] and read_chunks < max_chunks:
                try:
                    data = stream.read(1024, exception_on_overflow=False)
                    frames.append(data)
                    read_chunks += 1
                except Exception as e:
                    print(f"Recording read error: {e}")
                    break
        finally:
            if stream is not None:
                try: stream.stop_stream()
                except Exception: pass
                try: stream.close()
                except Exception: pass
            p.terminate()

        # Always write whatever we got so the analyzer has something to work with
        if not frames:
            # fall back to a short silent clip so downstream code doesn't choke
            frames = [b"\x00" * 2048]
        wf = wave.open(output_path, "wb")
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(2)  # int16
        wf.setframerate(AUDIO_SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
        wf.close()
    else:
        # arecord fallback — record for up to max_duration, terminate early if asked
        try:
            proc = subprocess.Popen([
                "arecord",
                "-D", "plughw:1,0",
                "-d", str(max_duration),
                "-r", str(AUDIO_SAMPLE_RATE),
                "-c", str(AUDIO_CHANNELS),
                "-f", "S16_LE",
                output_path,
            ])
            recording_state["process"] = proc
            while recording_state["active"] and proc.poll() is None:
                time.sleep(0.1)
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
            recording_state["process"] = None
        except (FileNotFoundError, Exception) as e:
            print(f"arecord unavailable ({e}); creating synthetic audio")
            create_test_tone(output_path, 3)

    recording_state["active"] = False


def start_recording(output_path, max_duration=AUDIO_MAX_DURATION):
    """Kick off a background recording. Returns True on success, False if
    something is already recording."""
    with recording_lock:
        if recording_state["active"] or (
            recording_state["thread"] is not None and recording_state["thread"].is_alive()
        ):
            return False
        recording_state["active"] = True
        recording_state["path"] = output_path
        t = threading.Thread(
            target=_recording_worker,
            args=(output_path, max_duration),
            daemon=True,
        )
        recording_state["thread"] = t
        t.start()
    return True


def stop_recording(wait_timeout=6.0):
    """Signal the recording worker to stop and wait until it has flushed the
    WAV file. Returns the output path, or None if nothing was recording."""
    with recording_lock:
        if not recording_state["active"] and not (
            recording_state["thread"] is not None and recording_state["thread"].is_alive()
        ):
            return None
        recording_state["active"] = False
        # If arecord is mid-flight, terminate it so we don't wait the full -d duration
        proc = recording_state.get("process")
        if proc is not None and proc.poll() is None:
            try: proc.terminate()
            except Exception: pass

    t = recording_state.get("thread")
    if t is not None:
        t.join(timeout=wait_timeout)
    return recording_state.get("path")


def record_audio_jetson(duration=AUDIO_DURATION, output_path=None):
    """Record audio from microphone on Jetson"""
    if output_path is None:
        output_path = os.path.join(AUDIO_UPLOAD_FOLDER, f"capture_{int(time.time())}.wav")

    if PYAUDIO_AVAILABLE:
        # Use PyAudio for recording
        p = pyaudio.PyAudio()
        frames = []

        stream = p.open(
            format=pyaudio.paInt16,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_SAMPLE_RATE,
            input=True,
            frames_per_buffer=1024
        )

        print(f"Recording audio for {duration} seconds...")
        for _ in range(0, int(AUDIO_SAMPLE_RATE / 1024 * duration)):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save as WAV
        wf = wave.open(output_path, 'wb')
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(AUDIO_SAMPLE_RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    else:
        # Fallback to arecord (Linux/Jetson)
        try:
            subprocess.run([
                'arecord',
                '-D', 'plughw:1,0',  # Default USB mic
                '-d', str(duration),
                '-r', str(AUDIO_SAMPLE_RATE),
                '-c', str(AUDIO_CHANNELS),
                '-f', 'S16_LE',
                output_path
            ], check=True, timeout=duration + 5)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("WARNING: arecord not available. Creating synthetic audio.")
            # Create a test tone as fallback
            create_test_tone(output_path, duration)

    return output_path


def create_test_tone(output_path, duration=3, frequency=200):
    """Create a test tone WAV file as fallback"""
    sample_rate = AUDIO_SAMPLE_RATE
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Create a tone with some harmonics to simulate bee buzzing
    tone = np.sin(frequency * t * 2 * np.pi) * 0.3
    tone += np.sin(frequency * 2 * t * 2 * np.pi) * 0.2
    tone += np.sin(frequency * 0.5 * t * 2 * np.pi) * 0.1
    # Add some noise
    tone += np.random.normal(0, 0.05, len(t))
    # Normalize
    tone = np.int16(tone / np.max(np.abs(tone)) * 32767)

    wf = wave.open(output_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(tone.tobytes())
    wf.close()


# ============================================================
# VIDEO PROCESSING FUNCTIONS
# ============================================================
def sanitize_video(input_path):
    """
    Re-encode uploaded video to a clean H.264/yuv420p mp4 so OpenCV's FFmpeg
    decoder stops emitting 'mmco: unref short failure' warnings and returns
    consistent frames. Returns the fixed path, or the original if ffmpeg
    is missing / the re-encode fails.
    """
    try:
        base, _ = os.path.splitext(input_path)
        fixed_path = f"{base}_sanitized.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-err_detect", "ignore_err",
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",  # strip audio, not used
                "-loglevel", "error",
                fixed_path,
            ],
            check=True,
            timeout=600,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if os.path.exists(fixed_path) and os.path.getsize(fixed_path) > 0:
            return fixed_path
    except FileNotFoundError:
        print("ffmpeg not found, skipping video sanitization")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg re-encode failed (return code {e.returncode}), using original")
    except Exception as e:
        print(f"Video sanitize skipped: {e}")
    return input_path


NUM_VIDEO_FRAMES = 5  # Evenly-spaced frames sampled from each uploaded video


def process_video_file(video_path):
    """
    Split the uploaded video into NUM_VIDEO_FRAMES evenly-spaced frames,
    run each frame through the Roboflow workflow (team-yolo), and upload
    ONLY the annotated output frames to AWS S3. The raw source video is
    NOT uploaded. Returns a summary dict with per-frame results (count,
    timestamp, S3 URL, inline base64 preview) for the frontend gallery.
    """
    # Re-encode for clean seeking on weird codecs; no-op if ffmpeg missing
    sanitized_path = sanitize_video(video_path)

    cap = cv2.VideoCapture(sanitized_path)
    if not cap.isOpened():
        if sanitized_path != video_path and os.path.exists(sanitized_path):
            try: os.remove(sanitized_path)
            except OSError: pass
        return None, "Failed to open video file"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total_frames / fps) if fps > 0 else 0

    if total_frames < 1:
        cap.release()
        if sanitized_path != video_path and os.path.exists(sanitized_path):
            try: os.remove(sanitized_path)
            except OSError: pass
        return None, "Video has no readable frames"

    # Pick frames at 10%, 30%, 50%, 70%, 90% of the video
    positions = [0.10, 0.30, 0.50, 0.70, 0.90]
    frame_indices = [
        max(0, min(total_frames - 1, int(total_frames * p))) for p in positions
    ]

    timestamp = int(time.time())
    s3_folder = f"video/processed/{timestamp}"

    frame_results = []
    total_bees = 0
    max_bees = 0
    peak_position = 0

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            frame_results.append({
                "position": i + 1,
                "frame_index": int(frame_idx),
                "timestamp": round(frame_idx / fps, 2) if fps else 0,
                "bee_count": 0,
                "s3_url": None,
                "raw_base64": None,
                "annotated_base64": None,
                "error": "Could not read frame",
            })
            continue

        temp_path = os.path.join(
            CAPTURE_FOLDER, f"video_{timestamp}_frame_{i + 1}.jpg"
        )
        cv2.imwrite(temp_path, frame)

        # Encode the raw (before) frame as base64 for the frontend before/after view
        raw_b64 = None
        try:
            ok, raw_buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                raw_b64 = base64.b64encode(raw_buf.tobytes()).decode("utf-8")
        except Exception as e:
            print(f"Raw frame encode error: {e}")

        bee_count = 0
        annotated_b64 = None
        s3_url = None
        error = None

        try:
            result = realtime_client.run_workflow(
                workspace_name="team-yolo",
                workflow_id="detect-count-and-visualize-2",
                images={"image": temp_path},
                use_cache=True,
            )
            data = result[0] if result else {}
            bee_count = int(data.get("count_objects", 0) or 0)
            annotated_b64 = data.get("output_image")

            if annotated_b64:
                # Write the annotated frame to disk so we can upload it to S3
                annotated_path = os.path.join(
                    RESULT_FOLDER, f"annotated_{timestamp}_frame_{i + 1}.jpg"
                )
                try:
                    with open(annotated_path, "wb") as f:
                        f.write(base64.b64decode(annotated_b64))
                    # Upload ONLY the annotated frame — the original video is NOT uploaded
                    s3_url = upload_to_s3(
                        annotated_path,
                        f"frame_{i + 1}.jpg",
                        folder=s3_folder,
                    )
                finally:
                    if os.path.exists(annotated_path):
                        try: os.remove(annotated_path)
                        except OSError: pass
        except Exception as e:
            error = str(e)
            print(f"Video frame {i + 1} inference error: {e}")
        finally:
            if os.path.exists(temp_path):
                try: os.remove(temp_path)
                except OSError: pass

        total_bees += bee_count
        if bee_count > max_bees:
            max_bees = bee_count
            peak_position = i + 1

        frame_results.append({
            "position": i + 1,
            "frame_index": int(frame_idx),
            "timestamp": round(frame_idx / fps, 2) if fps else 0,
            "bee_count": bee_count,
            "s3_url": s3_url,
            "raw_base64": raw_b64,
            "annotated_base64": annotated_b64,
            **({"error": error} if error else {}),
        })

    cap.release()
    if sanitized_path != video_path and os.path.exists(sanitized_path):
        try: os.remove(sanitized_path)
        except OSError: pass

    processed_count = sum(1 for r in frame_results if r.get("s3_url") or r.get("annotated_base64"))
    avg_bees = round(total_bees / len(frame_results), 2) if frame_results else 0

    summary = {
        "total_frames": total_frames,
        "processed_frames": processed_count,
        "num_sampled_frames": len(frame_results),
        "fps": round(fps, 2),
        "duration": round(duration, 2),
        "total_bees_detected": total_bees,
        "average_bees_per_frame": avg_bees,
        "max_bees_in_single_frame": max_bees,
        "peak_position": peak_position,
        "frame_results": frame_results,
    }
    return summary, None


# ============================================================
# HTML TEMPLATE
# ============================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BeeDetect AI - Jetson Nano Edition</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --bg-card-hover: #334155;
            --accent-primary: #f59e0b;
            --accent-secondary: #fbbf24;
            --accent-green: #10b981;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-red: #ef4444;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border-color: #334155;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            min-height: 100vh;
            color: var(--text-primary);
            line-height: 1.6;
            overflow-x: hidden;
        }

        .loading-bee-container {
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            pointer-events: none;
            overflow: hidden;
            z-index: 1;
        }

        .loading-bee {
            position: absolute;
            font-size: 1.8rem;
            opacity: 0.7;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
        }

        @keyframes flyBee1 {
            0% { transform: translate(-10vw, 10vh) rotate(15deg); }
            20% { transform: translate(20vw, 5vh) rotate(-10deg); }
            40% { transform: translate(40vw, 15vh) rotate(20deg); }
            60% { transform: translate(60vw, 8vh) rotate(-15deg); }
            80% { transform: translate(80vw, 20vh) rotate(10deg); }
            100% { transform: translate(110vw, 12vh) rotate(-5deg); }
        }

        @keyframes flyBee2 {
            0% { transform: translate(110vw, 60vh) rotate(-20deg) scaleX(-1); }
            25% { transform: translate(80vw, 50vh) rotate(15deg) scaleX(-1); }
            50% { transform: translate(50vw, 65vh) rotate(-10deg) scaleX(-1); }
            75% { transform: translate(20vw, 55vh) rotate(20deg) scaleX(-1); }
            100% { transform: translate(-10vw, 70vh) rotate(-15deg) scaleX(-1); }
        }

        @keyframes flyBee3 {
            0% { transform: translate(-5vw, 85vh) rotate(25deg); }
            30% { transform: translate(30vw, 75vh) rotate(-20deg); }
            60% { transform: translate(70vw, 90vh) rotate(15deg); }
            100% { transform: translate(105vw, 80vh) rotate(-25deg); }
        }

        @keyframes flyBee4 {
            0% { transform: translate(50vw, -10vh) rotate(180deg); }
            25% { transform: translate(30vw, 25vh) rotate(200deg); }
            50% { transform: translate(60vw, 50vh) rotate(160deg); }
            75% { transform: translate(40vw, 75vh) rotate(190deg); }
            100% { transform: translate(55vw, 110vh) rotate(170deg); }
        }

        @keyframes flyBee5 {
            0% { transform: translate(90vw, 30vh) rotate(-30deg) scaleX(-1); }
            20% { transform: translate(65vw, 45vh) rotate(25deg) scaleX(-1); }
            40% { transform: translate(75vw, 25vh) rotate(-20deg) scaleX(-1); }
            60% { transform: translate(45vw, 55vh) rotate(30deg) scaleX(-1); }
            80% { transform: translate(25vw, 35vh) rotate(-25deg) scaleX(-1); }
            100% { transform: translate(5vw, 50vh) rotate(20deg) scaleX(-1); }
        }

        .lbee1 { animation: flyBee1 18s linear infinite; }
        .lbee2 { animation: flyBee2 22s linear infinite; animation-delay: -5s; }
        .lbee3 { animation: flyBee3 20s linear infinite; animation-delay: -10s; }
        .lbee4 { animation: flyBee4 25s linear infinite; animation-delay: -7s; }
        .lbee5 { animation: flyBee5 19s linear infinite; animation-delay: -3s; }
        .lbee6 { animation: flyBee1 24s linear infinite; animation-delay: -12s; top: 40vh; font-size: 1.2rem; }
        .lbee7 { animation: flyBee2 21s linear infinite; animation-delay: -8s; top: 25vh; font-size: 1.4rem; }

        .honeycomb-bg {
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            pointer-events: none;
            z-index: 0;
            opacity: 0.02;
            background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M30 0l25.98 15v30L30 60 4.02 45V15L30 0z' fill='none' stroke='%23f59e0b' stroke-width='1'/%3E%3C/svg%3E");
            background-size: 80px 80px;
        }

        .glow-line {
            position: fixed;
            top: 0; left: 0; width: 100%; height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent-primary), var(--accent-secondary), var(--accent-primary), transparent);
            z-index: 9997;
            opacity: 0.5;
            animation: shimmer 4s ease-in-out infinite;
        }

        @keyframes shimmer {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 0.7; }
        }

        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 2rem;
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo-icon {
            width: 45px;
            height: 45px;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            animation: buzz 2s ease-in-out infinite;
            box-shadow: 0 0 20px rgba(245, 158, 11, 0.3);
        }

        @keyframes buzz {
            0%, 100% { transform: scale(1) rotate(0deg); }
            25% { transform: scale(1.1) rotate(-5deg); }
            50% { transform: scale(0.95) rotate(5deg); }
            75% { transform: scale(1.05) rotate(-3deg); }
        }

        .logo-text h2 { font-size: 1.25rem; font-weight: 700; color: var(--text-primary); }
        .logo-text p { font-size: 0.75rem; color: var(--text-secondary); }

        .nav-status {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid var(--accent-green);
            border-radius: 50px;
            font-size: 0.85rem;
            color: var(--accent-green);
        }

        .aws-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: rgba(245, 158, 11, 0.15);
            border: 1px solid var(--accent-primary);
            border-radius: 50px;
            font-size: 0.85rem;
            color: var(--accent-primary);
        }

        .jetson-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: rgba(59, 130, 246, 0.15);
            border: 1px solid var(--accent-blue);
            border-radius: 50px;
            font-size: 0.85rem;
            color: var(--accent-blue);
        }

        .status-dot {
            width: 8px; height: 8px;
            background: var(--accent-green);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .hero {
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(180deg, rgba(59, 130, 246, 0.12) 0%, transparent 100%);
            position: relative;
        }

        .hero h1 {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .hero p {
            font-size: 1.1rem;
            color: var(--text-secondary);
            max-width: 700px;
            margin: 0 auto 2rem;
        }

        .hardware-info {
            display: flex;
            justify-content: center;
            gap: 2rem;
            flex-wrap: wrap;
            margin-top: 1.5rem;
        }

        .hw-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        .hw-item i { color: var(--accent-blue); }
        .hw-item.active i { color: var(--accent-green); }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            max-width: 1100px;
            margin: 0 auto 3rem;
            padding: 0 2rem;
        }

        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            transition: all 0.3s ease;
        }

        .stat-card:hover {
            background: var(--bg-card-hover);
            transform: translateY(-3px);
            box-shadow: 0 8px 30px rgba(59, 130, 246, 0.15);
        }

        .stat-icon {
            width: 50px; height: 50px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
        }

        .stat-icon.camera { background: rgba(59, 130, 246, 0.2); color: var(--accent-blue); }
        .stat-icon.mic { background: rgba(139, 92, 246, 0.2); color: var(--accent-purple); }
        .stat-icon.detections { background: rgba(245, 158, 11, 0.2); color: var(--accent-primary); }
        .stat-icon.video { background: rgba(16, 185, 129, 0.2); color: var(--accent-green); }

        .mode-toggle {
            display: flex;
            justify-content: center;
            gap: 0.5rem;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }

        .mode-btn {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.875rem 1.5rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.95rem;
            border: 2px solid var(--border-color);
            background: var(--bg-card);
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .mode-btn.active {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            border-color: var(--accent-primary);
            color: var(--bg-primary);
            box-shadow: 0 4px 20px rgba(245, 158, 11, 0.4);
        }

        .mode-btn:hover:not(.active) {
            border-color: var(--accent-primary);
            color: var(--text-primary);
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 2rem 3rem;
            position: relative;
            z-index: 1;
        }

        .content-section { display: none; }
        .content-section.active {
            display: block;
            animation: fadeIn 0.5s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1.5rem;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }

        .card:hover {
            border-color: rgba(59, 130, 246, 0.3);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
        }

        .card-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }

        .card-header i {
            color: var(--accent-primary);
            font-size: 1.25rem;
        }

        .card-header h3 {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .camera-section {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .camera-feed-wrapper {
            position: relative;
            width: 100%;
            background: #000;
            border-radius: 16px;
            overflow: hidden;
            margin-bottom: 1rem;
        }

        .camera-feed-wrapper img {
            width: 100%;
            height: auto;
            display: block;
            min-height: 360px;
            object-fit: contain;
        }

        .camera-overlay {
            position: absolute;
            top: 1rem; left: 1rem;
            background: rgba(15, 23, 42, 0.8);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            color: var(--accent-green);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .camera-controls {
            display: flex;
            justify-content: center;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .camera-btn {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.875rem 2rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .camera-btn.primary {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            color: var(--bg-primary);
            box-shadow: 0 4px 20px rgba(245, 158, 11, 0.3);
        }

        .camera-btn.primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(245, 158, 11, 0.4);
        }

        .camera-btn.secondary {
            background: var(--bg-primary);
            color: var(--text-primary);
            border: 2px solid var(--border-color);
        }

        .camera-btn.secondary:hover {
            border-color: var(--accent-primary);
            color: var(--accent-primary);
        }

        .camera-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }

        .camera-btn.recording {
            background: linear-gradient(135deg, var(--accent-red), #dc2626);
            animation: pulse-record 1s infinite;
        }

        @keyframes pulse-record {
            0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
            50% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        }

        .capture-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .capture-item {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            overflow: hidden;
            border: 2px solid var(--border-color);
        }

        .capture-item img {
            width: 100%;
            height: auto;
            object-fit: contain;
            display: block;
        }

        .capture-item .capture-info {
            padding: 0.75rem;
            text-align: center;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        .audio-visualizer {
            height: 100px;
            background: rgba(139, 92, 246, 0.1);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
            position: relative;
            overflow: hidden;
        }

        .audio-visualizer .visualizer-bars {
            display: flex;
            align-items: flex-end;
            gap: 3px;
            height: 60px;
        }

        .vbar {
            width: 4px;
            background: linear-gradient(to top, var(--accent-purple), #a78bfa);
            border-radius: 2px;
            transition: height 0.1s ease;
        }

        .audio-status {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }

        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.875rem 0;
            border-bottom: 1px solid var(--border-color);
        }

        .metric-row:last-child { border-bottom: none; }

        .metric-label {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }

        .metric-value {
            font-weight: 600;
            color: var(--text-primary);
        }

        .detection-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .detection-stat {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }

        .detection-stat h4 {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--bg-primary);
        }

        .detection-stat p {
            font-size: 0.8rem;
            color: rgba(15, 23, 42, 0.8);
        }

        .batch-results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(450px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }

        .batch-result-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1.5rem;
        }

        .batch-result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .batch-result-badge {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            color: var(--bg-primary);
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .image-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            align-items: start;
        }

        .image-box {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 0.75rem;
            text-align: center;
        }

        .image-box h5 {
            margin-bottom: 0.75rem;
            color: var(--text-primary);
            font-size: 0.9rem;
        }

        .image-wrapper {
            position: relative;
            border-radius: 10px;
            overflow: hidden;
            border: 2px solid var(--border-color);
            background: var(--bg-primary);
        }

        .image-wrapper img {
            width: 100%;
            height: auto;
            max-height: 500px;
            object-fit: contain;
            display: block;
        }

        .image-label {
            position: absolute;
            top: 8px; left: 8px;
            background: rgba(15, 23, 42, 0.9);
            color: var(--accent-primary);
            padding: 0.2rem 0.6rem;
            border-radius: 20px;
            font-size: 0.7rem;
            font-weight: 600;
        }

        .loading-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(5px);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }

        .loading-overlay.show { display: flex; }

        .spinner {
            width: 80px; height: 80px;
            border: 4px solid var(--border-color);
            border-top-color: var(--accent-primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            position: relative;
        }

        .spinner::after {
            content: '🐝';
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            font-size: 1.5rem;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        .loading-text {
            margin-top: 1.5rem;
            font-size: 1.2rem;
            color: var(--accent-primary);
            font-weight: 600;
        }

        .loading-subtext {
            margin-top: 0.5rem;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }

        .audio-results {
            display: none;
            margin-top: 2rem;
        }

        .audio-results.show {
            display: block;
            animation: fadeIn 0.5s ease;
        }

        .audio-classification {
            background: linear-gradient(135deg, var(--accent-purple), #a78bfa);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 1.5rem;
        }

        .audio-classification h3 {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }

        .prob-bars { margin-top: 1rem; }

        .prob-bar-row {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.75rem;
        }

        .prob-label {
            width: 80px;
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-align: right;
        }

        .prob-bar-bg {
            flex: 1;
            height: 20px;
            background: var(--bg-primary);
            border-radius: 10px;
            overflow: hidden;
        }

        .prob-bar-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.8s ease;
        }

        .prob-value {
            width: 50px;
            font-size: 0.85rem;
            color: var(--text-primary);
            font-weight: 600;
        }

        .audio-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 1rem;
        }

        .audio-metric {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
        }

        .audio-metric h4 {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--accent-primary);
            margin-bottom: 0.25rem;
        }

        .audio-metric.status-low h4 { color: var(--accent-green); }
        .audio-metric.status-moderate h4 { color: var(--accent-primary); }
        .audio-metric.status-high h4 { color: var(--accent-red); }

        .frequency-chart-container {
            margin-top: 1.5rem;
            height: 200px;
        }

        .footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.9rem;
            border-top: 1px solid var(--border-color);
            position: relative;
            z-index: 1;
        }

        /* Upload area styles */
        .upload-area {
            border: 2px dashed var(--border-color);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            margin-bottom: 1rem;
        }

        .upload-area:hover {
            border-color: var(--accent-primary);
            background: rgba(245, 158, 11, 0.05);
        }

        .upload-area.dragover {
            border-color: var(--accent-primary);
            background: rgba(245, 158, 11, 0.1);
        }

        .upload-area i {
            font-size: 2.5rem;
            color: var(--accent-primary);
            margin-bottom: 1rem;
        }

        .upload-area p {
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        .upload-area input { display: none; }

        .file-list { margin-top: 1rem; }

        .file-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem;
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            margin-bottom: 0.5rem;
            border: 1px solid var(--border-color);
        }

        .file-item span {
            color: var(--text-primary);
            font-size: 0.9rem;
        }

        .file-item i {
            color: var(--accent-red);
            cursor: pointer;
        }

        /* Video results */
        .video-results {
            display: none;
            margin-top: 2rem;
        }

        .video-results.show {
            display: block;
            animation: fadeIn 0.5s ease;
        }

        .timeline-chart-container {
            margin-top: 1.5rem;
            height: 250px;
        }

        .video-stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .video-stat {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }

        .video-stat h4 {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--accent-primary);
        }

        .video-stat p {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        /* Dashboard / Toggle / Alert / Map styles */
        .toggle-group {
            display: flex;
            gap: 0.5rem;
        }

        .toggle-btn {
            padding: 0.5rem 1.25rem;
            border-radius: 10px;
            border: 1px solid var(--border-color);
            background: var(--bg-primary);
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .toggle-btn.active {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            color: var(--bg-primary);
            border-color: var(--accent-primary);
        }

        .toggle-btn:hover:not(.active) {
            border-color: var(--accent-primary);
            color: var(--text-primary);
        }

        .alert-item {
            display: flex;
            gap: 0.75rem;
            padding: 0.875rem 0;
            border-bottom: 1px solid var(--border-color);
            align-items: flex-start;
        }

        .alert-item:last-child { border-bottom: none; }

        .alert-icon {
            width: 32px; height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
            flex-shrink: 0;
            margin-top: 2px;
        }

        .alert-icon.critical { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); }
        .alert-icon.warning { background: rgba(245, 158, 11, 0.15); color: var(--accent-primary); }
        .alert-icon.info { background: rgba(16, 185, 129, 0.15); color: var(--accent-green); }

        .alert-content { flex: 1; }

        .alert-time {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-bottom: 0.15rem;
        }

        .alert-title {
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 0.15rem;
        }

        .alert-message {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        .map-container {
            height: 320px;
            border-radius: 12px;
            overflow: hidden;
            border: 2px solid var(--border-color);
            background: #1e293b;
        }

        .section-title-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .section-title-row .card-header {
            margin-bottom: 0;
        }

        .chart-wrapper {
            height: 300px;
            position: relative;
        }

        .no-data {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--text-secondary);
            font-size: 1rem;
        }

        @media (max-width: 768px) {
            .hero h1 { font-size: 1.5rem; }
            .cards-grid { grid-template-columns: 1fr; }
            .batch-results-grid { grid-template-columns: 1fr; }
            .image-comparison { grid-template-columns: 1fr !important; }
            .hardware-info { gap: 0.5rem; }
            .mode-btn { padding: 0.75rem 1rem; font-size: 0.85rem; }
            .section-title-row { flex-direction: column; align-items: flex-start; }
        }
    </style>
</head>
<body>
    <div class="honeycomb-bg"></div>
    <div class="glow-line"></div>

    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-bee-container">
            <div class="loading-bee lbee1">🐝</div>
            <div class="loading-bee lbee2">🐝</div>
            <div class="loading-bee lbee3">🐝</div>
            <div class="loading-bee lbee4">🐝</div>
            <div class="loading-bee lbee5">🐝</div>
            <div class="loading-bee lbee6">🐝</div>
            <div class="loading-bee lbee7">🐝</div>
        </div>
        <div class="spinner"></div>
        <div class="loading-text" id="loadingText">Initializing Jetson...</div>
        <div class="loading-subtext" id="loadingSubtext">Starting camera and audio systems</div>
    </div>

    <nav class="navbar">
        <div class="logo">
            <div class="logo-icon">🐝</div>
            <div class="logo-text">
                <h2>BeeDetect AI</h2>
                <p>Jetson Nano Edition</p>
            </div>
        </div>
        <div class="nav-status">
            <div class="jetson-badge">
                <i class="fas fa-microchip"></i>
                <span>Jetson Nano</span>
            </div>
            <div class="aws-badge">
                <i class="fas fa-cloud-upload-alt"></i>
                <span>AWS S3</span>
            </div>
            <div class="status-badge">
                <span class="status-dot"></span>
                <span id="systemStatus">Online</span>
            </div>
        </div>
    </nav>

    <section class="hero">
        <h1>BeeDetect AI - Jetson Nano</h1>
        <p>Real-time multimodal honeybee detection using camera, microphone, and video processing on NVIDIA Jetson Nano.</p>
        <div class="hardware-info">
            <div class="hw-item active" id="cameraStatus">
                <i class="fas fa-camera"></i>
                <span>Camera Ready</span>
            </div>
            <div class="hw-item active" id="micStatus">
                <i class="fas fa-microphone"></i>
                <span>Mic Ready</span>
            </div>
            <div class="hw-item">
                <i class="fas fa-video"></i>
                <span>Video Ready</span>
            </div>
            <div class="hw-item">
                <i class="fas fa-wifi"></i>
                <span id="connectionStatus">Connected</span>
            </div>
        </div>
    </section>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-icon camera"><i class="fas fa-camera"></i></div>
            <div class="stat-info">
                <h3 id="captureCount">0</h3>
                <p>Captures</p>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon mic"><i class="fas fa-microphone"></i></div>
            <div class="stat-info">
                <h3 id="audioCount">0</h3>
                <p>Audio Samples</p>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon video"><i class="fas fa-video"></i></div>
            <div class="stat-info">
                <h3 id="videoCount">0</h3>
                <p>Videos Processed</p>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon detections"><i class="fas fa-bug"></i></div>
            <div class="stat-info">
                <h3 id="totalDetections">0</h3>
                <p>Total Detections</p>
            </div>
        </div>
    </div>

    <div class="mode-toggle">
        <button class="mode-btn active" onclick="switchMode('cv')">
            <i class="fas fa-camera"></i>
            <span>Camera</span>
        </button>
        <button class="mode-btn" onclick="switchMode('video')">
            <i class="fas fa-film"></i>
            <span>Video</span>
        </button>
        <button class="mode-btn" onclick="switchMode('audio')">
            <i class="fas fa-microphone"></i>
            <span>Audio</span>
        </button>
        <button class="mode-btn" onclick="switchMode('dashboard')">
            <i class="fas fa-chart-line"></i>
            <span>Dashboard</span>
        </button>
    </div>

    <main class="container">
        <!-- COMPUTER VISION SECTION -->
        <section id="cv-section" class="content-section active">
            <div class="camera-section">
                <div class="card-header">
                    <i class="fas fa-video"></i>
                    <h3>Live Camera Feed</h3>
                </div>
                <div class="camera-feed-wrapper">
                    <img src="/video_feed" id="cameraFeed" alt="Live Camera Feed">
                    <div class="camera-overlay">
                        <span class="status-dot"></span>
                        <span id="liveFeedLabel">LIVE</span>
                    </div>
                    <div id="liveCountBadge" style="display:none; position:absolute; top:0.75rem; right:0.75rem; background: rgba(245, 158, 11, 0.95); color:#0f172a; font-weight:700; padding:0.4rem 0.75rem; border-radius:999px; box-shadow:0 2px 8px rgba(0,0,0,0.35);">
                        <i class="fas fa-bug"></i>
                        <span id="liveCount">0</span> bees
                    </div>
                </div>
                <div class="camera-controls">
                    <button class="camera-btn primary" id="captureBtn" onclick="captureAndDetect()">
                        <i class="fas fa-camera"></i>
                        Capture & Detect
                    </button>
                    <button class="camera-btn secondary" id="burstBtn" onclick="captureBurst()">
                        <i class="fas fa-images"></i>
                        Burst Capture (5)
                    </button>
                    <button class="camera-btn primary" id="liveDetectBtn" onclick="toggleLiveDetection()">
                        <i class="fas fa-bolt"></i>
                        Start Live Detection
                    </button>
                </div>
            </div>

            <!-- IMAGE UPLOAD SECTION -->
            <div class="card" style="margin-bottom: 1.5rem;">
                <div class="card-header">
                    <i class="fas fa-upload"></i>
                    <h3>Upload Image for Detection</h3>
                </div>
                <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                    Upload an image file to detect and count bees. Both original and annotated versions are stored in AWS S3.
                </p>

                <div class="upload-area" id="imageUploadArea" onclick="document.getElementById('imageFileInput').click()">
                    <i class="fas fa-cloud-upload-alt"></i>
                    <p><strong>Click to upload image</strong> or drag and drop</p>
                    <p style="font-size: 0.8rem;">Supports JPG, PNG, WEBP (Max 10MB)</p>
                    <input type="file" id="imageFileInput" accept="image/*" onchange="handleImageSelect(event)">
                </div>

                <div class="file-list" id="imageFileList"></div>

                <div class="camera-controls" style="margin-top: 1rem;">
                    <button class="camera-btn primary" id="uploadImageBtn" onclick="uploadAndDetectImage()" disabled>
                        <i class="fas fa-brain"></i>
                        Detect Bees & Upload to Cloud
                    </button>
                </div>
            </div>

            <!-- IMAGE UPLOAD RESULTS -->
            <div id="imageUploadResults" style="display: none; margin-bottom: 1.5rem;">
                <div class="card">
                    <div class="card-header" style="justify-content: center;">
                        <i class="fas fa-poll"></i>
                        <h3>Upload Detection Result</h3>
                    </div>
                    <div class="detection-stats" id="imageUploadStats">
                        <div class="detection-stat">
                            <h4 id="uploadBeeCount">0</h4>
                            <p>Bees Detected</p>
                        </div>
                        <div class="detection-stat" style="background: linear-gradient(135deg, #3b82f6, #1d4ed8);">
                            <h4 id="uploadInferenceTime">0s</h4>
                            <p>Inference Time</p>
                        </div>
                    </div>
                    <div class="image-comparison" id="imageUploadComparison"></div>

                    <!-- S3 Info -->
                    <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid var(--border-color);">
                        <div class="metric-row">
                            <span class="metric-label"><i class="fas fa-cloud" style="color: var(--accent-primary); margin-right: 0.5rem;"></i>Original S3 URL</span>
                            <span class="metric-value" id="uploadS3Original" style="font-size: 0.75rem; word-break: break-all; max-width: 60%; text-align: right;">-</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label"><i class="fas fa-cloud" style="color: var(--accent-green); margin-right: 0.5rem;"></i>Annotated S3 URL</span>
                            <span class="metric-value" id="uploadS3Annotated" style="font-size: 0.75rem; word-break: break-all; max-width: 60%; text-align: right;">-</span>
                        </div>
                    </div>
                </div>
            </div>

            <div id="cvResults" style="display: none;">
                <div class="card" style="margin-bottom: 1.5rem;">
                    <div class="card-header" style="justify-content: center;">
                        <i class="fas fa-poll"></i>
                        <h3>Detection Result</h3>
                    </div>
                    <div class="detection-stats" id="cvStats">
                        <div class="detection-stat">
                            <h4 id="cvBeeCount">0</h4>
                            <p>Bees Detected</p>
                        </div>
                        <div class="detection-stat" style="background: linear-gradient(135deg, #3b82f6, #1d4ed8);">
                            <h4 id="cvInferenceTime">0s</h4>
                            <p>Inference Time</p>
                        </div>
                    </div>
                    <div class="image-comparison" id="cvComparison"></div>
                </div>
            </div>

            <!-- CV Detection History Chart -->
            <div class="card" style="margin-bottom: 1.5rem;">
                <div class="card-header">
                    <i class="fas fa-chart-bar" style="color: var(--accent-primary);"></i>
                    <h3>Detection History (from AWS)</h3>
                </div>
                <div class="chart-wrapper">
                    <canvas id="cvHistoryChart"></canvas>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <i class="fas fa-history"></i>
                    <h3>Recent Captures</h3>
                </div>
                <div class="capture-grid" id="recentCaptures">
                    <p style="color: var(--text-secondary); text-align: center; grid-column: 1 / -1;">
                        No captures yet. Use the camera above to capture images.
                    </p>
                </div>
            </div>
        </section>

        <!-- VIDEO PROCESSING SECTION -->
        <section id="video-section" class="content-section">
            <div class="cards-grid">
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-upload"></i>
                        <h3>Upload Video for Processing</h3>
                    </div>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                        Upload a video file to process bee detection frame-by-frame. The system analyzes every Nth frame and aggregates results.
                    </p>

                    <div class="upload-area" id="videoUploadArea" onclick="document.getElementById('videoFileInput').click()">
                        <i class="fas fa-cloud-upload-alt"></i>
                        <p><strong>Click to upload</strong> or drag and drop</p>
                        <p style="font-size: 0.8rem;">Supports MP4, AVI, MOV (Max 100MB)</p>
                        <input type="file" id="videoFileInput" accept="video/*" onchange="handleVideoSelect(event)">
                    </div>

                    <div class="file-list" id="videoFileList"></div>

                    <div class="camera-controls" style="margin-top: 1rem;">
                        <button class="camera-btn primary" id="processVideoBtn" onclick="processVideo()" disabled>
                            <i class="fas fa-cogs"></i>
                            Process Video
                        </button>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-info-circle"></i>
                        <h3>Video Processing Info</h3>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Frame Interval</span>
                        <span class="metric-value">Every 5th frame</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Output Format</span>
                        <span class="metric-value">MP4 with annotations</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Cloud Storage</span>
                        <span class="metric-value" style="color: var(--accent-green);">AWS S3</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Inference Model</span>
                        <span class="metric-value">Roboflow Workflow</span>
                    </div>
                </div>
            </div>

            <div class="video-results" id="videoResults">
                <div class="card" style="margin-bottom: 1.5rem;">
                    <div class="card-header" style="justify-content: center;">
                        <i class="fas fa-chart-line"></i>
                        <h3>Video Analysis Summary</h3>
                    </div>

                    <div class="video-stat-grid" id="videoStatGrid">
                        <div class="video-stat">
                            <h4 id="vidTotalFrames">0</h4>
                            <p>Total Frames</p>
                        </div>
                        <div class="video-stat">
                            <h4 id="vidProcessedFrames">0</h4>
                            <p>Processed</p>
                        </div>
                        <div class="video-stat">
                            <h4 id="vidDuration">0s</h4>
                            <p>Duration</p>
                        </div>
                        <div class="video-stat">
                            <h4 id="vidAvgBees">0</h4>
                            <p>Avg Bees/Frame</p>
                        </div>
                        <div class="video-stat">
                            <h4 id="vidMaxBees">0</h4>
                            <p>Peak Count</p>
                        </div>
                        <div class="video-stat">
                            <h4 id="vidTotalBees">0</h4>
                            <p>Total Detections</p>
                        </div>
                    </div>

                    <div class="timeline-chart-container">
                        <canvas id="videoTimelineChart"></canvas>
                    </div>
                </div>

                <div class="card" id="videoFramesCard" style="display: none;">
                    <div class="card-header">
                        <i class="fas fa-images"></i>
                        <h3>Processed Frames (stored on AWS S3)</h3>
                    </div>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem; font-size: 0.9rem;">
                        5 frames sampled evenly across the video, each annotated by Roboflow. Only the annotated outputs are uploaded to S3; the raw video is discarded.
                    </p>
                    <div id="videoFramesContainer"></div>
                </div>
            </div>
        </section>

        <!-- AUDIO ANALYSIS SECTION -->
        <section id="audio-section" class="content-section">
            <div class="cards-grid">
                <!-- Audio File Upload -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-file-audio"></i>
                        <h3>Upload Test Audio</h3>
                    </div>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                        Upload a test audio file (WAV/MP3) to estimate bee population range using BeeCNN mel-spectrogram analysis.
                    </p>

                    <div class="upload-area" id="audioUploadArea" onclick="document.getElementById('audioFileInput').click()">
                        <i class="fas fa-music"></i>
                        <p><strong>Click to upload audio</strong></p>
                        <p style="font-size: 0.8rem;">WAV, MP3, OGG supported</p>
                        <input type="file" id="audioFileInput" accept="audio/*" onchange="handleAudioFileSelect(event)">
                    </div>

                    <div class="file-list" id="audioFileList"></div>

                    <div class="camera-controls" style="margin-top: 1rem;">
                        <button class="camera-btn primary" id="uploadAudioBtn" onclick="uploadAndAnalyzeAudio()" disabled>
                            <i class="fas fa-brain"></i>
                            Analyze & Upload to Cloud
                        </button>
                    </div>
                </div>

                <!-- Microphone Capture -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-microphone"></i>
                        <h3>Microphone Capture</h3>
                    </div>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                        Record hive audio using the connected microphone. The BeeCNN model analyzes mel-spectrogram patterns for population estimation.
                    </p>

                    <div class="audio-visualizer" id="audioVisualizer">
                        <div class="visualizer-bars" id="visualizerBars"></div>
                    </div>

                    <div class="audio-status" id="audioStatus">Ready to record</div>

                    <div class="camera-controls">
                        <button class="camera-btn primary" id="recordBtn" onclick="toggleRecording()">
                            <i class="fas fa-circle"></i>
                            Record Audio
                        </button>
                    </div>
                </div>

                <!-- Audio Model Info -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-info-circle"></i>
                        <h3>Audio Model Information</h3>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Model Architecture</span>
                        <span class="metric-value">BeeCNN (3-Layer CNN)</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Feature Input</span>
                        <span class="metric-value">128-bin Mel-Spectrogram</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Sample Rate</span>
                        <span class="metric-value">16 kHz</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Recording Duration</span>
                        <span class="metric-value">up to 20 s (user-stoppable)</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Estimation Levels</span>
                        <span class="metric-value">6 (0 to 1000+ bees)</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Cloud Path</span>
                        <span class="metric-value" style="color: var(--accent-blue);">audio/wav/</span>
                    </div>
                </div>
            </div>

            <!-- Audio Results -->
            <div class="audio-results" id="audioResults">
                <div class="audio-classification" id="audioClassification">
                    <i class="fas fa-chart-bar" style="font-size: 2rem; margin-bottom: 0.5rem;"></i>
                    <h3 id="classificationResult">Normal Hive</h3>
                    <p id="classificationSubtext">Bee Activity Level</p>
                </div>

                <div class="card" style="margin-bottom: 1.5rem;">
                    <div class="card-header">
                        <i class="fas fa-percentage"></i>
                        <h3>Confidence Scores</h3>
                    </div>
                    <div class="prob-bars">
                        <div class="prob-bar-row">
                            <span class="prob-label">Low</span>
                            <div class="prob-bar-bg">
                                <div class="prob-bar-fill" id="probBarLow" style="width: 0%; background: linear-gradient(90deg, #10b981, #059669);"></div>
                            </div>
                            <span class="prob-value" id="probValLow">0%</span>
                        </div>
                        <div class="prob-bar-row">
                            <span class="prob-label">Medium</span>
                            <div class="prob-bar-bg">
                                <div class="prob-bar-fill" id="probBarMed" style="width: 0%; background: linear-gradient(90deg, #f59e0b, #d97706);"></div>
                            </div>
                            <span class="prob-value" id="probValMed">0%</span>
                        </div>
                        <div class="prob-bar-row">
                            <span class="prob-label">High</span>
                            <div class="prob-bar-bg">
                                <div class="prob-bar-fill" id="probBarHigh" style="width: 0%; background: linear-gradient(90deg, #ef4444, #dc2626);"></div>
                            </div>
                            <span class="prob-value" id="probValHigh">0%</span>
                        </div>
                    </div>
                </div>

                <div class="audio-metrics">
                    <div class="audio-metric" style="background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));">
                        <h4 id="estimatedCount" style="color: #0f172a;">~125</h4>
                        <p style="color: rgba(15, 23, 42, 0.8);">Estimated Bee Count</p>
                    </div>
                    <div class="audio-metric" id="levelMetric">
                        <h4 id="levelValue">Medium</h4>
                        <p>Activity Level</p>
                    </div>
                    <div class="audio-metric" id="stressMetric">
                        <h4 id="stressValue">Low</h4>
                        <p>Stress Level</p>
                    </div>
                    <div class="audio-metric">
                        <h4 id="swarmingValue">12%</h4>
                        <p>Swarming Probability</p>
                    </div>
                    <div class="audio-metric">
                        <h4 id="activityValue">78%</h4>
                        <p>Activity Intensity</p>
                    </div>
                    <div class="audio-metric" id="anomalyMetric">
                        <h4 id="anomalyValue">No</h4>
                        <p>Anomaly Detected</p>
                    </div>
                </div>

                <div class="card" style="margin-top: 1.5rem;">
                    <div class="card-header">
                        <i class="fas fa-wave-square"></i>
                        <h3>Frequency Spectrum Analysis</h3>
                    </div>
                    <div class="frequency-chart-container">
                        <canvas id="frequencyChart"></canvas>
                    </div>
                </div>

            </div>

            <!-- Audio Historical Trends -->
            <div class="card" style="margin-top: 1.5rem;">
                <div class="section-title-row">
                    <div class="card-header">
                        <i class="fas fa-calendar-alt" style="color: var(--accent-primary);"></i>
                        <h3>Historical Population Trends</h3>
                    </div>
                    <div class="toggle-group">
                        <button class="toggle-btn active" onclick="switchAudioRange('daily', this)">Daily</button>
                        <button class="toggle-btn" onclick="switchAudioRange('weekly', this)">Weekly</button>
                        <button class="toggle-btn" onclick="switchAudioRange('monthly', this)">Monthly</button>
                    </div>
                </div>
                <div class="chart-wrapper">
                    <canvas id="audioTrendChart"></canvas>
                </div>
            </div>
        </section>

        <!-- DASHBOARD SECTION -->
        <section id="dashboard-section" class="content-section">
            <div class="cards-grid">
                <!-- CV History -->
                <div class="card" style="grid-column: 1 / -1;">
                    <div class="card-header">
                        <i class="fas fa-history" style="color: var(--accent-primary);"></i>
                        <h3>Detection History (from AWS)</h3>
                    </div>
                    <div class="chart-wrapper">
                        <canvas id="dashCVHistoryChart"></canvas>
                    </div>
                </div>

                <!-- Audio Trends -->
                <div class="card" style="grid-column: 1 / -1;">
                    <div class="section-title-row">
                        <div class="card-header">
                            <i class="fas fa-chart-area" style="color: var(--accent-purple);"></i>
                            <h3>Historical Population Trends</h3>
                        </div>
                        <div class="toggle-group">
                            <button class="toggle-btn active" onclick="switchDashAudioRange('daily', this)">Daily</button>
                            <button class="toggle-btn" onclick="switchDashAudioRange('weekly', this)">Weekly</button>
                            <button class="toggle-btn" onclick="switchDashAudioRange('monthly', this)">Monthly</button>
                        </div>
                    </div>
                    <div class="chart-wrapper">
                        <canvas id="dashAudioTrendChart"></canvas>
                    </div>
                </div>

                <!-- Map -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-map-marker-alt" style="color: var(--accent-green);"></i>
                        <h3>Test Locations</h3>
                    </div>
                    <div class="map-container" id="testMap"></div>
                    <p style="color: var(--text-secondary); font-size: 0.8rem; margin-top: 0.75rem; text-align: center;">
                        <i class="fas fa-info-circle"></i> Field tests conducted in Topi and Ghazi, KPK
                    </p>
                </div>

                <!-- Alerts -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-bell" style="color: var(--accent-red);"></i>
                        <h3>Recent Activity and Alerts</h3>
                    </div>
                    <div id="alertsList" style="max-height: 320px; overflow-y: auto;">
                        <p style="color: var(--text-secondary); text-align: center;">Loading alerts...</p>
                    </div>
                </div>
            </div>
        </section>
    </main>

    <footer class="footer">
        <p> BeeDetect AI | Jetson Nano Edition | Multimodal Honeybee Detection System</p>
    </footer>

    <script>
        // State
        let captureCount = 0;
        let audioCount = 0;
        let videoCount = 0;
        let totalDetections = 0;
        let isRecording = false;
        let recentCaptures = [];
        let selectedAudioFile = null;
        let selectedVideoFile = null;
        let selectedImageFile = null;

        // Mode switching
        function switchMode(mode) {
            document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.content-section').forEach(section => section.classList.remove('active'));
            event.target.closest('.mode-btn').classList.add('active');
            document.getElementById(mode + '-section').classList.add('active');

            if (mode === 'dashboard') {
                loadCVHistory('dashCVHistoryChart');
                loadAudioTrends('daily', 'dashAudioTrendChart');
                initMap();
                loadAlerts();
            }
            if (mode === 'cv') {
                loadCVHistory('cvHistoryChart');
            }
            if (mode === 'audio') {
                loadAudioTrends('daily', 'audioTrendChart');
            }
        }

        // Loading overlay
        function showLoading(text, subtext) {
            const overlay = document.getElementById('loadingOverlay');
            document.getElementById('loadingText').textContent = text || 'Processing...';
            document.getElementById('loadingSubtext').textContent = subtext || '';
            overlay.classList.add('show');
        }

        function hideLoading() {
            document.getElementById('loadingOverlay').classList.remove('show');
        }

        // ===== LIVE WEBRTC DETECTION =====
        let liveDetectionActive = false;
        let liveCountInterval = null;

        function toggleLiveDetection() {
            if (liveDetectionActive) {
                stopLiveDetection();
            } else {
                startLiveDetection();
            }
        }

        function startLiveDetection() {
            const btn = document.getElementById('liveDetectBtn');
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
            fetch('/start-live-detection', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    btn.disabled = false;
                    if (data.error) {
                        alert('Error: ' + data.error);
                        btn.innerHTML = '<i class="fas fa-bolt"></i> Start Live Detection';
                        return;
                    }
                    liveDetectionActive = true;
                    // Swap the live feed to the annotated stream
                    const feed = document.getElementById('cameraFeed');
                    feed.src = '/live-feed?ts=' + Date.now();
                    document.getElementById('liveFeedLabel').textContent = 'LIVE • DETECTING';
                    document.getElementById('liveCountBadge').style.display = 'block';
                    btn.innerHTML = '<i class="fas fa-stop"></i> Stop Live Detection';
                    btn.classList.remove('primary');
                    btn.classList.add('secondary');
                    // Poll the count ~2x/sec
                    liveCountInterval = setInterval(pollLiveCount, 500);
                })
                .catch(err => {
                    btn.disabled = false;
                    btn.innerHTML = '<i class="fas fa-bolt"></i> Start Live Detection';
                    alert('Failed to start live detection: ' + err.message);
                });
        }

        function stopLiveDetection() {
            const btn = document.getElementById('liveDetectBtn');
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Stopping...';
            fetch('/stop-live-detection', { method: 'POST' })
                .then(r => r.json())
                .then(() => {
                    liveDetectionActive = false;
                    if (liveCountInterval) { clearInterval(liveCountInterval); liveCountInterval = null; }
                    // Swap back to the local camera MJPEG feed
                    const feed = document.getElementById('cameraFeed');
                    feed.src = '/video_feed?ts=' + Date.now();
                    document.getElementById('liveFeedLabel').textContent = 'LIVE';
                    document.getElementById('liveCountBadge').style.display = 'none';
                    btn.disabled = false;
                    btn.innerHTML = '<i class="fas fa-bolt"></i> Start Live Detection';
                    btn.classList.remove('secondary');
                    btn.classList.add('primary');
                    loadDashboardStats();
                    loadAlerts();
                })
                .catch(err => {
                    btn.disabled = false;
                    btn.innerHTML = '<i class="fas fa-bolt"></i> Start Live Detection';
                    alert('Failed to stop live detection: ' + err.message);
                });
        }

        function pollLiveCount() {
            fetch('/live-count')
                .then(r => r.json())
                .then(data => {
                    if (!data.active) {
                        // session ended server-side — tidy up the UI
                        if (liveDetectionActive) stopLiveDetection();
                        return;
                    }
                    document.getElementById('liveCount').textContent = data.count_objects;
                })
                .catch(() => {/* ignore transient errors */});
        }

        // Capture and detect from camera
        function captureAndDetect() {
            showLoading('Capturing & Detecting...', 'Running inference on captured frame');
            fetch('/capture-and-detect', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    hideLoading();
                    if (data.error) { alert('Error: ' + data.error); return; }
                    displayCVResult(data);
                    addRecentCapture(data);
                    loadDashboardStats();
                    loadCVHistory('cvHistoryChart');
                })
                .catch(err => { hideLoading(); alert('Capture failed: ' + err.message); });
        }

        function captureBurst() {
            showLoading('Burst Capture...', 'Capturing 5 frames and running detection');
            fetch('/capture-burst', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    hideLoading();
                    if (data.error) { alert('Error: ' + data.error); return; }
                    if (data.results && data.results.length > 0) {
                        displayCVResult(data.results[data.results.length - 1]);
                        data.results.forEach(r => addRecentCapture(r));
                        loadDashboardStats();
                        loadCVHistory('cvHistoryChart');
                    }
                })
                .catch(err => { hideLoading(); alert('Burst capture failed: ' + err.message); });
        }

        function displayCVResult(data) {
            document.getElementById('cvResults').style.display = 'block';
            document.getElementById('cvBeeCount').textContent = data.bee_count || 0;
            document.getElementById('cvInferenceTime').textContent = (data.inference_time || 0) + 's';
            const comparison = document.getElementById('cvComparison');
            comparison.innerHTML = `
                <div class="image-box">
                    <h5><i class="fas fa-camera"></i> Captured</h5>
                    <div class="image-wrapper">
                        <span class="image-label">CAPTURE</span>
                        <img src="data:image/jpeg;base64,${data.captured_image}" alt="Captured">
                    </div>
                </div>
                <div class="image-box">
                    <h5><i class="fas fa-magic"></i> Detected</h5>
                    <div class="image-wrapper">
                        <span class="image-label">OUTPUT</span>
                        <img src="data:image/jpeg;base64,${data.output_image}" alt="Detection Result">
                    </div>
                </div>
            `;
        }

        function addRecentCapture(data) {
            recentCaptures.unshift(data);
            if (recentCaptures.length > 6) recentCaptures.pop();
            const grid = document.getElementById('recentCaptures');
            grid.innerHTML = recentCaptures.map(c => `
                <div class="capture-item">
                    <img src="data:image/jpeg;base64,${c.output_image || c.captured_image || c.original_image}" alt="Capture">
                    <div class="capture-info">
                        <i class="fas fa-bug" style="color: var(--accent-primary);"></i>
                        ${c.bee_count || 0} bees
                    </div>
                </div>
            `).join('');
        }

        // Load persisted dashboard counters from AWS S3 via backend
        function loadDashboardStats() {
            return fetch('/api/dashboard-stats')
                .then(r => r.json())
                .then(stats => {
                    captureCount = stats.captures || 0;
                    audioCount = stats.audio_samples || 0;
                    videoCount = stats.videos_processed || 0;
                    totalDetections = stats.total_detections || 0;
                    document.getElementById('captureCount').textContent = captureCount;
                    document.getElementById('audioCount').textContent = audioCount;
                    document.getElementById('videoCount').textContent = videoCount;
                    document.getElementById('totalDetections').textContent = totalDetections;
                })
                .catch(err => console.error('Failed to load dashboard stats:', err));
        }

        document.addEventListener('DOMContentLoaded', loadDashboardStats);

        // ===== IMAGE UPLOAD =====
        function handleImageSelect(event) {
            const file = event.target.files[0];
            if (!file) return;
            selectedImageFile = file;
            const list = document.getElementById('imageFileList');
            list.innerHTML = `
                <div class="file-item">
                    <span><i class="fas fa-image" style="color: var(--accent-primary); margin-right: 0.5rem;"></i>${file.name}</span>
                    <i class="fas fa-times" onclick="clearImageFile()"></i>
                </div>
            `;
            document.getElementById('uploadImageBtn').disabled = false;
        }

        function clearImageFile() {
            selectedImageFile = null;
            document.getElementById('imageFileList').innerHTML = '';
            document.getElementById('uploadImageBtn').disabled = true;
            document.getElementById('imageFileInput').value = '';
        }

        function uploadAndDetectImage() {
            if (!selectedImageFile) return;
            showLoading('Processing Image...', 'Running bee detection and uploading to AWS S3');

            const formData = new FormData();
            formData.append('image', selectedImageFile);

            fetch('/upload-image', { method: 'POST', body: formData })
                .then(r => r.json())
                .then(data => {
                    hideLoading();
                    if (data.error) { alert('Error: ' + data.error); return; }
                    displayImageUploadResult(data);
                    addRecentCapture(data);
                    loadDashboardStats();
                    loadCVHistory('cvHistoryChart');
                })
                .catch(err => {
                    hideLoading();
                    alert('Image upload failed: ' + err.message);
                });
        }

        function displayImageUploadResult(data) {
            document.getElementById('imageUploadResults').style.display = 'block';
            document.getElementById('uploadBeeCount').textContent = data.bee_count || 0;
            document.getElementById('uploadInferenceTime').textContent = (data.inference_time || 0) + 's';

            const comparison = document.getElementById('imageUploadComparison');
            comparison.innerHTML = `
                <div class="image-box">
                    <h5><i class="fas fa-image"></i> Original</h5>
                    <div class="image-wrapper">
                        <span class="image-label">ORIGINAL</span>
                        <img src="data:image/jpeg;base64,${data.original_image}" alt="Original">
                    </div>
                </div>
                <div class="image-box">
                    <h5><i class="fas fa-magic"></i> Detected</h5>
                    <div class="image-wrapper">
                        <span class="image-label">OUTPUT</span>
                        <img src="data:image/jpeg;base64,${data.output_image}" alt="Detection Result">
                    </div>
                </div>
            `;

            document.getElementById('uploadS3Original').textContent = data.s3_original_url || 'Upload failed';
            document.getElementById('uploadS3Original').style.color = data.s3_original_url ? 'var(--accent-green)' : 'var(--accent-red)';

            document.getElementById('uploadS3Annotated').textContent = data.s3_annotated_url || 'Upload failed';
            document.getElementById('uploadS3Annotated').style.color = data.s3_annotated_url ? 'var(--accent-green)' : 'var(--accent-red)';
        }

        // ===== AUDIO FILE UPLOAD =====
        function handleAudioFileSelect(event) {
            const file = event.target.files[0];
            if (!file) return;
            selectedAudioFile = file;
            const list = document.getElementById('audioFileList');
            list.innerHTML = `
                <div class="file-item">
                    <span><i class="fas fa-music" style="color: var(--accent-purple); margin-right: 0.5rem;"></i>${file.name}</span>
                    <i class="fas fa-times" onclick="clearAudioFile()"></i>
                </div>
            `;
            document.getElementById('uploadAudioBtn').disabled = false;
        }

        function clearAudioFile() {
            selectedAudioFile = null;
            document.getElementById('audioFileList').innerHTML = '';
            document.getElementById('uploadAudioBtn').disabled = true;
            document.getElementById('audioFileInput').value = '';
        }

        function uploadAndAnalyzeAudio() {
            if (!selectedAudioFile) return;
            showLoading('Analyzing Audio...', 'Running BeeCNN inference');

            const formData = new FormData();
            formData.append('audio', selectedAudioFile);

            fetch('/upload-audio', { method: 'POST', body: formData })
                .then(r => r.json())
                .then(data => {
                    hideLoading();
                    if (data.error) { alert('Error: ' + data.error); return; }
                    displayAudioResults(data);
                    loadDashboardStats();
                    loadAudioTrends('daily', 'audioTrendChart');
                })
                .catch(err => {
                    hideLoading();
                    alert('Audio upload failed: ' + err.message);
                });
        }

        // ===== VIDEO UPLOAD =====
        function handleVideoSelect(event) {
            const file = event.target.files[0];
            if (!file) return;
            selectedVideoFile = file;
            const list = document.getElementById('videoFileList');
            list.innerHTML = `
                <div class="file-item">
                    <span><i class="fas fa-video" style="color: var(--accent-green); margin-right: 0.5rem;"></i>${file.name}</span>
                    <i class="fas fa-times" onclick="clearVideoFile()"></i>
                </div>
            `;
            document.getElementById('processVideoBtn').disabled = false;
        }

        function clearVideoFile() {
            selectedVideoFile = null;
            document.getElementById('videoFileList').innerHTML = '';
            document.getElementById('processVideoBtn').disabled = true;
            document.getElementById('videoFileInput').value = '';
        }

        function processVideo() {
            if (!selectedVideoFile) return;
            showLoading('Processing Video...', 'Extracting frames and running bee detection. This may take a while.');

            const formData = new FormData();
            formData.append('video', selectedVideoFile);

            fetch('/upload-video', { method: 'POST', body: formData })
                .then(r => r.json())
                .then(data => {
                    hideLoading();
                    if (data.error) { alert('Error: ' + data.error); return; }
                    displayVideoResults(data);
                    loadDashboardStats();
                })
                .catch(err => {
                    hideLoading();
                    alert('Video processing failed: ' + err.message);
                });
        }

        function displayVideoResults(data) {
            const results = document.getElementById('videoResults');
            results.classList.add('show');

            const s = data.summary;
            document.getElementById('vidTotalFrames').textContent = s.total_frames;
            document.getElementById('vidProcessedFrames').textContent = s.processed_frames;
            document.getElementById('vidDuration').textContent = s.duration + 's';
            document.getElementById('vidAvgBees').textContent = s.average_bees_per_frame;
            document.getElementById('vidMaxBees').textContent = s.max_bees_in_single_frame;
            document.getElementById('vidTotalBees').textContent = s.total_bees_detected;

            // Bar chart across the 5 sampled frames (replaces the line timeline)
            const ctx = document.getElementById('videoTimelineChart');
            // Use a prefixed window property to avoid collision with the canvas
            // element's DOM id (browsers expose named elements on `window`, and
            // an HTMLCanvasElement has no `.destroy()` method).
            if (window._videoChartInstance && typeof window._videoChartInstance.destroy === 'function') {
                window._videoChartInstance.destroy();
            }
            window._videoChartInstance = new Chart(ctx.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: s.frame_results.map(f => 'Frame ' + f.position + ' (' + f.timestamp + 's)'),
                    datasets: [{
                        label: 'Bees Detected',
                        data: s.frame_results.map(f => f.bee_count),
                        backgroundColor: 'rgba(245, 158, 11, 0.85)',
                        borderColor: '#f59e0b',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: '#334155' },
                            ticks: { color: '#94a3b8' },
                            title: { display: true, text: 'Bee Count', color: '#94a3b8' }
                        },
                        x: {
                            grid: { color: '#334155' },
                            ticks: { color: '#94a3b8' }
                        }
                    },
                    plugins: {
                        legend: { display: false },
                        title: {
                            display: true,
                            text: 'Bee Count Across 5 Sampled Frames',
                            color: '#f8fafc',
                            font: { size: 14 }
                        }
                    }
                }
            });

            // Render each sampled frame as its own Original / Detected pair,
            // matching the layout of the image-upload results card.
            const card = document.getElementById('videoFramesCard');
            const container = document.getElementById('videoFramesContainer');
            if (s.frame_results && s.frame_results.length) {
                card.style.display = 'block';
                container.innerHTML = s.frame_results.map(f => {
                    const rawSrc = f.raw_base64 ? 'data:image/jpeg;base64,' + f.raw_base64 : '';
                    const annSrc = f.annotated_base64
                        ? 'data:image/jpeg;base64,' + f.annotated_base64
                        : (f.s3_url || '');
                    const s3Link = f.s3_url
                        ? `<a href="${f.s3_url}" target="_blank" style="color: var(--accent-primary); font-size: 0.75rem; word-break: break-all;">${f.s3_url}</a>`
                        : '<span style="color: var(--accent-red); font-size: 0.75rem;">Upload failed</span>';
                    const originalHTML = rawSrc
                        ? `<img src="${rawSrc}" alt="Frame ${f.position} original">`
                        : '<div style="padding: 2rem; text-align: center; color: var(--accent-red);">Frame unavailable</div>';
                    const detectedHTML = annSrc
                        ? `<img src="${annSrc}" alt="Frame ${f.position} detected">`
                        : '<div style="padding: 2rem; text-align: center; color: var(--accent-red);">No detection output</div>';
                    return `
                        <div style="margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid var(--border-color);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.5rem;">
                                <h4 style="color: var(--text-primary); margin: 0;">
                                    <i class="fas fa-film" style="color: var(--accent-primary); margin-right: 0.5rem;"></i>
                                    Frame ${f.position}
                                    <span style="color: var(--text-secondary); font-weight: 400; font-size: 0.85rem; margin-left: 0.5rem;">
                                        t = ${f.timestamp}s  •  video frame ${f.frame_index}
                                    </span>
                                </h4>
                                <div style="display: flex; gap: 1rem; align-items: center;">
                                    <span style="background: rgba(245, 158, 11, 0.15); color: var(--accent-primary); padding: 0.4rem 0.9rem; border-radius: 20px; font-weight: 700;">
                                        <i class="fas fa-bug"></i> ${f.bee_count} bees
                                    </span>
                                </div>
                            </div>
                            <div class="image-comparison">
                                <div class="image-box">
                                    <h5><i class="fas fa-image"></i> Original</h5>
                                    <div class="image-wrapper">
                                        <span class="image-label">ORIGINAL</span>
                                        ${originalHTML}
                                    </div>
                                </div>
                                <div class="image-box">
                                    <h5><i class="fas fa-magic"></i> Detected</h5>
                                    <div class="image-wrapper">
                                        <span class="image-label">OUTPUT</span>
                                        ${detectedHTML}
                                    </div>
                                </div>
                            </div>
                            <div style="margin-top: 0.75rem; font-size: 0.75rem;">
                                <i class="fas fa-cloud" style="color: var(--accent-green); margin-right: 0.3rem;"></i>
                                <span style="color: var(--text-secondary);">Detected frame on S3: </span>
                                ${s3Link}
                            </div>
                        </div>`;
                }).join('');
            } else {
                card.style.display = 'none';
                container.innerHTML = '';
            }
        }

        // ===== MICROPHONE RECORDING =====
        function toggleRecording() {
            // Click #1 starts recording, click #2 stops + analyzes.
            // Auto-stops at max_duration (20 s) if the user doesn't click Stop first.
            if (isRecording) {
                stopRecordingAndAnalyze();
                return;
            }
            const btn = document.getElementById('recordBtn');
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';

            fetch('/record-audio', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        alert('Error: ' + data.error);
                        btn.disabled = false;
                        btn.innerHTML = '<i class="fas fa-circle"></i> Record Audio';
                        return;
                    }
                    isRecording = true;
                    btn.classList.add('recording');
                    btn.disabled = false;
                    btn.innerHTML = '<i class="fas fa-stop"></i> Stop & Analyze';
                    startVisualizer();

                    const maxDur = data.max_duration || 20;
                    const startTs = Date.now();
                    const statusEl = document.getElementById('audioStatus');
                    statusEl.textContent = `Recording… 0.0 / ${maxDur}s  (tap Stop when done)`;

                    // Live elapsed timer
                    if (window._recTimer) clearInterval(window._recTimer);
                    window._recTimer = setInterval(() => {
                        if (!isRecording) { clearInterval(window._recTimer); window._recTimer = null; return; }
                        const elapsed = (Date.now() - startTs) / 1000;
                        statusEl.textContent = `Recording… ${elapsed.toFixed(1)} / ${maxDur}s  (tap Stop when done)`;
                    }, 100);

                    // Auto-stop at max duration (server also enforces the cap)
                    if (window._recAutoStop) clearTimeout(window._recAutoStop);
                    window._recAutoStop = setTimeout(() => {
                        if (isRecording) stopRecordingAndAnalyze();
                    }, (maxDur + 0.5) * 1000);
                })
                .catch(err => {
                    btn.disabled = false;
                    btn.innerHTML = '<i class="fas fa-circle"></i> Record Audio';
                    alert('Failed to start recording: ' + err.message);
                });
        }

        function stopRecordingAndAnalyze() {
            const btn = document.getElementById('recordBtn');
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            stopVisualizer();
            if (window._recTimer) { clearInterval(window._recTimer); window._recTimer = null; }
            if (window._recAutoStop) { clearTimeout(window._recAutoStop); window._recAutoStop = null; }
            document.getElementById('audioStatus').textContent = 'Analyzing with BeeCNN...';

            fetch('/analyze-recorded-audio', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    isRecording = false;
                    btn.classList.remove('recording');
                    btn.disabled = false;
                    btn.innerHTML = '<i class="fas fa-circle"></i> Record Audio';
                    document.getElementById('audioStatus').textContent = 'Ready to record';
                    if (data.error) { alert('Analysis error: ' + data.error); return; }
                    displayAudioResults(data);
                    loadDashboardStats();
                    loadAudioTrends('daily', 'audioTrendChart');
                })
                .catch(err => {
                    isRecording = false;
                    btn.classList.remove('recording');
                    btn.disabled = false;
                    btn.innerHTML = '<i class="fas fa-circle"></i> Record Audio';
                    document.getElementById('audioStatus').textContent = 'Ready to record';
                    alert('Analysis failed: ' + err.message);
                });
        }

        function displayAudioResults(data) {
            const results = document.getElementById('audioResults');
            results.classList.add('show');

            document.getElementById('classificationResult').textContent = data.level;
            document.getElementById('classificationSubtext').textContent = 'Estimated Range: ' + data.bee_range + ' bees';

            const lowPct = Math.round(data.low_prob * 100);
            const medPct = Math.round(data.med_prob * 100);
            const highPct = Math.round(data.high_prob * 100);

            document.getElementById('probBarLow').style.width = lowPct + '%';
            document.getElementById('probValLow').textContent = lowPct + '%';
            document.getElementById('probBarMed').style.width = medPct + '%';
            document.getElementById('probValMed').textContent = medPct + '%';
            document.getElementById('probBarHigh').style.width = highPct + '%';
            document.getElementById('probValHigh').textContent = highPct + '%';

            document.getElementById('estimatedCount').textContent = '~' + data.estimated_count;
            document.getElementById('levelValue').textContent = data.level;
            document.getElementById('stressValue').textContent = data.stress_level;
            document.getElementById('swarmingValue').textContent = data.swarming_probability + '%';
            document.getElementById('activityValue').textContent = data.activity_intensity + '%';
            document.getElementById('anomalyValue').textContent = data.anomaly_detected ? 'Yes' : 'No';

            const stressMetric = document.getElementById('stressMetric');
            stressMetric.className = 'audio-metric status-' + data.stress_level.toLowerCase();

            const anomalyMetric = document.getElementById('anomalyMetric');
            anomalyMetric.className = 'audio-metric ' + (data.anomaly_detected ? 'status-high' : 'status-low');

            const levelMetric = document.getElementById('levelMetric');
            const levelLower = data.level.toLowerCase();
            if (levelLower.includes('very low') || levelLower.includes('low')) {
                levelMetric.className = 'audio-metric status-low';
            } else if (levelLower.includes('medium')) {
                levelMetric.className = 'audio-metric status-moderate';
            } else {
                levelMetric.className = 'audio-metric status-high';
            }

            const classificationCard = document.getElementById('audioClassification');
            if (levelLower.includes('low')) {
                classificationCard.style.background = 'linear-gradient(135deg, #10b981, #059669)';
            } else if (levelLower.includes('medium')) {
                classificationCard.style.background = 'linear-gradient(135deg, #f59e0b, #d97706)';
            } else {
                classificationCard.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
            }

            if (window.frequencyChart) {
                window.frequencyChart.data.datasets[0].data = data.frequency_data;
                window.frequencyChart.update();
            }
        }

        // Audio visualizer animation
        let visualizerInterval;
        function startVisualizer() {
            const container = document.getElementById('visualizerBars');
            container.innerHTML = '';
            for (let i = 0; i < 30; i++) {
                const bar = document.createElement('div');
                bar.className = 'vbar';
                bar.style.height = '5px';
                container.appendChild(bar);
            }
            visualizerInterval = setInterval(() => {
                document.querySelectorAll('.vbar').forEach(bar => {
                    bar.style.height = (Math.random() * 50 + 5) + 'px';
                });
            }, 100);
        }

        function stopVisualizer() {
            clearInterval(visualizerInterval);
            document.querySelectorAll('.vbar').forEach(bar => { bar.style.height = '5px'; });
        }

        // ===== CHARTS & DASHBOARD =====
        function loadCVHistory(canvasId) {
            fetch('/api/cv-history')
                .then(r => r.json())
                .then(data => {
                    const ctx = document.getElementById(canvasId);
                    if (!ctx) return;
                    if (ctx.tagName !== 'CANVAS') {
                        ctx.parentElement.innerHTML = `<canvas id="${canvasId}"></canvas>`;
                    }
                    const canvas = document.getElementById(canvasId);
                    if (!canvas) return;

                    if (data.length === 0) {
                        canvas.parentElement.innerHTML = '<div class="no-data">No detection data yet. Run some captures!</div>';
                        return;
                    }

                    // Decide whether to show the full date or just HH:MM. If the
                    // bucket range spans more than one day, include the date.
                    const firstDay = new Date(data[0].timestamp).toDateString();
                    const spansMultipleDays = data.some(d => new Date(d.timestamp).toDateString() !== firstDay);

                    const fmt = (iso) => {
                        const date = new Date(iso);
                        const hm = date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
                        if (!spansMultipleDays) return hm;
                        const md = date.toLocaleDateString('en-US', { month: '2-digit', day: '2-digit' });
                        return `${md}  ${hm}`;
                    };
                    const labels = data.map(d => fmt(d.timestamp));
                    const counts = data.map(d => d.bee_count);

                    const chartKey = 'cvChart_' + canvasId;
                    if (window[chartKey] && typeof window[chartKey].destroy === 'function') {
                        window[chartKey].destroy();
                    }

                    window[chartKey] = new Chart(canvas.getContext('2d'), {
                        type: 'bar',
                        data: {
                            labels: labels,
                            datasets: [{
                                label: 'Peak bees per 10-min window',
                                data: counts,
                                backgroundColor: '#f59e0b',
                                borderColor: '#f59e0b',
                                borderWidth: 1,
                                borderRadius: 4,
                                barPercentage: 0.8,
                                categoryPercentage: 0.9
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    grid: { color: '#334155' },
                                    ticks: { color: '#94a3b8', precision: 0 },
                                    title: { display: true, text: 'Bee count', color: '#94a3b8' }
                                },
                                x: {
                                    grid: { display: false },
                                    ticks: {
                                        color: '#94a3b8',
                                        maxRotation: 45,
                                        minRotation: 45,
                                        // One label per bar — we already aggregated to at most 25 buckets
                                        autoSkip: false
                                    }
                                }
                            },
                            plugins: {
                                legend: { labels: { color: '#94a3b8' } },
                                tooltip: {
                                    callbacks: {
                                        title: (items) => items[0] ? 'Window start: ' + labels[items[0].dataIndex] : '',
                                        label: (item) => `Peak bees: ${item.parsed.y}`
                                    }
                                }
                            }
                        }
                    });
                });
        }

        function loadAudioTrends(range, canvasId) {
            fetch(`/api/audio-history?range=${range}`)
                .then(r => r.json())
                .then(data => {
                    const ctx = document.getElementById(canvasId);
                    if (!ctx) return;

                    if (data.length === 0) {
                        ctx.parentElement.innerHTML = '<div class="no-data">No audio data yet. Upload or record audio!</div>';
                        return;
                    }

                    // If canvas was replaced by no-data, restore it
                    if (ctx.tagName !== 'CANVAS') {
                        ctx.parentElement.innerHTML = `<canvas id="${canvasId}"></canvas>`;
                    }

                    const labels = data.map(d => {
                        const date = new Date(d.timestamp);
                        return date.toLocaleString('en-US', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
                    });
                    const counts = data.map(d => d.estimated_count);

                    const chartKey = 'audioChart_' + canvasId;
                    if (window[chartKey]) window[chartKey].destroy();

                    window[chartKey] = new Chart(document.getElementById(canvasId).getContext('2d'), {
                        type: 'line',
                        data: {
                            labels: labels,
                            datasets: [{
                                label: 'Estimated Bee Count',
                                data: counts,
                                borderColor: '#8b5cf6',
                                backgroundColor: 'rgba(139, 92, 246, 0.1)',
                                fill: true,
                                tension: 0.4,
                                pointRadius: 3,
                                pointBackgroundColor: '#8b5cf6'
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    grid: { color: '#334155' },
                                    ticks: { color: '#94a3b8' }
                                },
                                x: {
                                    grid: { color: '#334155' },
                                    ticks: { color: '#94a3b8', maxTicksLimit: 8 }
                                }
                            },
                            plugins: {
                                legend: { display: false }
                            }
                        }
                    });
                });
        }

        function switchAudioRange(range, btn) {
            btn.parentElement.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadAudioTrends(range, 'audioTrendChart');
        }

        function switchDashAudioRange(range, btn) {
            btn.parentElement.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadAudioTrends(range, 'dashAudioTrendChart');
        }

        function loadAlerts() {
            fetch('/api/alerts')
                .then(r => r.json())
                .then(data => {
                    const list = document.getElementById('alertsList');
                    if (!list) return;

                    if (data.length === 0) {
                        list.innerHTML = '<p style="color: var(--text-secondary); text-align: center;">No recent alerts</p>';
                        return;
                    }

                    list.innerHTML = data.map(alert => {
                        const date = new Date(alert.timestamp);
                        const timeStr = date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true });
                        let iconClass = 'fa-info-circle';
                        let alertClass = 'info';
                        if (alert.severity === 'critical') { iconClass = 'fa-exclamation-circle'; alertClass = 'critical'; }
                        else if (alert.severity === 'warning') { iconClass = 'fa-exclamation-triangle'; alertClass = 'warning'; }

                        return `
                            <div class="alert-item">
                                <div class="alert-icon ${alertClass}">
                                    <i class="fas ${iconClass}"></i>
                                </div>
                                <div class="alert-content">
                                    <div class="alert-time">${timeStr}</div>
                                    <div class="alert-title">${alert.title}</div>
                                    <div class="alert-message">${alert.message}</div>
                                </div>
                            </div>
                        `;
                    }).join('');
                });
        }

        // ===== MAP =====
        let mapInstance = null;
                function initMap() {
            const mapEl = document.getElementById('testMap');
            if (!mapEl || typeof L === 'undefined') return;
            if (mapInstance) { mapInstance.remove(); }

            mapInstance = L.map('testMap').setView([34.04, 72.63], 11);

            L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
                attribution: '© OpenStreetMap, © CARTO',
                maxZoom: 19
            }).addTo(mapInstance);

            const topiIcon = L.divIcon({
                html: '<div style="background: var(--accent-green); width: 14px; height: 14px; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 10px rgba(16,185,129,0.6);"></div>',
                className: 'custom-div-icon',
                iconSize: [14, 14],
                iconAnchor: [7, 7]
            });

            const ghaziIcon = L.divIcon({
                html: '<div style="background: var(--accent-primary); width: 14px; height: 14px; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 10px rgba(245,158,11,0.6);"></div>',
                className: 'custom-div-icon',
                iconSize: [14, 14],
                iconAnchor: [7, 7]
            });

            L.marker([34.0708, 72.6200], { icon: topiIcon }).addTo(mapInstance)
                .bindPopup('<b>Topi, KPK</b><br>Test Site 1<br><span style="color: var(--accent-green);">● Active</span>');
            L.marker([34.0167, 72.6500], { icon: ghaziIcon }).addTo(mapInstance)
                .bindPopup('<b>Ghazi, KPK</b><br>Test Site 2<br><span style="color: var(--accent-primary);">● Active</span>');
        }

        // Initialize charts
        document.addEventListener('DOMContentLoaded', function() {
            const ctx = document.getElementById('frequencyChart');
            if (ctx) {
                window.frequencyChart = new Chart(ctx.getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: Array.from({length: 20}, (_, i) => i * 100),
                        datasets: [{
                            label: 'Amplitude',
                            data: [],
                            borderColor: '#8b5cf6',
                            backgroundColor: 'rgba(139, 92, 246, 0.1)',
                            fill: true,
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: { beginAtZero: true, grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                            x: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } }
                        },
                        plugins: { legend: { display: false } }
                    }
                });
            }

            showLoading('Initializing Jetson...', 'Starting camera and audio systems');
            fetch('/init-camera', { method: 'POST' })
                .then(() => {
                    hideLoading();
                    loadCVHistory('cvHistoryChart');
                })
                .catch(() => {
                    hideLoading();
                    document.getElementById('cameraStatus').innerHTML = '<i class="fas fa-camera" style="color: var(--accent-red);"></i><span>Camera Offline</span>';
                });
        });
    </script>
</body>
</html>
"""


# ============================================================
# FLASK ROUTES - JETSON HARDWARE CONTROL
# ============================================================

@app.route("/init-camera", methods=["POST"])
def init_camera_route():
    """Initialize the camera"""
    success = init_camera(camera_type="csi", device_id=0)
    if not success:
        success = init_camera(camera_type="usb", device_id=0)
    return jsonify({"success": success})


@app.route("/video_feed")
def video_feed():
    """MJPEG video stream from camera"""
    return Response(
        generate_camera_feed(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ============================================================
# LIVE DETECTION ROUTES
# Background thread grabs a frame from the local camera, posts it to the
# existing Roboflow run_workflow (team-yolo) API, and exposes the latest
# annotated frame via the /live-feed MJPEG endpoint.
# ============================================================


def _live_capture_loop():
    """Dedicated capture thread. Reads frames from the camera as fast as the
    hardware allows and publishes them to `latest_raw_frame`. This keeps the
    MJPEG feed smooth even while a slow Roboflow API call is in flight."""
    global latest_raw_frame
    while live_session is not None and live_session.get("active"):
        try:
            frame = capture_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            with live_session_lock:
                latest_raw_frame = frame
            # ~20 FPS target; the camera hardware is the real bottleneck anyway.
            time.sleep(0.05)
        except Exception as e:
            print(f"Live capture error: {e}")
            time.sleep(0.2)


def _live_inference_loop():
    """Dedicated inference thread. Repeatedly grabs the latest raw frame (from
    memory, not the camera — no lock contention), sends it to Roboflow, and
    publishes the annotated result + bee count. Runs as fast as the API allows."""
    global latest_annotated_frame, latest_live_count
    iteration = 0
    LIVE_MAX_WIDTH = 640  # aggressive downscale = faster round-trip

    while live_session is not None and live_session.get("active"):
        # Snapshot the latest raw frame
        with live_session_lock:
            frame = latest_raw_frame.copy() if latest_raw_frame is not None else None
        if frame is None:
            time.sleep(0.1)
            continue

        # Downscale before uploading to Roboflow
        h, w = frame.shape[:2]
        if w > LIVE_MAX_WIDTH:
            scale = LIVE_MAX_WIDTH / float(w)
            send_frame = cv2.resize(frame, (LIVE_MAX_WIDTH, int(h * scale)))
        else:
            send_frame = frame

        temp_path = os.path.join(
            CAPTURE_FOLDER, f"live_{int(time.time() * 1000)}.jpg"
        )
        cv2.imwrite(temp_path, send_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        try:
            result = realtime_client.run_workflow(
                workspace_name="team-yolo",
                workflow_id="detect-count-and-visualize-2",
                images={"image": temp_path},
                use_cache=True,
            )
            data = result[0] if result else {}
            bee_count = int(data.get("count_objects", 0) or 0)
            output_b64 = data.get("output_image")

            if output_b64:
                try:
                    nparr = np.frombuffer(base64.b64decode(output_b64), np.uint8)
                    annotated = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if annotated is not None:
                        with live_session_lock:
                            latest_annotated_frame = annotated
                except Exception as e:
                    print(f"Live: failed to decode annotated frame: {e}")

            latest_live_count = bee_count

            # Log every ~10 inferences so the dashboard trend stays live
            iteration += 1
            if iteration % 10 == 0:
                log_cv_detection(bee_count, "live")
                increment_dashboard_stats(total_detections=bee_count)
        except Exception as e:
            print(f"Live inference error: {e}")
            time.sleep(0.3)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
    print("Live inference thread exited")


@app.route("/start-live-detection", methods=["POST"])
def start_live_detection():
    """Start the two-thread live-detection system (capture + inference)."""
    global live_session, live_capture_thread, live_inference_thread
    global latest_raw_frame, latest_annotated_frame, latest_live_count

    if live_session is not None:
        return jsonify({"error": "Live detection already running"}), 400

    if camera is None:
        if not init_camera():
            return jsonify({"error": "Camera not available"}), 500

    live_session = {"active": True}
    with live_session_lock:
        latest_raw_frame = None
        latest_annotated_frame = None
    latest_live_count = 0

    live_capture_thread = threading.Thread(target=_live_capture_loop, daemon=True)
    live_capture_thread.start()
    live_inference_thread = threading.Thread(target=_live_inference_loop, daemon=True)
    live_inference_thread.start()

    add_alert("Live Detection Started", "Real-time Roboflow detection session started", "info")
    return jsonify({"success": True})


@app.route("/stop-live-detection", methods=["POST"])
def stop_live_detection():
    """Signal both background threads to exit and clear shared state."""
    global live_session, latest_raw_frame, latest_annotated_frame, latest_live_count
    if live_session is None:
        return jsonify({"error": "No live session running"}), 400
    live_session["active"] = False
    live_session = None
    with live_session_lock:
        latest_raw_frame = None
        latest_annotated_frame = None
    latest_live_count = 0
    add_alert("Live Detection Stopped", "Real-time detection session stopped", "info")
    return jsonify({"success": True})


@app.route("/live-feed")
def live_feed():
    """MJPEG stream: prefers the latest annotated frame, falls back to the
    latest raw frame so the browser always sees current video even during
    long Roboflow round-trips."""
    def gen():
        while True:
            if live_session is None:
                break
            with live_session_lock:
                frame = latest_annotated_frame if latest_annotated_frame is not None else latest_raw_frame
            if frame is not None:
                ok, buffer = cv2.imencode(".jpg", frame)
                if ok:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + buffer.tobytes()
                        + b"\r\n"
                    )
            time.sleep(0.05)  # ~20 FPS UI refresh
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/live-count", methods=["GET"])
def live_count():
    """Polled by the browser to display the real-time bee count."""
    return jsonify({
        "count_objects": int(latest_live_count),
        "active": live_session is not None,
    })


@app.route("/capture-and-detect", methods=["POST"])
def capture_and_detect():
    """Capture a frame from camera and run detection"""
    global camera
    if camera is None:
        if not init_camera():
            return jsonify({"error": "Camera not available"}), 500

    frame = capture_frame()
    if frame is None:
        return jsonify({"error": "Failed to capture frame"}), 500

    timestamp = int(time.time())
    filename = f"capture_{timestamp}.jpg"
    capture_path = save_captured_frame(frame, filename)
    captured_base64 = frame_to_base64(frame)

    try:
        start_time = time.time()
        result = realtime_client.run_workflow(
            workspace_name="team-yolo",
            workflow_id="detect-count-and-visualize-2",
            images={"image": capture_path},
            use_cache=True
        )
        inference_time = round(time.time() - start_time, 2)

        data = result[0]
        bee_count = data.get("count_objects", 0)
        output_image_base64 = data.get("output_image", None)

        s3_url = upload_to_s3(capture_path, filename, folder="jetson/captures")

        # Log to history
        log_cv_detection(bee_count, "camera")
        increment_dashboard_stats(captures=1, total_detections=int(bee_count or 0))

        return jsonify({
            "success": True,
            "filename": filename,
            "bee_count": bee_count,
            "inference_time": inference_time,
            "captured_image": captured_base64,
            "output_image": output_image_base64 if output_image_base64 else captured_base64,
            "s3_url": s3_url
        })

    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "filename": filename,
            "bee_count": 0,
            "inference_time": 0,
            "captured_image": captured_base64,
            "output_image": captured_base64,
            "s3_url": None
        })
    finally:
        if os.path.exists(capture_path):
            os.remove(capture_path)


@app.route("/capture-burst", methods=["POST"])
def capture_burst():
    """Capture 5 frames in burst mode"""
    global camera
    if camera is None:
        if not init_camera():
            return jsonify({"error": "Camera not available"}), 500

    results = []
    for i in range(5):
        frame = capture_frame()
        if frame is None:
            continue

        timestamp = int(time.time())
        filename = f"burst_{timestamp}_{i}.jpg"
        capture_path = save_captured_frame(frame, filename)
        captured_base64 = frame_to_base64(frame)

        try:
            start_time = time.time()
            result = realtime_client.run_workflow(
                workspace_name="team-yolo",
                workflow_id="detect-count-and-visualize-2",
                images={"image": capture_path},
                use_cache=True
            )
            inference_time = round(time.time() - start_time, 2)

            data = result[0]
            bee_count = data.get("count_objects", 0)
            output_image_base64 = data.get("output_image", None)

            s3_url = upload_to_s3(capture_path, filename, folder="jetson/burst")

            # Log each burst detection
            log_cv_detection(bee_count, "burst")
            increment_dashboard_stats(captures=1, total_detections=int(bee_count or 0))

            results.append({
                "filename": filename,
                "bee_count": bee_count,
                "inference_time": inference_time,
                "captured_image": captured_base64,
                "output_image": output_image_base64 if output_image_base64 else captured_base64,
                "s3_url": s3_url
            })
        except Exception as e:
            print(f"Burst frame {i} error: {e}")
            results.append({
                "filename": filename,
                "bee_count": 0,
                "inference_time": 0,
                "captured_image": captured_base64,
                "output_image": captured_base64,
                "s3_url": None
            })
        finally:
            if os.path.exists(capture_path):
                os.remove(capture_path)

        time.sleep(0.2)

    return jsonify({"success": True, "results": results})


# ============================================================
# AUDIO ROUTES - FILE UPLOAD + MICROPHONE
# ============================================================

@app.route("/upload-audio", methods=["POST"])
def upload_audio():
    """
    Upload test_audio file, run BeeCNN prediction,
    store to S3 at audio/wav, and return graph data.
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded file
    timestamp = int(time.time())
    filename = f"test_audio_{timestamp}_{file.filename}"
    filepath = os.path.join(AUDIO_UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # Run BeeCNN prediction (audio clips are NOT uploaded to S3)
        result = predict_audio(filepath)

        # Log analysis result to history (stored on S3)
        log_audio_analysis(result)
        increment_dashboard_stats(audio_samples=1)

        return jsonify(result)

    except Exception as e:
        print(f"Audio upload/analysis error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


audio_recording_path = None

@app.route("/record-audio", methods=["POST"])
def start_audio_recording():
    """Start a user-stoppable audio recording (hard-capped at 20 s).
    The frontend calls /analyze-recorded-audio to stop + analyze."""
    global audio_recording_path
    try:
        timestamp = int(time.time())
        filename = f"audio_capture_{timestamp}.wav"
        audio_recording_path = os.path.join(AUDIO_UPLOAD_FOLDER, filename)

        if not start_recording(audio_recording_path, max_duration=AUDIO_MAX_DURATION):
            return jsonify({"error": "Already recording"}), 400

        return jsonify({
            "success": True,
            "max_duration": AUDIO_MAX_DURATION,
            "filename": filename,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-recorded-audio", methods=["POST"])
def analyze_recorded_audio():
    """Stop the active recording (if any), then run BeeCNN on the resulting WAV.
    Works for recordings of any length between ~1 s and AUDIO_MAX_DURATION s."""
    global audio_recording_path

    # If the user has been recording, stop the worker and wait for the WAV to flush
    stop_recording()

    if audio_recording_path is None or not os.path.exists(audio_recording_path):
        return jsonify({"error": "No recording found"}), 400

    try:
        result = predict_audio(audio_recording_path)
        result["filename"] = os.path.basename(audio_recording_path)

        log_audio_analysis(result)
        increment_dashboard_stats(audio_samples=1)

        os.remove(audio_recording_path)
        audio_recording_path = None
        return jsonify(result)
    except Exception as e:
        if audio_recording_path and os.path.exists(audio_recording_path):
            try: os.remove(audio_recording_path)
            except OSError: pass
            audio_recording_path = None
        return jsonify({"error": str(e)}), 500


# ============================================================
# VIDEO PROCESSING ROUTE
# ============================================================

@app.route("/upload-video", methods=["POST"])
def upload_video():
    """
    Upload video file, process frame-by-frame with Roboflow,
    generate annotated output video, upload to S3, return results.
    """
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    timestamp = int(time.time())
    filename = f"video_{timestamp}_{file.filename}"
    input_path = os.path.join(VIDEO_UPLOAD_FOLDER, filename)
    file.save(input_path)

    try:
        # Split the video into 5 frames, run each through Roboflow,
        # upload ONLY the annotated frames to S3. Raw video is NOT uploaded.
        summary, err = process_video_file(input_path)
        if summary is None:
            return jsonify({"error": err or "Video processing failed"}), 500

        # Log video detections + dashboard stats
        log_cv_detection(summary.get("max_bees_in_single_frame", 0), "video")
        increment_dashboard_stats(
            videos_processed=1,
            total_detections=int(summary.get("total_bees_detected", 0) or 0),
        )
        if summary.get("total_bees_detected", 0) > 200:
            add_alert(
                "High Video Activity",
                f"Video peak: {summary.get('max_bees_in_single_frame', 0)} bees",
                "warning",
            )

        return jsonify({"success": True, "summary": summary})

    except Exception as e:
        print(f"Video processing error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Raw uploaded video is discarded locally — nothing to upload to S3.
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass

# ============================================================
# IMAGE UPLOAD & DETECTION ROUTE
# ============================================================

@app.route("/upload-image", methods=["POST"])
def upload_image():
    """
    Upload image file, run Roboflow bee detection,
    store original + annotated versions to S3,
    return results for frontend display.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    timestamp = int(time.time())
    filename = f"upload_{timestamp}_{file.filename}"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    try:
        # Read original image for base64 response
        with open(input_path, "rb") as f:
            original_base64 = base64.b64encode(f.read()).decode('utf-8')

        # Run Roboflow detection
        start_time = time.time()
        result = client.run_workflow(
            workspace_name="asad-fnvcs",
            workflow_id="detect-count-and-visualize-2",
            images={"image": input_path},
            use_cache=True
        )
        inference_time = round(time.time() - start_time, 2)

        data = result[0]
        bee_count = data.get("count_objects", 0)
        output_image_base64 = data.get("output_image", None)

        # Save annotated image temporarily for S3 upload
        annotated_path = None
        if output_image_base64:
            annotated_path = os.path.join(RESULT_FOLDER, f"annotated_{filename}")
            with open(annotated_path, "wb") as f:
                f.write(base64.b64decode(output_image_base64))

        # Upload original to S3
        s3_original_url = upload_to_s3(input_path, filename, folder="images/uploaded/original")

        # Upload annotated to S3
        s3_annotated_url = None
        if annotated_path and os.path.exists(annotated_path):
            s3_annotated_url = upload_to_s3(
                annotated_path, 
                f"annotated_{filename}", 
                folder="images/uploaded/annotated"
            )

        # Log to history
        log_cv_detection(bee_count, "upload")
        increment_dashboard_stats(captures=1, total_detections=int(bee_count or 0))

        # Build response
        response = {
            "success": True,
            "filename": filename,
            "bee_count": bee_count,
            "inference_time": inference_time,
            "original_image": original_base64,
            "output_image": output_image_base64 if output_image_base64 else original_base64,
            "s3_original_url": s3_original_url,
            "s3_annotated_url": s3_annotated_url,
            "model_loaded": True
        }

        return jsonify(response)

    except Exception as e:
        print(f"Image upload/detection error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "filename": filename,
            "bee_count": 0,
            "inference_time": 0,
            "original_image": original_base64 if 'original_base64' in locals() else None,
            "output_image": original_base64 if 'original_base64' in locals() else None,
            "s3_original_url": None,
            "s3_annotated_url": None
        }), 500

    finally:
        # Cleanup temp files
        if os.path.exists(input_path):
            os.remove(input_path)
        annotated_path = os.path.join(RESULT_FOLDER, f"annotated_{filename}")
        if os.path.exists(annotated_path):
            os.remove(annotated_path)
# ============================================================
# DASHBOARD API ROUTES
# ============================================================

@app.route("/api/cv-history", methods=["GET"])
def get_cv_history():
    """Return cv-history aggregated into 10-minute buckets (max bee_count per
    bucket) so the dashboard bar chart shows a clean, time-accurate timeline.
    Caps the output at the most recent 25 buckets."""
    history = load_json_from_s3(HISTORY_S3_KEY)
    if not history:
        return jsonify([])

    buckets = {}
    for entry in history:
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
        except Exception:
            continue
        # Round down to the nearest 10-minute boundary
        bucket_minute = (ts.minute // 10) * 10
        bucket_ts = ts.replace(minute=bucket_minute, second=0, microsecond=0)
        key = bucket_ts.isoformat()
        count = int(entry.get("bee_count", 0) or 0)
        if key not in buckets or count > buckets[key]["bee_count"]:
            buckets[key] = {
                "timestamp": key,
                "bee_count": count,
                "source": entry.get("source", "camera"),
            }

    sorted_buckets = sorted(buckets.values(), key=lambda x: x["timestamp"])
    return jsonify(sorted_buckets[-25:])

@app.route("/api/audio-history", methods=["GET"])
def get_audio_history():
    range_type = request.args.get("range", "daily")
    history = load_json_from_s3(AUDIO_HISTORY_S3_KEY)

    now = datetime.now()
    if range_type == "daily":
        cutoff = now - timedelta(days=1)
    elif range_type == "weekly":
        cutoff = now - timedelta(weeks=1)
    else:  # monthly
        cutoff = now - timedelta(days=30)

    filtered = [h for h in history if datetime.fromisoformat(h["timestamp"]) > cutoff]
    return jsonify(filtered)

@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    alerts = load_json_from_s3(ALERTS_S3_KEY)
    return jsonify(alerts[:20])  # Last 20 alerts

@app.route("/api/dashboard-stats", methods=["GET"])
def get_dashboard_stats():
    """Return cumulative dashboard counters stored on S3."""
    return jsonify(load_dashboard_stats())


@app.route("/api/clear-history", methods=["POST"])
def clear_history():
    """Wipe the detection history, audio history, and dashboard counters on S3.
    Alerts are left intact so the startup 'Device Online' trail is preserved.
    Useful for cleaning up data accumulated during a prior auto-start session."""
    save_json_to_s3(HISTORY_S3_KEY, [])
    save_json_to_s3(AUDIO_HISTORY_S3_KEY, [])
    save_json_to_s3(DASHBOARD_STATS_S3_KEY, dict(DEFAULT_DASHBOARD_STATS))
    return jsonify({"success": True, "message": "History and dashboard stats cleared."})


# ============================================================
# MAIN PAGE
# ============================================================
@app.route("/")
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)


# Cleanup on shutdown
def cleanup():
    """Release resources on shutdown"""
    release_camera()
    if audio_recording_path and os.path.exists(audio_recording_path):
        os.remove(audio_recording_path)

import atexit
atexit.register(cleanup)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    
    print("=" * 60)
    print("Your App Name")  # Replace with your actual app name
    print("=" * 60)
    print(f"Port: {port}")
    print(f"Environment: {'Render' if os.getenv('RENDER') else 'Local'}")
    print("=" * 60)

    # Auto-open browser when running locally (not on Render)
    if os.getenv("RENDER") is None:
        url = f"http://127.0.0.1:{port}"
        print(f"Opening browser at {url}")
        webbrowser.open(url)

    try:
        # Bind to all interfaces for deployment compatibility
        # threaded=True allows handling multiple requests concurrently
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    finally:
        # Add any cleanup logic here if needed
        pass