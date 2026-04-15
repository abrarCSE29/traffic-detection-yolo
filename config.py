from pathlib import Path

# Paths
UPLOAD_FOLDER = Path("uploads")
RESULT_FOLDER = Path("results")
DEMO_FOLDER = Path("demo_video")
DEMO_VIDEO_PATH = DEMO_FOLDER / "demo.mp4"
BEST_MODEL_PATH = Path("models/best.pt")

# Detection classes (must match model training order)
CLASSES = ["Bike", "Bus", "Car", "Cng", "People", "Rickshaw", "Truck", "Mini-Truck", "Cycle"]

# BGR colors per class (for OpenCV annotation)
CLASS_COLORS = {
    "Bike":      (0, 255, 255),
    "Bus":       (255, 0, 0),
    "Car":       (0, 255, 0),
    "Cng":       (0, 165, 255),
    "People":    (255, 0, 255),
    "Rickshaw":  (255, 255, 0),
    "Truck":     (0, 0, 255),
    "Mini-Truck":(128, 0, 128),
    "Cycle":     (0, 255, 128),
}

# Streaming
TARGET_FPS    = 15
JPEG_QUALITY  = 60

# Inference
CONF_THRESHOLD  = 0.3
IOU_THRESHOLD   = 0.5
INFERENCE_SIZE  = 480

# Speed estimation
PX_TO_METER       = 0.05
SPEED_WINDOW      = 10
TRACK_HISTORY_MAX = 20
SPEED_MIN_KMH     = 2   # below this is treated as stationary
