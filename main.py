import os
import cv2
import json
import uuid
import base64
import asyncio
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Request, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI()

# ── CORS Configuration ────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Configuration ─────────────────────────────────────────────────────────────
UPLOAD_FOLDER = Path("uploads")
RESULT_FOLDER = Path("results")
DEMO_FOLDER = Path("demo_video")
DEMO_VIDEO_PATH = DEMO_FOLDER / "demo.mp4"
BEST_MODEL_PATH = Path("models/best.pt")
CLASSES = ["Bike", "Bus", "Car", "Cng", "People", "Rickshaw", "Truck", "Mini-Truck", "Cycle"]

# Color map for each class (BGR format for OpenCV)
CLASS_COLORS = {
    "Bike": (0, 255, 255),      # Yellow
    "Bus": (255, 0, 0),          # Blue
    "Car": (0, 255, 0),          # Green
    "Cng": (0, 165, 255),        # Orange
    "People": (255, 0, 255),     # Magenta
    "Rickshaw": (255, 255, 0),   # Cyan
    "Truck": (0, 0, 255),        # Red
    "Mini-Truck": (128, 0, 128), # Purple
    "Cycle": (0, 255, 128)       # Spring Green
}

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(exist_ok=True)

# Load YOLO model and move to GPU
try:
    model = YOLO(str(BEST_MODEL_PATH))
    model.to('cuda')
    print("✓ Model loaded on GPU (CUDA)")
except Exception as e:
    print(f"⚠ CUDA failed, trying CPU: {e}")
    model = YOLO(str(BEST_MODEL_PATH))
    print("✓ Model loaded on CPU")

# Use a specific ThreadPoolExecutor to control parallel worker count
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4) 

# Mount static folders
app.mount("/results", StaticFiles(directory=str(RESULT_FOLDER)), name="results")
app.mount("/demo_video", StaticFiles(directory=str(DEMO_FOLDER)), name="demo_video")

# Storage for active jobs
processing_jobs: Dict[str, dict] = {}

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("templates/index.html", "r") as f:
        return f.read()

@app.get("/classes")
async def get_classes():
    """Return available classes for filtering"""
    return {"classes": CLASSES}

@app.post("/upload")
async def upload_video(video: UploadFile = File(...), selected_classes: str = Form("all")):
    job_id = str(uuid.uuid4())
    filename = f"{job_id}_{video.filename}"
    file_path = UPLOAD_FOLDER / filename
    
    with open(file_path, "wb") as buffer:
        buffer.write(await video.read())
    
    # Parse selected classes
    if selected_classes == "all":
        filter_classes = CLASSES
    else:
        filter_classes = selected_classes.split(",")

    processing_jobs[job_id] = {
        "status": "ready",
        "video_path": str(file_path),
        "total_frames": 0,
        "current_frame": 0,
        "counts": {},
        "filter_classes": filter_classes
    }
    
    return {"job_id": job_id}

@app.post("/demo-job")
async def create_demo_job(selected_classes: str = Form("all")):
    """Create a processing job using built-in demo video."""
    if not DEMO_VIDEO_PATH.exists():
        raise HTTPException(status_code=404, detail="Demo video not found")

    job_id = str(uuid.uuid4())

    if selected_classes == "all":
        filter_classes = CLASSES
    else:
        filter_classes = selected_classes.split(",")

    processing_jobs[job_id] = {
        "status": "ready",
        "video_path": str(DEMO_VIDEO_PATH),
        "total_frames": 0,
        "current_frame": 0,
        "counts": {},
        "filter_classes": filter_classes
    }

    return {"job_id": job_id, "video_name": DEMO_VIDEO_PATH.name}

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    
    # Create a lock for WebSocket writes to prevent concurrent access
    ws_lock = asyncio.Lock()
    
    job = processing_jobs.get(job_id)
    if not job:
        async with ws_lock:
            await websocket.send_json({"error": "Job not found"})
        await websocket.close()
        return

    video_path = job["video_path"]
    filter_classes = job.get("filter_classes", CLASSES)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    job["total_frames"] = total_frames
    cap.release()
    
    # Live streaming settings
    target_fps = 15
    frame_skip = max(1, round(fps / target_fps))
    jpeg_quality = 60  # Lower quality = faster encoding
    
    try:
        # Live streaming with per-frame delivery
        loop = asyncio.get_event_loop()
        
        # Create a queue for frame communication
        frame_queue = asyncio.Queue()
        
        def process_video_stream():
            """Process video and stream frames live at target FPS"""
            # Load a fresh model instance per thread to avoid conflicts
            try:
                thread_model = YOLO(str(BEST_MODEL_PATH))
                thread_model.to('cuda')
                device = 'cuda'
            except Exception as e:
                print(f"Thread {job_id}: CUDA unavailable ({e}), using CPU")
                thread_model = YOLO(str(BEST_MODEL_PATH))
                device = 'cpu'
            
            # Robust unique counting:
            # - track_to_object maps volatile tracker IDs to stable object IDs
            # - stable_objects stores last bbox per stable object for re-association
            unique_objects = {}   # {stable_object_id: class_name}
            track_to_object = {}  # {track_id: stable_object_id}
            stable_objects = {}   # {stable_object_id: {"class_name": str, "bbox": np.ndarray, "last_seen": int}}
            next_object_id = 0
            reid_iou_threshold = 0.3
            reid_max_gap_frames = 100
            center_dist_factor = 2.0
            
            # JPEG encoding params for speed
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

            def compute_iou(box_a, box_b):
                ax1, ay1, ax2, ay2 = box_a
                bx1, by1, bx2, by2 = box_b
                inter_x1 = max(ax1, bx1)
                inter_y1 = max(ay1, by1)
                inter_x2 = min(ax2, bx2)
                inter_y2 = min(ay2, by2)
                inter_w = max(0.0, inter_x2 - inter_x1)
                inter_h = max(0.0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
                area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                union = area_a + area_b - inter_area
                return inter_area / union if union > 0 else 0.0

            def box_center(box):
                x1, y1, x2, y2 = box
                return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

            def box_diag(box):
                x1, y1, x2, y2 = box
                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)
                return (w * w + h * h) ** 0.5
            
            try:
                # Run YOLO tracking. If CUDA fails at runtime, retry on CPU.
                try:
                    track_results = thread_model.track(
                        source=video_path,
                        stream=True,
                        persist=True,
                        device=device,
                        vid_stride=frame_skip,
                        conf=0.4,  # Detection confidence threshold
                        verbose=False
                    )
                except Exception as track_err:
                    if device == 'cuda':
                        print(f"Thread {job_id}: CUDA tracking failed ({track_err}), retrying on CPU")
                        cpu_model = YOLO(str(BEST_MODEL_PATH))
                        device = 'cpu'
                        track_results = cpu_model.track(
                            source=video_path,
                            stream=True,
                            persist=True,
                            device=device,
                            vid_stride=frame_skip,
                            conf=0.4,
                            verbose=False
                        )
                    else:
                        raise
                
                for frame_idx, result in enumerate(track_results):
                    # Get original frame for custom annotation
                    frame = result.orig_img.copy()
                    claimed_stable_ids = set()
                    
                    # Process and filter detections
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                        confs = result.boxes.conf.cpu().numpy()
                        track_ids_arr = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else None
                        
                        # Draw only selected class boxes
                        for i, (box, cls_id, conf) in enumerate(zip(boxes, cls_ids, confs)):
                            if cls_id < len(CLASSES):
                                class_name = CLASSES[cls_id]
                                
                                # Skip if class not in filter
                                if class_name not in filter_classes:
                                    continue
                                
                                # Get tracking ID if available
                                track_id = int(track_ids_arr[i]) if track_ids_arr is not None else None
                                
                                # Register unique object with tracker-ID re-association guard
                                if track_id is not None:
                                    if track_id in track_to_object:
                                        stable_id = track_to_object[track_id]
                                        stable_objects[stable_id]["bbox"] = box
                                        stable_objects[stable_id]["last_seen"] = frame_idx
                                        claimed_stable_ids.add(stable_id)
                                    else:
                                        # Tracker ID may switch mid-video. Match existing stable
                                        # object by IoU OR center distance (helps far/small objects).
                                        matched_stable_id = None
                                        best_score = -1.0
                                        cx, cy = box_center(box)
                                        diag = box_diag(box)
                                        for stable_id, obj in stable_objects.items():
                                            if stable_id in claimed_stable_ids:
                                                continue
                                            if obj["class_name"] != class_name:
                                                continue
                                            if frame_idx - obj["last_seen"] > reid_max_gap_frames:
                                                continue
                                            prev_box = obj["bbox"]
                                            iou = compute_iou(box, prev_box)
                                            pcx, pcy = box_center(prev_box)
                                            center_dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
                                            near_enough = center_dist <= (center_dist_factor * diag)
                                            if iou >= reid_iou_threshold or near_enough:
                                                score = iou - (center_dist / (center_dist_factor * diag + 1e-6))
                                                if score > best_score:
                                                    best_score = score
                                                    matched_stable_id = stable_id

                                        if matched_stable_id is None:
                                            matched_stable_id = next_object_id
                                            next_object_id += 1
                                            unique_objects[matched_stable_id] = class_name
                                            stable_objects[matched_stable_id] = {
                                                "class_name": class_name,
                                                "bbox": box,
                                                "last_seen": frame_idx,
                                            }
                                        else:
                                            stable_objects[matched_stable_id]["bbox"] = box
                                            stable_objects[matched_stable_id]["last_seen"] = frame_idx

                                        track_to_object[track_id] = matched_stable_id
                                        claimed_stable_ids.add(matched_stable_id)
                                
                                # Get class-specific color
                                color = CLASS_COLORS.get(class_name, (0, 255, 0))  # Default green
                                
                                # Draw bounding box
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                
                                # Draw label with adaptive text color for contrast
                                label = f"{class_name} #{track_id}" if track_id else f"{class_name} {conf:.2f}"
                                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                                # Compute brightness from BGR and choose dark/light text
                                b, g, r = color
                                brightness = 0.114 * b + 0.587 * g + 0.299 * r
                                text_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)
                                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
                    
                    # Encode frame with lower quality for speed
                    _, buffer = cv2.imencode('.jpg', frame, encode_params)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Count unique objects by class
                    current_counts = Counter(unique_objects.values())
                    
                    processed_frames = min(total_frames, (frame_idx + 1) * frame_skip)
                    progress = int((processed_frames / total_frames) * 100) if total_frames > 0 else 0
                    
                    asyncio.run_coroutine_threadsafe(
                        frame_queue.put({
                            "frame": frame_base64,
                            "progress": progress,
                            "counts": dict(current_counts),
                            "status": "processing",
                            "target_fps": target_fps
                        }),
                        loop
                    )
                
                # Signal completion with final unique counts
                final_counts = Counter(unique_objects.values())
                asyncio.run_coroutine_threadsafe(
                    frame_queue.put({
                        "status": "completed",
                        "counts": dict(final_counts)
                    }),
                    loop
                )
                
            except Exception as thread_err:
                print(f"Error in tracking thread {job_id}: {thread_err}")
                asyncio.run_coroutine_threadsafe(
                    frame_queue.put({"error": str(thread_err)}),
                    loop
                )
        
        # Start processing in background thread
        loop.run_in_executor(executor, process_video_stream)
        
        # Stream frames as they arrive in the queue
        while True:
            frame_data = await frame_queue.get()
            
            # Check for errors
            if "error" in frame_data:
                try:
                    async with ws_lock:
                        await websocket.send_json({"error": frame_data["error"]})
                except:
                    pass
                break
            
            # Check if completed
            if frame_data.get("status") == "completed":
                try:
                    async with ws_lock:
                        await websocket.send_json(frame_data)
                except:
                    pass
                break
            
            # Send frame
            try:
                async with ws_lock:
                    await websocket.send_json(frame_data)
            except Exception as send_err:
                print(f"Error sending frame for job {job_id}: {send_err}")
                break

    except WebSocketDisconnect:
        print(f"Client disconnected for job {job_id}")
    except Exception as e:
        print(f"Error processing job {job_id}: {e}")
        try:
            async with ws_lock:
                await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
