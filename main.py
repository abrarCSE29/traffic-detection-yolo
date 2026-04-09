import os
import cv2
import json
import uuid
import base64
import asyncio
import numpy as np
import threading
import time
import torch
from pathlib import Path
from collections import Counter
from typing import Dict

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Request, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

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
        stop_event = threading.Event()
        
        # Create a queue for frame communication
        frame_queue = asyncio.Queue()
        
        def process_video_stream():
            """Process video and stream frames live at target FPS"""
            try:
                thread_model = YOLO(str(BEST_MODEL_PATH))
                if torch.cuda.is_available():
                    thread_model.to('cuda')
                    device = 'cuda'
                else:
                    device = 'cpu'
            except Exception as e:
                print(f"Thread {job_id}: Device init failed ({e}), using CPU")
                thread_model = YOLO(str(BEST_MODEL_PATH))
                device = 'cpu'
            
            # Cumulative unique counts using ByteTrack IDs
            seen_track_ids = set()
            class_counts = Counter()
            
            # JPEG encoding params for speed
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            
            try:
                # Run YOLO with ByteTrack enabled
                try:
                    track_results = thread_model.track(
                        source=video_path,
                        stream=True,
                        persist=True,
                        device=device,
                        vid_stride=frame_skip,
                        conf=0.3,
                        iou=0.5,
                        tracker="bytetrack.yaml", # Native ByteTrack integration
                        verbose=False,
                        half=True if device == 'cuda' else False
                    )
                except Exception as track_err:
                    print(f"Tracking failed: {track_err}")
                    raise
                
                for frame_idx, result in enumerate(track_results):
                    t_start = time.time()
                    if stop_event.is_set(): break

                    frame = result.orig_img.copy()
                    current_frame_counts = Counter() # To track objects in THIS frame
                    
                    if result.boxes is not None and result.boxes.id is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                        track_ids = result.boxes.id.cpu().numpy().astype(int)
                        
                        for i, (box, cls_id, track_id) in enumerate(zip(boxes, cls_ids, track_ids)):
                            if cls_id < len(CLASSES):
                                class_name = CLASSES[cls_id]
                                if class_name not in filter_classes: continue
                                
                                # 1. Update per-frame count
                                current_frame_counts[class_name] += 1

                                # 2. Update cumulative unique count
                                if track_id not in seen_track_ids:
                                    seen_track_ids.add(track_id)
                                    class_counts[class_name] += 1
                                
                                color = CLASS_COLORS.get(class_name, (0, 255, 0))
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                
                                label = f"{class_name} #{track_id}"
                                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                                b, g, r = color
                                brightness = 0.114 * b + 0.587 * g + 0.299 * r
                                text_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)
                                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
                    
                    _, buffer = cv2.imencode('.jpg', frame, encode_params)
                    frame_bytes = buffer.tobytes()
                    
                    processed_frames = min(total_frames, (frame_idx + 1) * frame_skip)
                    progress = int((processed_frames / total_frames) * 100) if total_frames > 0 else 0
                    inf_ms = int((time.time() - t_start) * 1000)

                    asyncio.run_coroutine_threadsafe(
                        frame_queue.put({
                            "type": "metadata",
                            "progress": progress,
                            "current_counts": dict(current_frame_counts),
                            "cumulative_counts": dict(class_counts),
                            "status": "processing",
                            "target_fps": target_fps,
                            "inference_time": inf_ms
                        }),
                        loop
                    )
                    asyncio.run_coroutine_threadsafe(
                        frame_queue.put({"type": "frame", "data": frame_bytes}),
                        loop
                    )

                    if stop_event.is_set():
                        break
                
                # Signal completion with final unique counts
                final_counts = Counter(unique_objects.values())
                if not stop_event.is_set():
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
        detection_future = loop.run_in_executor(executor, process_video_stream)
        
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
            
            # Send data based on type
            try:
                async with ws_lock:
                    if frame_data.get("type") == "metadata":
                        await websocket.send_json(frame_data)
                    elif frame_data.get("type") == "frame":
                        await websocket.send_bytes(frame_data["data"])
            except Exception as send_err:
                print(f"Error sending data for job {job_id}: {send_err}")
                break

    except WebSocketDisconnect:
        stop_event.set()
        print(f"Client disconnected for job {job_id}")
    except Exception as e:
        stop_event.set()
        print(f"Error processing job {job_id}: {e}")
        try:
            async with ws_lock:
                await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        stop_event.set()
        try:
            detection_future.cancel()
        except Exception:
            pass
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
