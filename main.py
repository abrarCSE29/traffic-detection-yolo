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

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Request, Form
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
            
            # Track unique objects by their tracking ID
            unique_objects = {}  # {track_id: class_name}
            
            # JPEG encoding params for speed
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            
            try:
                # Run YOLO tracking with fresh model instance
                track_results = thread_model.track(
                    source=video_path, 
                    stream=True, 
                    persist=True, 
                    device=device,
                    vid_stride=frame_skip,
                    conf=0.25,  # Detection confidence threshold
                    verbose=False
                )
                
                for frame_idx, result in enumerate(track_results):
                    # Get original frame for custom annotation
                    frame = result.orig_img.copy()
                    
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
                                
                                # Register unique object
                                if track_id is not None and track_id not in unique_objects:
                                    unique_objects[track_id] = class_name
                                
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
