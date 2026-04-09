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
        filter_classes = list(CLASSES)
    elif not selected_classes:
        filter_classes = []
    else:
        filter_classes = [c.strip() for c in selected_classes.split(",") if c.strip()]

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
        filter_classes = list(CLASSES)
    elif not selected_classes:
        filter_classes = []
    else:
        filter_classes = [c.strip() for c in selected_classes.split(",") if c.strip()]

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
            """Inference Thread: Handles YOLO + ByteTrack only"""
            try:
                thread_model = YOLO(str(BEST_MODEL_PATH))
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if device == 'cuda': thread_model.to('cuda')
            except Exception as e:
                print(f"Thread {job_id}: Init failed ({e}), using CPU")
                thread_model = YOLO(str(BEST_MODEL_PATH)); device = 'cpu'
            
            seen_track_ids = set()
            class_counts = Counter()
            track_history = {}
            PX_TO_METER = 0.05
            SPEED_WINDOW = 10
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            
            try:
                # Optimized track call: imgsz=480 for speed boost
                track_results = thread_model.track(
                    source=video_path, stream=True, persist=True, device=device,
                    vid_stride=frame_skip, conf=0.3, iou=0.5,
                    tracker="bytetrack.yaml", verbose=False,
                    imgsz=480, # Lower resolution inference
                    half=(device == 'cuda')
                )
                
                for frame_idx, result in enumerate(track_results):
                    t_start = time.time()
                    if stop_event.is_set(): break

                    current_filters = processing_jobs.get(job_id, {}).get("filter_classes", [])
                    frame = result.orig_img # No copy here, do it in post-process if needed
                    current_frame_counts = Counter()
                    current_speeds = []
                    boxes_to_draw = []
                    
                    if result.boxes is not None and result.boxes.id is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                        track_ids = result.boxes.id.cpu().numpy().astype(int)
                        
                        for i, (box, cls_id, track_id) in enumerate(zip(boxes, cls_ids, track_ids)):
                            if cls_id < len(CLASSES):
                                class_name = CLASSES[cls_id]
                                if class_name not in current_filters: continue
                                
                                current_frame_counts[class_name] += 1
                                if track_id not in seen_track_ids:
                                    seen_track_ids.add(track_id); class_counts[class_name] += 1
                                
                                x1, y1, x2, y2 = map(int, box)
                                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                if track_id not in track_history: track_history[track_id] = []
                                track_history[track_id].append((cx, cy, time.time()))
                                
                                speed_kmh = 0
                                if len(track_history[track_id]) > SPEED_WINDOW:
                                    start_pt, end_pt = track_history[track_id][-SPEED_WINDOW], track_history[track_id][-1]
                                    pixel_dist = ((end_pt[0]-start_pt[0])**2 + (end_pt[1]-start_pt[1])**2)**0.5
                                    time_diff = end_pt[2] - start_pt[2]
                                    if time_diff > 0:
                                        speed_kmh = ((pixel_dist * PX_TO_METER) / time_diff) * 3.6
                                        current_speeds.append(speed_kmh)
                                    if len(track_history[track_id]) > 20: track_history[track_id].pop(0)
                                
                                boxes_to_draw.append({
                                    "box": [x1, y1, x2, y2],
                                    "label": f"{class_name} #{track_id}{f' {int(speed_kmh)} km/h' if speed_kmh > 2 else ''}",
                                    "color": CLASS_COLORS.get(class_name, (0, 255, 0))
                                })
                    
                    # Offload Visualization and Encoding to Post-Process logic
                    # We reuse the frame_queue for metadata but handle frame separately or together
                    # To keep it simple and fast, we'll do encoding here but move it after the 'next' result is requested
                    
                    def visualize_and_encode(f, b_list):
                        for b in b_list:
                            x1, y1, x2, y2 = b["box"]
                            cv2.rectangle(f, (x1, y1), (x2, y2), b["color"], 2)
                            (w, h), _ = cv2.getTextSize(b["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                            cv2.rectangle(f, (x1, y1 - 20), (x1 + w, y1), b["color"], -1)
                            b_val, g, r = b["color"]
                            text_color = (0, 0, 0) if (0.114*b_val + 0.587*g + 0.299*r) > 140 else (255, 255, 255)
                            cv2.putText(f, b["label"], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
                        _, buf = cv2.imencode('.jpg', f, encode_params)
                        return buf.tobytes()

                    # Calculate metrics BEFORE creating payload
                    processed_frames = min(total_frames, (frame_idx + 1) * frame_skip)
                    progress = int((processed_frames / total_frames) * 100) if total_frames > 0 else 0
                    inf_ms = int((time.time() - t_start) * 1000)
                    avg_speed = sum(current_speeds)/len(current_speeds) if current_speeds else 0

                    m_payload = {
                        "type": "metadata", "progress": progress,
                        "current_counts": dict(current_frame_counts),
                        "cumulative_counts": dict(class_counts),
                        "avg_speed": round(avg_speed, 1),
                        "status": "processing", "target_fps": target_fps,
                        "inference_time": inf_ms
                    }
                    
                    # Execute visualization in parallel
                    def post_process_task(f, b_list, m_data):
                        try:
                            f_bytes = visualize_and_encode(f, b_list)
                            asyncio.run_coroutine_threadsafe(frame_queue.put(m_data), loop)
                            asyncio.run_coroutine_threadsafe(frame_queue.put({"type": "frame", "data": f_bytes}), loop)
                        except Exception as e:
                            print(f"Post-process error: {e}")

                    # Use the outer executor or a dedicated one for encoding
                    executor.submit(post_process_task, frame.copy(), boxes_to_draw, m_payload)

                    if stop_event.is_set():
                        break
                
                # Signal completion with final unique counts
                if not stop_event.is_set():
                    asyncio.run_coroutine_threadsafe(
                        frame_queue.put({
                            "status": "completed",
                            "counts": dict(class_counts)
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
        
        async def receiver():
            try:
                while True:
                    data = await websocket.receive_json()
                    if data.get("type") == "filter_update":
                        new_filters = data.get("filter_classes", [])
                        job["filter_classes"] = new_filters
                        print(f"Job {job_id}: Updated filters to {new_filters}")
            except WebSocketDisconnect:
                stop_event.set()
            except Exception as e:
                print(f"Receiver error: {e}")

        async def sender():
            while not stop_event.is_set():
                try:
                    frame_data = await frame_queue.get()
                    
                    if "error" in frame_data:
                        print(f"Job {job_id}: Sender received error from queue: {frame_data['error']}")
                        try: await websocket.send_json({"error": frame_data["error"]})
                        except: pass
                        break
                    
                    if frame_data.get("status") == "completed":
                        print(f"Job {job_id}: Sender received completion signal")
                        try: await websocket.send_json(frame_data)
                        except: pass
                        break
                    
                    async with ws_lock:
                        if frame_data.get("type") == "metadata":
                            await websocket.send_json(frame_data)
                        elif frame_data.get("type") == "frame":
                            await websocket.send_bytes(frame_data["data"])
                except Exception as send_err:
                    print(f"Job {job_id}: Sender error: {send_err}")
                    break
            stop_event.set()

        # Run both concurrently
        await asyncio.gather(receiver(), sender())

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
