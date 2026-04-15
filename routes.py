import cv2
import uuid
import asyncio
import threading
from typing import Dict

from fastapi import APIRouter, UploadFile, File, WebSocket, WebSocketDisconnect, Form, HTTPException
from fastapi.responses import HTMLResponse

from config import CLASSES, UPLOAD_FOLDER, DEMO_VIDEO_PATH, TARGET_FPS, JPEG_QUALITY
from detector import process_video_stream, executor

router = APIRouter()

processing_jobs: Dict[str, dict] = {}


def _parse_classes(selected: str) -> list:
    if selected == "all":
        return list(CLASSES)
    if not selected:
        return []
    return [c.strip() for c in selected.split(",") if c.strip()]


@router.get("/", response_class=HTMLResponse)
async def get_index():
    with open("templates/index.html") as f:
        return f.read()


@router.get("/classes")
async def get_classes():
    return {"classes": CLASSES}


@router.post("/upload")
async def upload_video(video: UploadFile = File(...), selected_classes: str = Form("all")):
    job_id = str(uuid.uuid4())
    file_path = UPLOAD_FOLDER / f"{job_id}_{video.filename}"
    with open(file_path, "wb") as buf:
        buf.write(await video.read())

    processing_jobs[job_id] = {
        "status":        "ready",
        "video_path":    str(file_path),
        "total_frames":  0,
        "counts":        {},
        "filter_classes": _parse_classes(selected_classes),
    }
    return {"job_id": job_id}


@router.post("/demo-job")
async def create_demo_job(selected_classes: str = Form("all")):
    if not DEMO_VIDEO_PATH.exists():
        raise HTTPException(status_code=404, detail="Demo video not found")

    job_id = str(uuid.uuid4())
    processing_jobs[job_id] = {
        "status":        "ready",
        "video_path":    str(DEMO_VIDEO_PATH),
        "total_frames":  0,
        "counts":        {},
        "filter_classes": _parse_classes(selected_classes),
    }
    return {"job_id": job_id, "video_name": DEMO_VIDEO_PATH.name}


@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    ws_lock = asyncio.Lock()

    job = processing_jobs.get(job_id)
    if not job:
        await websocket.send_json({"error": "Job not found"})
        await websocket.close()
        return

    cap = cv2.VideoCapture(job["video_path"])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    cap.release()
    job["total_frames"] = total_frames

    frame_skip  = max(1, round(source_fps / TARGET_FPS))
    loop        = asyncio.get_event_loop()
    stop_event  = threading.Event()
    frame_queue = asyncio.Queue()

    try:
        detection_future = loop.run_in_executor(
            executor,
            process_video_stream,
            job_id,
            job["video_path"],
            total_frames,
            frame_skip,
            stop_event,
            frame_queue,
            loop,
            processing_jobs,
        )

        async def receiver():
            try:
                while True:
                    data = await websocket.receive_json()
                    if data.get("type") == "filter_update":
                        job["filter_classes"] = data.get("filter_classes", [])
            except WebSocketDisconnect:
                stop_event.set()
            except Exception as e:
                print(f"Receiver error [{job_id}]: {e}")

        async def sender():
            while not stop_event.is_set():
                try:
                    item = await frame_queue.get()

                    if "error" in item:
                        try:
                            await websocket.send_json({"error": item["error"]})
                        except Exception:
                            pass
                        break

                    if item.get("status") == "completed":
                        try:
                            await websocket.send_json(item)
                        except Exception:
                            pass
                        break

                    async with ws_lock:
                        if item.get("type") == "metadata":
                            await websocket.send_json(item)
                        elif item.get("type") == "frame":
                            await websocket.send_bytes(item["data"])

                except Exception as e:
                    print(f"Sender error [{job_id}]: {e}")
                    break
            stop_event.set()

        await asyncio.gather(receiver(), sender())

    except WebSocketDisconnect:
        stop_event.set()
    except Exception as e:
        stop_event.set()
        try:
            async with ws_lock:
                await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        stop_event.set()
        try:
            detection_future.cancel()
        except Exception:
            pass
        await websocket.close()
