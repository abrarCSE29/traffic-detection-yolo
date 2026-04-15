import cv2
import time
import asyncio
import torch
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from ultralytics import YOLO

from config import (
    BEST_MODEL_PATH, CLASSES, CLASS_COLORS,
    CONF_THRESHOLD, IOU_THRESHOLD, INFERENCE_SIZE,
    PX_TO_METER, SPEED_WINDOW, TRACK_HISTORY_MAX, SPEED_MIN_KMH,
    TARGET_FPS, JPEG_QUALITY,
)

executor = ThreadPoolExecutor(max_workers=4)


def visualize_and_encode(frame, boxes_to_draw: list, jpeg_quality: int = JPEG_QUALITY) -> bytes:
    """Draw bounding boxes on frame and return JPEG-encoded bytes."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    for b in boxes_to_draw:
        x1, y1, x2, y2 = b["box"]
        color = b["color"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (w, _), _ = cv2.getTextSize(b["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        b_val, g, r = color
        text_color = (0, 0, 0) if (0.114 * b_val + 0.587 * g + 0.299 * r) > 140 else (255, 255, 255)
        cv2.putText(frame, b["label"], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
    _, buf = cv2.imencode(".jpg", frame, encode_params)
    return buf.tobytes()


def process_video_stream(
    job_id: str,
    video_path: str,
    total_frames: int,
    frame_skip: int,
    stop_event,
    frame_queue,
    loop,
    processing_jobs: dict,
):
    """
    Inference thread: runs YOLO + ByteTrack on every nth frame and pushes
    (metadata JSON, annotated JPEG) pairs onto frame_queue.
    """
    try:
        thread_model = YOLO(str(BEST_MODEL_PATH))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            thread_model.to("cuda")
    except Exception as e:
        print(f"Thread {job_id}: init failed ({e}), falling back to CPU")
        thread_model = YOLO(str(BEST_MODEL_PATH))
        device = "cpu"

    seen_track_ids: set = set()
    class_counts: Counter = Counter()
    track_history: dict = {}

    def _push(item):
        asyncio.run_coroutine_threadsafe(frame_queue.put(item), loop)

    def post_process(frame_copy, boxes, metadata):
        try:
            frame_bytes = visualize_and_encode(frame_copy, boxes)
            _push(metadata)
            _push({"type": "frame", "data": frame_bytes})
        except Exception as e:
            print(f"Post-process error: {e}")

    try:
        track_results = thread_model.track(
            source=video_path,
            stream=True,
            persist=True,
            device=device,
            vid_stride=frame_skip,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            tracker="bytetrack.yaml",
            verbose=False,
            imgsz=INFERENCE_SIZE,
            half=(device == "cuda"),
        )

        for frame_idx, result in enumerate(track_results):
            t_start = time.time()
            if stop_event.is_set():
                break

            current_filters = processing_jobs.get(job_id, {}).get("filter_classes", [])
            current_frame_counts: Counter = Counter()
            current_speeds: list = []
            current_class_speeds: dict = {}
            boxes_to_draw: list = []

            if result.boxes is not None and result.boxes.id is not None:
                boxes   = result.boxes.xyxy.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                track_ids = result.boxes.id.cpu().numpy().astype(int)

                for box, cls_id, track_id in zip(boxes, cls_ids, track_ids):
                    if cls_id >= len(CLASSES):
                        continue
                    class_name = CLASSES[cls_id]
                    if class_name not in current_filters:
                        continue

                    current_frame_counts[class_name] += 1
                    if track_id not in seen_track_ids:
                        seen_track_ids.add(track_id)
                        class_counts[class_name] += 1

                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    track_history.setdefault(track_id, []).append((cx, cy, time.time()))

                    speed_kmh = 0.0
                    history = track_history[track_id]
                    if len(history) > SPEED_WINDOW:
                        s, e = history[-SPEED_WINDOW], history[-1]
                        pixel_dist = ((e[0] - s[0]) ** 2 + (e[1] - s[1]) ** 2) ** 0.5
                        dt = e[2] - s[2]
                        if dt > 0:
                            speed_kmh = ((pixel_dist * PX_TO_METER) / dt) * 3.6
                            current_speeds.append(speed_kmh)
                            if speed_kmh > SPEED_MIN_KMH:
                                current_class_speeds.setdefault(class_name, []).append(speed_kmh)
                        if len(history) > TRACK_HISTORY_MAX:
                            history.pop(0)

                    label = f"{class_name} #{track_id}"
                    if speed_kmh > SPEED_MIN_KMH:
                        label += f" {int(speed_kmh)} km/h"
                    boxes_to_draw.append({
                        "box":   [x1, y1, x2, y2],
                        "label": label,
                        "color": CLASS_COLORS.get(class_name, (0, 255, 0)),
                    })

            processed_frames = min(total_frames, (frame_idx + 1) * frame_skip)
            progress  = int((processed_frames / total_frames) * 100) if total_frames > 0 else 0
            avg_speed = sum(current_speeds) / len(current_speeds) if current_speeds else 0
            class_avg_speeds = {
                cls: round(sum(speeds) / len(speeds), 1)
                for cls, speeds in current_class_speeds.items()
            }

            metadata = {
                "type":              "metadata",
                "status":            "processing",
                "progress":          progress,
                "current_counts":    dict(current_frame_counts),
                "cumulative_counts": dict(class_counts),
                "avg_speed":         round(avg_speed, 1),
                "class_speeds":      class_avg_speeds,
                "target_fps":        TARGET_FPS,
                "inference_time":    int((time.time() - t_start) * 1000),
            }

            executor.submit(post_process, result.orig_img.copy(), boxes_to_draw, metadata)

            if stop_event.is_set():
                break

        if not stop_event.is_set():
            _push({"status": "completed", "counts": dict(class_counts)})

    except Exception as err:
        print(f"Tracking thread {job_id} error: {err}")
        _push({"error": str(err)})
