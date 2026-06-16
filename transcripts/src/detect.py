from pathlib import Path

from utils import read_json, write_json


DEFAULT_CONFIDENCE_THRESHOLD = 0.5
MODEL_NAME = "yolov8n"

def _load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError:
        return None

    model = YOLO(f"{MODEL_NAME}.pt")
    return model


### Phát hiện vật thể trong một ảnh
def detect_objects(
    image_path: Path,
    model,
    confidence_threshold: float,
) -> list[dict]:
    results = model(str(image_path), verbose=False)[0]

    detections = []
    for box in results.boxes:
        confidence = float(box.conf[0])
        if confidence < confidence_threshold:
            continue

        label = results.names[int(box.cls[0])]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "label": label,
            "confidence": round(confidence, 4),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })

    return detections


# Chạy object detection cho tất cả keyframe của một video
def run_detection_for_video(
    video_output_dir: Path,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[dict]:
    """
    Đọc frames_metadata.json, phát hiện vật thể trong từng keyframe bằng YOLOv8n,
    ghi kết quả vào trường 'objects' rồi lưu lại.
    Bỏ qua frame đã có status=done.
    """
    frames_metadata_path = video_output_dir / "frames_metadata.json"
    frames_metadata = read_json(frames_metadata_path, [])

    model = _load_yolo()

    if model is None:
        for frame in frames_metadata:
            if frame.get("objects", {}).get("status") == "done":
                continue
            frame["objects"] = {
                "status": "skipped",
                "model": None,
                "detections": [],
                "reason": "ultralytics is not installed",
            }
        write_json(frames_metadata_path, frames_metadata)
        return frames_metadata

    for frame in frames_metadata:
        # Resume logic — bỏ qua frame đã detect xong
        if frame.get("objects", {}).get("status") == "done":
            continue

        image_path = Path(frame["image_path"])
        try:
            detections = detect_objects(image_path, model, confidence_threshold)
            frame["objects"] = {
                "status": "done",
                "model": MODEL_NAME,
                "detections": detections,
            }
        except Exception as e:
            frame["objects"] = {
                "status": "error",
                "model": MODEL_NAME,
                "detections": [],
                "reason": str(e),
            }

    write_json(frames_metadata_path, frames_metadata)
    return frames_metadata
