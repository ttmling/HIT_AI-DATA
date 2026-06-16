from pathlib import Path

from utils import read_json, write_json


def normalize_bbox(bbox) -> list[list[float]]:
    """Chuyen toa do bbox cua EasyOCR ve kieu float de ghi JSON khong loi."""
    return [
        [float(pt[0]), float(pt[1])]
        for pt in bbox
    ]


def _load_easyocr_reader(languages: list[str]):
    """
    Neu chua cai EasyOCR, ham nay tra ve None de pipeline bao trang thai
    skipped thay vi crash kho hieu.
    """
    try:
        import easyocr
    except ImportError:
        return None
    return easyocr.Reader(languages, gpu=False)


def run_ocr_for_video(video_output_dir: Path, languages: list[str]) -> list[dict]:
    
    frames_metadata_path = video_output_dir / "frames_metadata.json"
    frames_metadata = read_json(frames_metadata_path, [])

    reader = _load_easyocr_reader(languages)
    if reader is None:
        for frame in frames_metadata:
            frame["ocr"] = {
                "status": "skipped",
                "text": "",
                "avg_confidence": None,
                "boxes": [],
                "reason": "easyocr is not installed",
            }
        write_json(frames_metadata_path, frames_metadata)
        return frames_metadata

    for frame in frames_metadata:
        image_path = Path(frame["image_path"])
        results = reader.readtext(str(image_path))

        boxes = []
        texts = []
        confidences = []

        for bbox, text, confidence in results:
            boxes.append(
                {
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": normalize_bbox(bbox),
                }
            )
            texts.append(text)
            confidences.append(float(confidence))

        frame["ocr"] = {
            "status": "done",
            "text": "\n".join(texts),
            "avg_confidence": (
                sum(confidences) / len(confidences) if confidences else None
            ),
            "boxes": boxes,
        }

    write_json(frames_metadata_path, frames_metadata)
    return frames_metadata