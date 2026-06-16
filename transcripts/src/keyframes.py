from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel
from utils import ensure_dir, get_video_duration, run_command, seconds_to_time_text, write_json


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"
PROCESSOR = CLIPProcessor.from_pretrained(MODEL_NAME)
MODEL = CLIPVisionModel.from_pretrained(MODEL_NAME).to(DEVICE)

### cắt video-->raw frames
def extract_raw_frames(video_path: Path, frames_dir: Path, interval_seconds: int) -> list[Path]:
    ensure_dir(frames_dir)
    output_pattern = frames_dir / "raw_frame_%06d.jpg"

    command = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps=1/{interval_seconds}",
        "-q:v", "2", str(output_pattern),
    ]
    run_command(command)
    return sorted(frames_dir.glob("raw_frame_*.jpg"))



### Dùng mô hình CLIP để trích xuất Vector đặc trưng (Feature Vector) từ ảnh.
def extract_features(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    inputs = PROCESSOR(images=image, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad(): 
        outputs = MODEL(**inputs)
        image_features = outputs.pooler_output   # Lấy vector đặc trưng ở tầng cuối cùng (đại diện cho toàn bộ nội dung ảnh)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # Chuẩn hóa vector về độ dài bằng 1 để tính toán độ tương đồng chính xác hơn
    
    return image_features



### Loại frame trùng
def remove_duplicate_frames(
    raw_frames: list[Path],
    frames_dir: Path,
    video_id: str,
    interval_seconds: int,
    duplicate_threshold: float, # Ngưỡng tương đồng 
) -> list[dict]:
    frames_metadata = []
    saved_features = [] 
    
    selected_index = 0

    for raw_index, raw_path in enumerate(raw_frames, start=1):
        current_features = extract_features(raw_path)   # trích xuất Vector của ảnh hiện tại

        # So sánh với TẤT CẢ các ảnh đã có trong Database
        is_duplicate = False
        for saved_feature in saved_features:
            similarity = torch.nn.functional.cosine_similarity(current_features, saved_feature, dim=-1).item()   # Tính Cosine Similarity (Độ tương đồng góc giữa 2 vector)
            
            if similarity >= duplicate_threshold:
                is_duplicate = True
                break # Dừng kiểm tra, đánh dấu là trùng luôn

        if is_duplicate:
            raw_path.unlink(missing_ok=True) # Xóa ảnh trùng đi
            continue

        # Nếu là ảnh mới (không trùng)
        selected_index += 1
        
        # Thêm vector của ảnh mới này vào để làm căn cứ so sánh cho các ảnh sau
        saved_features.append(current_features)

        timestamp = float((raw_index - 1) * interval_seconds)
        frame_name = f"frame_{selected_index:06d}.jpg"
        final_path = frames_dir / frame_name
        raw_path.rename(final_path)

        frames_metadata.append({
            "frame_id": f"{video_id}_f{selected_index:06d}",
            "video_id": video_id,
            "frame_index": selected_index,
            "timestamp": timestamp,
            "timestamp_text": seconds_to_time_text(timestamp),
            "image_path": str(final_path).replace("\\", "/"),
            "ocr": {"status": "pending", "text": "", "avg_confidence": None, "boxes": []},
            "caption": {"status": "pending", "text": "", "model": None},
            "objects": {"status": "pending", "model": None, "detections": []},
        })

    return frames_metadata




def create_keyframes(
    video_path: Path,
    video_id: str,
    output_dir: Path,
    interval_seconds: int,
    duplicate_threshold: float, # Ngưỡng mặc định cho mô hình CLIP
    source_info: dict | None = None,
) -> list[dict]:
    
    video_output_dir = output_dir / video_id
    frames_dir = video_output_dir / "frames"
    ensure_dir(frames_dir)

    raw_frames = extract_raw_frames(video_path, frames_dir, interval_seconds)
    
    frames_metadata = remove_duplicate_frames(
        raw_frames=raw_frames,
        frames_dir=frames_dir,
        video_id=video_id,
        interval_seconds=interval_seconds,
        duplicate_threshold=duplicate_threshold,
    )

    # video metadata
    write_json(video_output_dir / "metadata.json", {
        "video_id": video_id,
        "filename": video_path.name,
        "source_path": str(video_path).replace("\\", "/"),
        "source": source_info or {"storage": "local"},
        "duration": get_video_duration(video_path),
        "frame_interval_seconds": interval_seconds,
        "selected_keyframes": len(frames_metadata),
    })

    # frame metadata
    write_json(video_output_dir / "frames_metadata.json", frames_metadata)
    return frames_metadata