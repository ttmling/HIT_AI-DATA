from pathlib import Path

from asr import run_asr_for_video
from caption import run_caption_for_video
from keyframes import create_keyframes
from ocr import run_ocr_for_video
from utils import ensure_dir


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

FRAME_INTERVAL_SECONDS = 3  
DUPLICATE_THRESHOLD = 0.92
OCR_LANGUAGES = ["vi", "en"]
WHISPER_MODEL = "small"

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def make_video_id(video_name: str) -> str:
    """Tạo video_id đơn giản từ tên file video."""
    return Path(video_name).stem.lower().replace(" ", "_")


def find_videos_in_data_folder(data_dir: Path) -> list[Path]:
    """Tìm tất cả video trong folder data/."""

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy folder {data_dir}. Hãy tạo folder data/ và đặt video vào đó."
        )

    videos = [
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    ]

    if len(videos) == 0:
        raise FileNotFoundError(
            f"Folder {data_dir} chưa có video. "
            f"Hãy thêm các file .mp4/.mov/.mkv/.avi/.webm vào data/."
        )

    videos.sort(key=lambda p: p.name)
    return videos


def run_pipeline_for_video(video_path: Path) -> Path:

    video_id = make_video_id(video_path.name)
    video_output_dir = OUTPUT_DIR / video_id
    ensure_dir(video_output_dir)

    source_info = {
        "storage": "local",
        "source_path": str(video_path).replace("\\", "/"),
    }

    print(f"PROCESSING VIDEO: {video_path.name}")

    print("[1/3] Cắt keyframe từ video")
    create_keyframes(
        video_path=video_path,
        video_id=video_id,
        output_dir=OUTPUT_DIR,
        interval_seconds=FRAME_INTERVAL_SECONDS,
        duplicate_threshold=DUPLICATE_THRESHOLD,
        source_info=source_info,
    )

    print("[2/4] OCR trên keyframe")
    run_ocr_for_video(
        video_output_dir=video_output_dir,
        languages=OCR_LANGUAGES,
    )

    print("[3/4] Sinh caption tự động cho keyframe (BLIP)")
    run_caption_for_video(
        video_output_dir=video_output_dir,
    )

    print("[4/4] ASR (Whisper) chuyển audio -> text")
    run_asr_for_video(
        video_path=video_path,
        video_output_dir=video_output_dir,
        video_id=video_id,
        model_size=WHISPER_MODEL,
    )

    print(f"Hoàn thành video: {video_path.name}")
    print(f"Output: {video_output_dir}")
    return video_output_dir


def main() -> None:
    videos = find_videos_in_data_folder(DATA_DIR)
    print(f"Found {len(videos)} video(s) in data/")

    results = []

    for idx, video_path in enumerate(videos, 1):
        print(f"\n[{idx}/{len(videos)}] Starting pipeline...")
        output_dir = run_pipeline_for_video(video_path)
        results.append(output_dir)

    print("PROCESSED:")
    for r in results:
        print(f"- {r}")


if __name__ == "__main__":
    main()