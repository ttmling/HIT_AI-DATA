import json
import subprocess
from pathlib import Path


def ensure_dir(path: Path) -> None:
    """Tao folder neu folder chua ton tai."""
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default):
    """Doc file JSON. Neu file chua ton tai thi tra ve gia tri mac dinh."""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def seconds_to_time_text(seconds: float) -> str:
    """Chuyen so giay, vi du 65.2, thanh dang 00:01:05."""
    total = int(round(seconds))
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def run_command(command: list[str]) -> None:
    """Chay lenh ben ngoai, vi du ffmpeg, va bao loi ro neu that bai."""
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(command)
            + "\n\nSTDERR:\n"
            + result.stderr
        )


def get_video_duration(video_path: Path) -> float:
    """Lay do dai video tinh bang giay bang ffprobe."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    return float(result.stdout.strip())