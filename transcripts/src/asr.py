from pathlib import Path

from utils import seconds_to_time_text, write_json


def run_asr_for_video(video_path: Path, video_output_dir: Path, video_id: str, model_size: str) -> dict:
    """Chuyen loi noi trong video thanh text va ghi transcript.json.
    """
    transcript_path = video_output_dir / "transcript.json"

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        transcript = {
            "video_id": video_id,
            "status": "skipped",
            "language": None,
            "asr_model": None,
            "reason": str(e),
            "segments": [],
        }
        write_json(transcript_path, transcript)
        return transcript

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(video_path), beam_size=5)

    transcript_segments = []
    for index, segment in enumerate(segments, start=1):
        
        # Gom tất cả độ tự tin của từng từ trong câu này lại
        word_confidences = []
        if segment.words:
            for word in segment.words:
                word_confidences.append(word.probability) # probability chính là độ tự tin của từ đó
        
        # Tính trung bình cộng độ tự tin của cả câu
        if word_confidences:
            avg_confidence = round(sum(word_confidences) / len(word_confidences), 4)
        else:
            avg_confidence = None 
            
        transcript_segments.append(
            {
                "segment_id": f"{video_id}_s{index:06d}",
                "start": float(segment.start),
                "end": float(segment.end),
                "start_text": seconds_to_time_text(segment.start),
                "end_text": seconds_to_time_text(segment.end),
                "text": segment.text.strip(),
                "confidence": avg_confidence
            }
        )

    transcript = {
        "video_id": video_id,
        "status": "done",
        "language": info.language,
        "asr_model": f"faster-whisper/{model_size}",
        "segments": transcript_segments,
    }
    write_json(transcript_path, transcript)
    return transcript