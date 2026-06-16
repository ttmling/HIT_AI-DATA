from pathlib import Path

from utils import read_json, write_json


def _load_blip(device: str = "cpu"):
    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
        import torch
    except ImportError:
        return None, None, None

    import torch

    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name) # convert image → tensor input cho model
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model, torch

### Sinh caption cho một ảnh bằng BLIP.
def generate_caption(image_path: Path, processor, model, torch, device: str = "cpu") -> str:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


# load qua các keyframes của video để sinh caption cho từng frame
def run_caption_for_video(
    video_output_dir: Path,
    device: str = "cpu",
) -> list[dict]:
    
    frames_metadata_path = video_output_dir / "frames_metadata.json"
    frames_metadata = read_json(frames_metadata_path, [])

    processor, model, torch = _load_blip(device)

    if processor is None:
        for frame in frames_metadata:
            frame["caption"] = {
                "status": "skipped",
                "text": "",
                "model": None,
                "reason": "transformers is not installed",
            }
        write_json(frames_metadata_path, frames_metadata)
        return frames_metadata


    MODEL_NAME = "Salesforce/blip-image-captioning-base"


    for frame in frames_metadata:
        if frame.get("caption", {}).get("status") == "done":
            continue

        image_path = Path(frame["image_path"])
        try:
            caption_text = generate_caption(image_path, processor, model, torch, device)
            frame["caption"] = {
                "status": "done",
                "text": caption_text,
                "model": MODEL_NAME,
            }
        except Exception as e:
            frame["caption"] = {
                "status": "error",
                "text": "",
                "model": MODEL_NAME,
                "reason": str(e),
            }

    write_json(frames_metadata_path, frames_metadata)
    return frames_metadata
