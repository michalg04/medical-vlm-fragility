# Shared VLM inference utilities.
# Handles the quirks of each model family so callers don't have to.
import gc
import io

import torch
from PIL import Image as PILImage
from transformers import pipeline

MAX_NEW_TOKENS = 2048
DO_SAMPLE = False


def make_pipe(model_id: str):
    dtype  = torch.bfloat16 if torch.cuda.is_available() else None
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("image-text-to-text", model=model_id, device=device, dtype=dtype)


def free_pipe(pipe) -> None:
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def to_pil(image) -> PILImage.Image:
    # Normalize whatever HuggingFace gives us into a PIL Image.
    if isinstance(image, PILImage.Image):
        return image.convert("RGB")
    if isinstance(image, dict):
        if image.get("bytes"):
            return PILImage.open(io.BytesIO(image["bytes"])).convert("RGB")
        if image.get("path"):
            return PILImage.open(image["path"]).convert("RGB")
    raise TypeError(f"Cannot convert image of type {type(image)} to PIL Image")


def _ensure_llava_patch_size(pipe, fallback: int = 14) -> None:
    # Some LLaVA conversions ship with patch_size=None in the processor.
    # Patch it from the model config or fall back to 14.
    proc = getattr(pipe, "processor", None)
    if proc is None:
        return

    ps = getattr(proc, "patch_size", None)
    if ps is None and hasattr(proc, "image_processor"):
        ps = getattr(proc.image_processor, "patch_size", None)
    cfg = getattr(getattr(pipe, "model", None), "config", None)
    if ps is None and cfg is not None:
        vision_cfg = getattr(cfg, "vision_config", None)
        ps = getattr(vision_cfg, "patch_size", None) if vision_cfg else None
    if ps is None:
        ps = fallback

    proc.patch_size = ps
    if hasattr(proc, "image_processor"):
        try:
            proc.image_processor.patch_size = ps
        except Exception:
            pass


def gen_w_img(pipe, model_id: str, prompt: str, image) -> str:
    mid = model_id.lower()
    image = to_pil(image)

    if "llava" in mid:
        if "llava-med" in mid:
            _ensure_llava_patch_size(pipe, fallback=14)
            text = f"[INST] <image>\n{prompt} [/INST]"
        else:
            text = f"USER: <image>\n{prompt}\nASSISTANT:"

        out = pipe(
            images=image,
            text=text,
            max_new_tokens=MAX_NEW_TOKENS,
            return_full_text=False,
            generate_kwargs={"do_sample": False, "temperature": 0.0, "min_new_tokens": 32},
        )
        if isinstance(out, list) and out and isinstance(out[0], dict):
            return out[0].get("generated_text", str(out[0]))
        return str(out)

    # Gemma / MedGemma - uses the chat template
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt},
    ]}]
    out = pipe(
        text=messages,
        return_full_text=False,
        generate_kwargs={"max_new_tokens": MAX_NEW_TOKENS, "do_sample": DO_SAMPLE},
    )
    if isinstance(out, list) and out and isinstance(out[0], dict):
        gt = out[0].get("generated_text", None)
        if isinstance(gt, list) and gt and isinstance(gt[-1], dict) and "content" in gt[-1]:
            return str(gt[-1]["content"])
        if isinstance(gt, str):
            return gt
        return str(out[0])
    return str(out)
