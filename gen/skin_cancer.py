import io
import re
import torch
import pandas as pd
from datasets import load_dataset
from PIL import Image as PILImage
from tqdm import tqdm
import torch
from transformers import pipeline

# ------------------------
# Config
# ------------------------
CLASS_NAME = "SkinCancer"  # Skin cancer detection
K = 50  # neg then pos => 2K images total

PROMPTS = [
   "Is this skin lesion abnormal?",
   "Is this skin lesion normal?",
   "Does this image contain evidence of malignant skin cancer?",
   "Does this image contain evidence of malignant skin cancer? Consider signs of irregular borders, asymmetry, uneven pigmentation, and other atypical morphology.",
   """
   Which does this image show? Choose one answer.
   A) Malignant
   B) Benign
   """,
   """
   Which does this image show? Choose one answer.
   A) Benign
   B) Malignant
   """,
    "Describe this image in great detail and pay attention to any relevant pixels. Imagine a physician will use your description.",
    "Describe this image in great detail and pay attention to any relevant pixels. Imagine a physician will use your description to diagnose skin cancer.",
]

PROMPTS = PROMPTS + [x + "\nAnswer in one word." for x in PROMPTS[:-2]]

print(PROMPTS)

# Use HF-converted LLaVA checkpoints (these work with transformers pipeline)
MODELS = [
    "google/medgemma-4b-it",
    "google/gemma-3-4b-it",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "wnkh/llava-med-v1.5-mistral-7b-hf",
]

# Label convention for Falah/skin-cancer:
# 0 = benign  (not malignant)
# 1 = malignant
NOT_PRESENT = 0
PRESENT = 1

MAX_NEW_TOKENS = 2048  # descriptions can be long; keep sane to avoid OOM/timeouts
DO_SAMPLE = False     # deterministic is better for eval repeatability

# If you want to avoid cached downloads in home, you can set these externally:
# export HF_HOME=/path/with/space/huggingface
# export TRANSFORMERS_CACHE=$HF_HOME/transformers

# ------------------------
# Helpers
# ------------------------
def safe_name(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    return s[:120]

def build_sample(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Sample K benign and K malignant images.

    Args:
        df: DataFrame with 'label' column (0=benign, 1=malignant)
        k: number of samples per class

    Returns:
        DataFrame with 2K rows (K benign + K malignant)
    """
    # Negatives: benign (label == 0)
    neg = df[df["label"] == NOT_PRESENT].sample(k, random_state=42)

    # Positives: malignant (label == 1)
    pos = df[df["label"] == PRESENT].sample(k, random_state=42)

    sample = pd.concat([neg, pos], axis=0)
    # Keep stable ordering: neg then pos
    sample = sample.reset_index(drop=False).rename(columns={"index": "row_id"})
    return sample

def make_pipe(model_id: str):
    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "image-text-to-text",
        model=model_id,
        device=device,
        dtype=dtype,
    )


def _ensure_llava_patch_size(pipe, fallback=14):
    """
    Some LLaVA processors ship with patch_size=None (esp. certain conversions).
    Fix it once by pulling from any available place, otherwise fallback to 14.
    """
    proc = getattr(pipe, "processor", None)
    if proc is None:
        return

    ps = getattr(proc, "patch_size", None)
    if ps is None and hasattr(proc, "image_processor"):
        ps = getattr(proc.image_processor, "patch_size", None)

    cfg = getattr(getattr(pipe, "model", None), "config", None)
    if ps is None and cfg is not None:
        vision_cfg = getattr(cfg, "vision_config", None)
        ps = getattr(vision_cfg, "patch_size", None) if vision_cfg is not None else None

    if ps is None:
        ps = fallback

    proc.patch_size = ps
    if hasattr(proc, "image_processor"):
        try:
            proc.image_processor.patch_size = ps
        except Exception:
            pass

def to_pil(image) -> PILImage.Image:
    """
    Normalise whatever the dataset gives us into a PIL Image.
    HuggingFace datasets decoded via pd.DataFrame() lose the Image feature
    and return dicts like {'bytes': b'...', 'path': '...'} instead of PIL Images.
    """
    if isinstance(image, PILImage.Image):
        return image
    if isinstance(image, dict):
        if image.get("bytes"):
            return PILImage.open(io.BytesIO(image["bytes"])).convert("RGB")
        if image.get("path"):
            return PILImage.open(image["path"]).convert("RGB")
    raise TypeError(f"Cannot convert image of type {type(image)} to PIL Image")


def gen_w_img(pipe, model_id: str, prompt: str, image):
    mid = (model_id or "").lower()
    image = to_pil(image)

    if "llava-med" in mid or "wnkh/llava-med" in mid:
        _ensure_llava_patch_size(pipe, fallback=14)

    if "llava" in mid:
        if "llava-med" in mid:
            text = f"[INST] <image>\n{prompt} [/INST]"
        else:
            text = f"USER: <image>\n{prompt}\nASSISTANT:"

        out = pipe(
            images=image,
            text=text,
            max_new_tokens=MAX_NEW_TOKENS,
            return_full_text=False,
            generate_kwargs={
                "do_sample": False,
                "temperature": 0.0,
                "min_new_tokens": 32,
            },
        )

        if isinstance(out, list) and out and isinstance(out[0], dict):
            return out[0].get("generated_text", str(out[0]))
        return str(out)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    out = pipe(
        text=messages,
        max_new_tokens=MAX_NEW_TOKENS,
        return_full_text=False,
        do_sample=DO_SAMPLE,
    )

    if isinstance(out, list) and out and isinstance(out[0], dict):
        gt = out[0].get("generated_text", None)
        if isinstance(gt, list) and gt and isinstance(gt[-1], dict) and "content" in gt[-1]:
            return str(gt[-1]["content"])
        if isinstance(gt, str):
            return gt
        return str(out[0])
    return str(out)


# ------------------------
# Main
# ------------------------
def main():
    print("Loading dataset...")
    ds = load_dataset("Falah/skin-cancer", revision="refs/convert/parquet")

    split_name = "train" if "train" in ds else list(ds.keys())[0]
    df = pd.DataFrame(ds[split_name])

    # Quick checks
    assert "image" in df.columns, "Expected an 'image' column with images."
    assert "label" in df.columns, "Expected a 'label' column (0=benign, 1=malignant)."

    # Print label distribution
    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())
    print(f"  0 (benign):    {(df['label'] == NOT_PRESENT).sum()}")
    print(f"  1 (malignant): {(df['label'] == PRESENT).sum()}")

    sample = build_sample(df, K)
    print(f"\nSampled {len(sample)} rows: {K} benign + {K} malignant.")
    print("Columns:", list(sample.columns))

    ids = sample["row_id"].tolist()

    for model_id in MODELS:
        print("\n" + "=" * 90)
        print(f"Running model: {model_id}")
        print("=" * 90)

        pipe = make_pipe(model_id)

        answers = []
        labels = sample["label"].tolist()

        for prompt in tqdm(PROMPTS, desc=f"prompts ({model_id})"):
            prompt_answers = []
            with torch.inference_mode():
                for row in tqdm(sample.itertuples(index=False), total=len(sample)):
                    image = row.image
                    text = gen_w_img(pipe, model_id, prompt, image)
                    prompt_answers.append(text)
            answers.append(prompt_answers)

        answers_df = pd.DataFrame(answers, columns=ids)
        answers_df.insert(0, "prompt", PROMPTS)

        meta_df = pd.DataFrame({
            "row_id": ids,
            "label": labels,
            "class_name": ["benign" if lbl == NOT_PRESENT else "malignant" for lbl in labels],
        })

        out_obj = {
            "model_id": model_id,
            "class_name": CLASS_NAME,
            "k": K,
            "prompts": PROMPTS,
            "answers_df": answers_df,
            "meta_df": meta_df,
        }

        out_path = f"{safe_name(model_id)}__{safe_name(CLASS_NAME)}__k{K}_skin_vqa_one_word.pth"
        torch.save(out_obj, out_path)
        print(f"Saved: {out_path}")

        del pipe
        import gc; gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
