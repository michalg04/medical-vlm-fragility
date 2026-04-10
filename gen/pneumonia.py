import os
import re
import torch
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import pipeline

# ------------------------
# Config
# ------------------------
CLASS_NAME = "Pneumonia"
K = 50  # neg then pos => 2K images total

PROMPTS = [
   "Is this image abnormal?",
   "Is this image normal?",
   "Does this image contain evidence of pneumonia?",
   "Does this image contain pneumonia? Consider signs of consolidation, air bronchograms, and silhouettes.",
   """
   Which does this image show? Choose one answer.
   A) Pneumonia
   B) No Pneumonia
   """,
   """
   Which does this image show? Choose one answer.
   A) No Pneumonia
   B) Pneumonia
   """,
    "Describe this image in great detail and pay attention to any relevant pixels. Imagine a physician will use your description.",
    "Describe this image in great detail and pay attention to any relevant pixels. Imagine a physician will use your description to diagnose pneumonia.",
]

PROMPTS = PROMPTS + [x + "\nAnswer in one word." for x in PROMPTS[:-2]]

# Use HF-converted LLaVA checkpoints (these work with transformers pipeline)
MODELS = [
   "google/medgemma-4b-it",
   "google/gemma-3-4b-it",
   "llava-hf/llava-v1.6-mistral-7b-hf",
    "wnkh/llava-med-v1.5-mistral-7b-hf",
]

# 0: unlabeled, 1: uncertain, 2: absent, 3: present
PRESENT = 3

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

def build_sample(df: pd.DataFrame, class_name: str, k: int) -> pd.DataFrame:
    # Negatives: "No Finding" present => normal
    neg = df[df["No Finding"] == PRESENT].sample(k, random_state=42)

    # Positives: chosen class present
    pos = df[df[class_name] == PRESENT].sample(k, random_state=42)

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

    # Try to discover patch_size from multiple places
    ps = getattr(proc, "patch_size", None)
    if ps is None and hasattr(proc, "image_processor"):
        ps = getattr(proc.image_processor, "patch_size", None)

    cfg = getattr(getattr(pipe, "model", None), "config", None)
    if ps is None and cfg is not None:
        vision_cfg = getattr(cfg, "vision_config", None)
        ps = getattr(vision_cfg, "patch_size", None) if vision_cfg is not None else None

    if ps is None:
        ps = fallback

    # Set everywhere relevant
    proc.patch_size = ps
    if hasattr(proc, "image_processor"):
        try:
            proc.image_processor.patch_size = ps
        except Exception:
            pass

def gen_w_img(pipe, model_id: str, prompt: str, image):
    mid = (model_id or "").lower()

    # Some conversions ship with patch_size=None
    if "llava-med" in mid or "wnkh/llava-med" in mid:
        _ensure_llava_patch_size(pipe, fallback=14)

    # ----
    # Path A: models whose chat_template cannot handle list-of-blocks content
    # (common for some LLaVA-Med / older LLaVA conversions)
    # -> use classic "<image>\n{prompt}" + images=...
    # ----
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

    # ----
    # Path B: chat-template capable models (Gemma3/MedGemma, etc.)
    # ----
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
    ds = load_dataset("danjacobellis/chexpert")
    df = pd.DataFrame(ds["train"])

    # Quick checks
    assert "image" in df.columns, "Expected an 'image' column with images."
    assert "No Finding" in df.columns, "Expected CheXpert label column 'No Finding'."
    assert CLASS_NAME in df.columns, f"Expected label column '{CLASS_NAME}'."

    sample = build_sample(df, CLASS_NAME, K)
    print(f"Sampled {len(sample)} rows: {K} negatives + {K} positives.")
    print("Columns:", list(sample.columns))

    ids = sample["row_id"].tolist()

    for model_id in MODELS:
        print("\n" + "=" * 90)
        print(f"Running model: {model_id}")
        print("=" * 90)

        pipe = make_pipe(model_id)

        # answers[prompt_idx][img_idx] = generated_text
        answers = []

        # Optional: store labels too for later analysis
        labels = sample[CLASS_NAME].tolist()
        no_finding = sample["No Finding"].tolist()

        for prompt in tqdm(PROMPTS, desc=f"prompts ({model_id})"):
            prompt_answers = []
            # torch.inference_mode helps a bit even though pipeline wraps it internally
            with torch.inference_mode():
                for row in tqdm(sample.itertuples(index=False), total=len(sample)):
                    image = row.image
                    text = gen_w_img(pipe, model_id, prompt, image)
                    tqdm.write(text)
                    prompt_answers.append(text)
            answers.append(prompt_answers)

        # Build a DataFrame: rows = prompts, cols = row_ids
        answers_df = pd.DataFrame(answers, columns=ids)
        answers_df.insert(0, "prompt", PROMPTS)

        # Add metadata table too (handy)
        # label=1 for pneumonia positives, label=0 for no-finding negatives
        binary_labels = [1 if lbl == PRESENT else 0 for lbl in labels]
        meta_df = pd.DataFrame({
            "row_id": ids,
            "label": binary_labels,
            "No Finding": no_finding,
            CLASS_NAME: labels,
        })

        out_obj = {
            "model_id": model_id,
            "class_name": CLASS_NAME,
            "k": K,
            "prompts": PROMPTS,
            "answers_df": answers_df,
            "meta_df": meta_df,
        }

        out_path = f"{safe_name(model_id)}__{safe_name(CLASS_NAME)}__k{K}__descriptions.pth"
        torch.save(out_obj, out_path)
        print(f"Saved: {out_path}")
        
        del pipe
        import gc; gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()

