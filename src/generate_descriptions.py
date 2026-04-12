# generate_descriptions.py - generate free-text image descriptions from a VLM.
#
# For each image in the dataset manifest, runs the description prompts and
# saves results to {model}_{task}_descriptions.csv.
# Step 1 of the description pipeline: image -> VLM description -> GPT diagnosis.
#
# Usage:
#   python generate_descriptions.py --model gemma --task pneumonia
#   python generate_descriptions.py --model medgemma --task brain_tumor
import argparse
import os

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

from config import ALL_DATASETS, MODEL_ALIASES, DatasetConfig
from vlm_utils import free_pipe, gen_w_img, make_pipe


def resolve_model(model_arg: str) -> str:
    return MODEL_ALIASES.get(model_arg, model_arg)


def load_hf_df(cfg: DatasetConfig) -> pd.DataFrame:
    kwargs = {"revision": cfg.hf_revision} if cfg.hf_revision else {}
    ds = load_dataset(cfg.hf_dataset, **kwargs)
    split = "train" if "train" in ds else list(ds.keys())[0]
    return pd.DataFrame(ds[split])


def main():
    parser = argparse.ArgumentParser(description="Generate VLM image descriptions")
    parser.add_argument("--model",       required=True,
                        help=f"Model alias or HF ID. Aliases: {list(MODEL_ALIASES)}")
    parser.add_argument("--task",        required=True,
                        help=f"Task name. Available: {list(ALL_DATASETS)}")
    parser.add_argument("--dataset_csv", default="medical_vqa_dataset.csv",
                        help="Manifest CSV from create_dataset.py")
    parser.add_argument("--out_dir",     default="results",
                        help="Output directory (default: results/)")
    args = parser.parse_args()

    if args.task not in ALL_DATASETS:
        raise ValueError(f"Unknown task '{args.task}'. Available: {list(ALL_DATASETS)}")

    cfg      = ALL_DATASETS[args.task]
    model_id = resolve_model(args.model)
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = pd.read_csv(args.dataset_csv)
    task_df  = manifest[manifest["task"] == args.task].reset_index(drop=True)
    if task_df.empty:
        raise RuntimeError(f"No rows for task '{args.task}' in {args.dataset_csv}. "
                           "Run create_dataset.py first.")

    print(f"Task:   {args.task}  ({len(task_df)} images)")
    print(f"Model:  {model_id}")
    print(f"Prompts ({len(cfg.describe_prompts)}): {cfg.describe_prompts}")

    print("Loading HuggingFace dataset for images...")
    hf_df = load_hf_df(cfg)

    pipe    = make_pipe(model_id)
    records = []

    for prompt_idx, prompt_text in enumerate(cfg.describe_prompts):
        print(f"\nPrompt {prompt_idx}: {prompt_text[:80]}...")
        with torch.inference_mode():
            for _, row in tqdm(task_df.iterrows(), total=len(task_df)):
                image       = hf_df.iloc[int(row["row_id"])][cfg.image_col]
                description = gen_w_img(pipe, model_id, prompt_text, image)
                records.append({
                    "task":        args.task,
                    "model":       model_id,
                    "row_id":      row["row_id"],
                    "label":       row["label"],
                    "label_name":  row["label_name"],
                    "prompt_idx":  prompt_idx,
                    "prompt":      prompt_text,
                    "description": description,
                })

    free_pipe(pipe)

    model_slug = args.model.replace("/", "_")
    out_path   = os.path.join(args.out_dir, f"{model_slug}_{args.task}_descriptions.csv")
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\nSaved {len(records)} descriptions -> {out_path}")


if __name__ == "__main__":
    main()
