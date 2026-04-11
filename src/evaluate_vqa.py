# evaluate_vqa.py - run closed-form VQA and judge responses with GPT.
#
# For each (model, task) pair:
#   1. loads the dataset manifest from medical_vqa_dataset.csv
#   2. fetches images on the fly from HuggingFace using row_ids
#   3. runs the VLM with the selected prompts
#   4. uses GPT-5 as a judge to classify each response as positive/negative
#   5. saves per-row results to {model}_{task}_vqa_results.csv
#
# Usage:
#   python evaluate_vqa.py --model gemma --task pneumonia
#   python evaluate_vqa.py --model medgemma --task brain_tumor --all_prompts True
#   python evaluate_vqa.py --model llava --task skin_cancer --prompt_type ab --one_word True
#   python evaluate_vqa.py --model medgemma --task brain_tumor --no_judge
import argparse
import json
import os

import pandas as pd
import torch
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from config import (ALL_DATASETS, GPT_JUDGE_MODEL_DEFAULT, MODEL_ALIASES,
                    DatasetConfig)
from vlm_utils import free_pipe, gen_w_img, make_pipe

JUDGE_USER_TEMPLATE = """\
Original prompt given to the model:
{prompt}

Model's response:
{response}

Classification (one word):"""


def resolve_model(model_arg: str) -> str:
    return MODEL_ALIASES.get(model_arg, model_arg)


def select_prompts(cfg: DatasetConfig, prompt_type: str, one_word: bool) -> list[tuple[str, str]]:
    buckets: dict[str, list[str]] = {
        "yesno": cfg.yesno_prompts,
        "ab":    cfg.ab_prompts,
    }
    if prompt_type == "all":
        selected = [(ptype, p) for ptype, ps in buckets.items() for p in ps]
    elif prompt_type in buckets:
        selected = [(prompt_type, p) for p in buckets[prompt_type]]
    else:
        raise ValueError(f"Unknown prompt_type '{prompt_type}'. Choose: yesno, ab, all")

    if one_word:
        selected = [(pt, p + "\nAnswer in one word.") for pt, p in selected]

    return selected


def judge_response(client: OpenAI, cfg: DatasetConfig, prompt: str, response: str, judge_model: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=judge_model,
            temperature=0,
            messages=[
                {"role": "system", "content": cfg.judge_system_prompt},
                {"role": "user",   "content": JUDGE_USER_TEMPLATE.format(
                    prompt=prompt, response=response)},
            ],
        )
        raw = completion.choices[0].message.content.strip().upper()
        # Check neg_token first - it may contain pos_token as a substring (e.g. NO_TUMOR / TUMOR)
        if cfg.judge_neg_token in raw:
            return cfg.judge_neg_token
        if cfg.judge_pos_token in raw:
            return cfg.judge_pos_token
        return "AMBIGUOUS"
    except Exception as e:
        print(f"  [judge error] {e}")
        return "ERROR"


def load_images_for_task(cfg: DatasetConfig) -> pd.DataFrame:
    # Returns the full HF DataFrame so we can look up images by row_id (positional index).
    kwargs = {"revision": cfg.hf_revision} if cfg.hf_revision else {}
    ds = load_dataset(cfg.hf_dataset, **kwargs)
    split = "train" if "train" in ds else list(ds.keys())[0]
    return pd.DataFrame(ds[split])


def main():
    parser = argparse.ArgumentParser(description="Run VQA evaluation for one model/task pair")
    parser.add_argument("--model",       required=True,
                        help=f"Model alias or full HF ID. Aliases: {list(MODEL_ALIASES)}")
    parser.add_argument("--task",        required=True,
                        help=f"Task name. Available: {list(ALL_DATASETS)}")
    parser.add_argument("--prompt_type", default="all",
                        choices=["yesno", "ab", "all"],
                        help="Which prompt type(s) to run (default: all)")
    parser.add_argument("--one_word",    default=False, type=lambda x: x.lower() == "true",
                        help="Append 'Answer in one word.' to prompts (default: False)")
    parser.add_argument("--all_prompts", default=False, type=lambda x: x.lower() == "true",
                        help="Shorthand for --prompt_type all (overrides --prompt_type)")
    parser.add_argument("--dataset_csv", default="medical_vqa_dataset.csv",
                        help="Path to manifest CSV from create_dataset.py")
    parser.add_argument("--out_dir",     default="results",
                        help="Output directory (default: results/)")
    parser.add_argument("--judge_model", default=GPT_JUDGE_MODEL_DEFAULT,
                        help=f"GPT model for judging (default: {GPT_JUDGE_MODEL_DEFAULT})")
    parser.add_argument("--no_judge",    action="store_true",
                        help="Skip GPT judging; save raw responses only")
    parser.add_argument("--cache",       default="vqa_judge_cache.json",
                        help="JSON cache for GPT judge calls")
    args = parser.parse_args()

    if args.all_prompts:
        args.prompt_type = "all"

    if args.task not in ALL_DATASETS:
        raise ValueError(f"Unknown task '{args.task}'. Available: {list(ALL_DATASETS)}")

    cfg      = ALL_DATASETS[args.task]
    model_id = resolve_model(args.model)
    prompts  = select_prompts(cfg, args.prompt_type, args.one_word)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load sampling manifest
    manifest = pd.read_csv(args.dataset_csv)
    task_df  = manifest[manifest["task"] == args.task].reset_index(drop=True)
    if task_df.empty:
        raise RuntimeError(f"No rows for task '{args.task}' in {args.dataset_csv}. "
                           "Run create_dataset.py first.")

    print(f"Task:   {args.task}  ({len(task_df)} images)")
    print(f"Model:  {model_id}")
    print(f"Prompts ({len(prompts)}): {args.prompt_type}, one_word={args.one_word}")

    # Load images
    print("Loading HuggingFace dataset for images...")
    hf_df = load_images_for_task(cfg)

    # Load GPT cache
    cache: dict = {}
    if not args.no_judge and os.path.exists(args.cache):
        with open(args.cache) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached judgments.")

    client = OpenAI() if not args.no_judge else None

    pipe = make_pipe(model_id)

    records = []
    for prompt_type_label, prompt_text in tqdm(prompts, desc="prompts"):
        with torch.inference_mode():
            for _, row in tqdm(task_df.iterrows(), total=len(task_df), leave=False):
                image     = hf_df.iloc[int(row["row_id"])][cfg.image_col]
                response  = gen_w_img(pipe, model_id, prompt_text, image)

                judgment, pred, correct = None, None, None
                if not args.no_judge:
                    cache_key = f"{model_id}||{args.task}||{prompt_text}||{row['row_id']}"
                    if cache_key in cache:
                        judgment = cache[cache_key]
                    else:
                        judgment = judge_response(client, cfg, prompt_text, response, args.judge_model)
                        cache[cache_key] = judgment
                        with open(args.cache, "w") as f:
                            json.dump(cache, f)

                    pred    = 1 if judgment == cfg.judge_pos_token else (0 if judgment == cfg.judge_neg_token else -1)
                    correct = int(pred == row["label"]) if pred in (0, 1) else None

                records.append({
                    "task":        args.task,
                    "model":       model_id,
                    "row_id":      row["row_id"],
                    "label":       row["label"],
                    "label_name":  row["label_name"],
                    "prompt_type": prompt_type_label,
                    "one_word":    args.one_word,
                    "prompt":      prompt_text,
                    "response":    response,
                    "judgment":    judgment,
                    "pred":        pred,
                    "correct":     correct,
                })

    free_pipe(pipe)

    model_slug = args.model.replace("/", "_")
    out_path   = os.path.join(args.out_dir, f"{model_slug}_{args.task}_vqa_results.csv")
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\nSaved {len(records)} rows -> {out_path}")

    if not args.no_judge:
        df = pd.DataFrame(records)
        scored = df[df["pred"].isin([0, 1])]
        if len(scored):
            acc = (scored["pred"] == scored["label"]).mean()
            print(f"Overall accuracy (scored rows): {acc:.4f}  "
                  f"({len(scored)}/{len(df)} scored, {len(df)-len(scored)} ambiguous/error)")


if __name__ == "__main__":
    main()
