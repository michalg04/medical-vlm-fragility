# evaluate_descriptions.py - GPT-based diagnosis from VLM descriptions.
#
# Reads the descriptions CSV from generate_descriptions.py, then for each
# description asks GPT to answer Yes/No using structured JSON output.
# Pipeline: image -> VLM description -> GPT Yes/No -> accuracy metrics
#
# Usage:
#   python evaluate_descriptions.py --model gemma --task pneumonia
#   python evaluate_descriptions.py --descriptions_file results/gemma_pneumonia_descriptions.csv
import argparse
import json
import os

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from config import (ALL_DATASETS, GPT_DESC_MODEL_DEFAULT, MODEL_ALIASES,
                    DatasetConfig)

YES_NO_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["Yes", "No"]}},
    "required": ["answer"],
    "additionalProperties": False,
}


def gpt_yesno(client: OpenAI, prompt: str, model_id: str, max_tokens: int = 64) -> tuple[str, str | None]:
    resp = client.responses.create(
        model=model_id,
        input=prompt,
        reasoning={"effort": "none"},
        temperature=0,
        max_output_tokens=max_tokens,
        text={"format": {
            "type": "json_schema",
            "name": "yes_no_answer",
            "strict": True,
            "schema": YES_NO_SCHEMA,
        }},
    )
    raw = resp.output_text
    try:
        yn = json.loads(raw).get("answer")
        return raw, yn if yn in ("Yes", "No") else None
    except Exception:
        return raw, None


def resolve_model_slug(model_arg: str) -> str:
    full_id = MODEL_ALIASES.get(model_arg, model_arg)
    return full_id.replace("/", "_")


def run_eval(desc_df: pd.DataFrame, cfg: DatasetConfig, client: OpenAI,
             gpt_model: str, cache: dict, cache_path: str) -> pd.DataFrame:
    records = []
    total   = len(desc_df) * len(cfg.desc_prompt_variants)

    with tqdm(total=total, desc=f"desc_eval [{cfg.name}]") as pbar:
        for _, row in desc_df.iterrows():
            desc   = row["description"] if pd.notna(row.get("description")) else ""
            y_true = int(row["label"])

            for variant_name, tmpl in cfg.desc_prompt_variants.items():
                prompt    = tmpl.format(desc=desc)
                cache_key = f"desc||{row['model']}||{cfg.name}||{row['prompt_idx']}||{row['row_id']}||{variant_name}"

                if cache_key in cache:
                    raw, yn = cache[cache_key]
                else:
                    raw, yn = gpt_yesno(client, prompt, gpt_model)
                    cache[cache_key] = [raw, yn]
                    with open(cache_path, "w") as f:
                        json.dump(cache, f)

                y_pred  = 1 if yn == "Yes" else (0 if yn == "No" else None)
                correct = int(y_pred == y_true) if y_pred is not None else None

                records.append({
                    "task":          cfg.name,
                    "vlm_model":     row["model"],
                    "row_id":        row["row_id"],
                    "label":         y_true,
                    "label_name":    row["label_name"],
                    "desc_prompt_idx": row["prompt_idx"],
                    "desc_prompt":   row["prompt"],
                    "description":   desc,
                    "variant":       variant_name,
                    "gpt_prompt":    prompt,
                    "raw_response":  raw,
                    "yn":            yn,
                    "y_pred":        y_pred,
                    "correct":       correct,
                })
                pbar.update(1)

    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame) -> None:
    usable = df.dropna(subset=["y_pred"])
    if usable.empty:
        print("No scoreable predictions.")
        return

    summary = (
        usable.groupby(["vlm_model", "task", "variant", "desc_prompt_idx"])
        .agg(
            n       = ("y_pred", "size"),
            acc     = ("correct", "mean"),
            yes_rate= ("y_pred", "mean"),
        )
        .reset_index()
        .sort_values("acc", ascending=False)
    )
    print("\n" + "="*80)
    print("Description Eval Summary")
    print("="*80)
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Evaluate descriptions with GPT Yes/No")
    parser.add_argument("--model",             default=None,
                        help="VLM model alias (e.g. gemma). Used to infer filename if --descriptions_file not set.")
    parser.add_argument("--task",              default=None,
                        help="Task name. Used to infer filename if --descriptions_file not set.")
    parser.add_argument("--descriptions_file", default=None,
                        help="Path to descriptions CSV from generate_descriptions.py.")
    parser.add_argument("--results_dir",       default="results")
    parser.add_argument("--out_dir",           default="results")
    parser.add_argument("--gpt_model",         default=GPT_DESC_MODEL_DEFAULT,
                        help=f"GPT model ID (default: {GPT_DESC_MODEL_DEFAULT})")
    parser.add_argument("--cache",             default="desc_eval_cache.json")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    if args.descriptions_file:
        csv_paths = [args.descriptions_file]
    elif args.model and args.task:
        slug = resolve_model_slug(args.model)
        csv_paths = [os.path.join(args.results_dir, f"{slug}_{args.task}_descriptions.csv")]
    else:
        import glob
        csv_paths = sorted(glob.glob(os.path.join(args.results_dir, "*_descriptions.csv")))
        if not csv_paths:
            raise FileNotFoundError(f"No *_descriptions.csv files in {args.results_dir}. "
                                    "Run generate_descriptions.py first.")

    cache: dict = {}
    if os.path.exists(args.cache):
        with open(args.cache) as f:
            raw_cache = json.load(f)
        # Values are [raw, yn] lists from JSON (tuples serialise as lists)
        cache = {k: v for k, v in raw_cache.items()}
        print(f"Loaded {len(cache)} cached GPT calls.")

    client     = OpenAI()
    os.makedirs(args.out_dir, exist_ok=True)

    all_results = []
    for csv_path in csv_paths:
        print(f"\nLoading {csv_path}")
        desc_df = pd.read_csv(csv_path)

        task = desc_df["task"].iloc[0] if "task" in desc_df.columns else args.task
        if task not in ALL_DATASETS:
            print(f"  [warn] unknown task '{task}', skipping.")
            continue

        cfg     = ALL_DATASETS[task]
        results = run_eval(desc_df, cfg, client, args.gpt_model, cache, args.cache)
        all_results.append(results)

        model_slug = desc_df["model"].iloc[0].replace("/", "_")
        out_path   = os.path.join(args.out_dir, f"{model_slug}_{task}_desc_eval.csv")
        results.to_csv(out_path, index=False)
        print(f"Saved -> {out_path}")

    if all_results:
        print_summary(pd.concat(all_results, ignore_index=True))


if __name__ == "__main__":
    main()
