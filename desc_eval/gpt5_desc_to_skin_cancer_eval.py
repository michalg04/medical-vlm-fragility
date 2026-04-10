#!/usr/bin/env python3
import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import torch
import pandas as pd
from tqdm import tqdm

from openai import OpenAI

# Adjust to match your labeling constants
# 0: benign, 1: malignant
PRESENT = 1

PROMPT_VARIANTS: Dict[str, str] = {
    "one_word": (
        "You are a careful clinician.\n"
        "Given the following skin area description, answer ONLY whether skin cancer is present.\n"
        "Return a single word: Yes or No.\n\n"
        "Description:\n{desc}\n"
    ),
    "tight": (
        "Task: Decide if skin cancer is present.\n"
        "Output must be Yes or No.\n\n"
        "Description:\n{desc}\n"
    ),
    "disease": (
        "Task: Decide if disease is present.\n"
        "Output must be Yes or No.\n\n"
        "Description:\n{desc}\n"
    ),
}


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def extract_descriptions(obj: dict) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Returns:
      answers_only: rows = desc_prompt_idx, cols = row_id
      meta_df: has row_id and labels
      desc_prompts: list[str], len = n_desc_prompts
    """
    answers_df = obj["answers_df"]
    meta_df = obj["meta_df"]

    if "prompt" in answers_df.columns:
        desc_prompts = answers_df["prompt"].tolist()
        answers_only = answers_df.drop(columns=["prompt"])
    else:
        desc_prompts = obj.get("prompts", [f"prompt_{i}" for i in range(len(answers_df))])
        answers_only = answers_df

    if "row_id" not in meta_df.columns:
        raise ValueError("meta_df missing 'row_id' column.")
    return answers_only, meta_df, desc_prompts


@dataclass
class GPTTextModel:
    name: str
    model_id: str


def gpt_yesno(
    client: OpenAI,
    prompt: str,
    model_id: str = "gpt-4o",
    max_output_tokens: int = 64,
) -> tuple[str, Optional[str]]:
    """
    Uses structured output (json_schema) to force Yes/No response.
    Returns: (raw_json_text, "Yes"/"No"/None)
    """
    schema_name = "yes_no_answer"
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": ["Yes", "No"]},
        },
        "required": ["answer"],
        "additionalProperties": False,
    }

    resp = client.responses.create(
        model=model_id,
        input=prompt,
        reasoning={"effort": "none"},
        temperature=0,
        max_output_tokens=max_output_tokens,
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    )

    raw = resp.output_text  # JSON string like {"answer":"Yes"}
    yn = None
    obj = json.loads(raw)
    if obj.get("answer") in ("Yes", "No"):
        yn = obj["answer"]
    return raw, yn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input .pth from skin_cancer_describe_eval.py")
    ap.add_argument("--out_dir", default="skin_cancer_eval_results", help="Output directory")
    ap.add_argument("--model", default="gpt-5.1", help="GPT model id")
    ap.add_argument("--max_output_tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="If >0, only evaluate first N images (debug)")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    safe_mkdir(args.out_dir)

    print(f"Loading descriptions: {args.in_path}")
    obj = torch.load(args.in_path, map_location="cpu", weights_only=False)

    answers_only, meta_df, desc_prompts = extract_descriptions(obj)

    # label: 0 = benign, 1 = malignant
    if "label" not in meta_df.columns:
        raise ValueError("meta_df missing 'label' column.")
    meta_df["y_true"] = (meta_df["label"] == PRESENT).astype(int)

    # Map row_id -> label, robust to dtype mismatch
    y_map: Dict[Any, int] = {}
    for rid, y in zip(meta_df["row_id"].tolist(), meta_df["y_true"].tolist()):
        y_map[rid] = int(y)
        y_map[str(rid)] = int(y)

    row_ids = list(answers_only.columns)
    if args.limit and args.limit > 0:
        row_ids = row_ids[: args.limit]
        answers_only = answers_only[row_ids]

    client = OpenAI()
    tm = GPTTextModel(name=args.model, model_id=args.model)

    records = []
    total = len(desc_prompts) * len(row_ids) * len(PROMPT_VARIANTS)
    pbar = tqdm(total=total, desc="scoring")

    for desc_prompt_idx, desc_prompt_text in enumerate(desc_prompts):
        desc_row = answers_only.iloc[desc_prompt_idx]

        for row_id in row_ids:
            desc = desc_row[row_id]
            y_true = y_map.get(row_id, y_map.get(str(row_id), None))

            if desc is None or (isinstance(desc, float) and pd.isna(desc)):
                desc = ""

            for variant_name, tmpl in PROMPT_VARIANTS.items():
                prompt = tmpl.format(desc=desc)

                raw, yn = gpt_yesno(
                    client=client,
                    prompt=prompt,
                    model_id=args.model,
                    max_output_tokens=args.max_output_tokens,
                )

                y_pred = 1 if yn == "Yes" else 0 if yn == "No" else None

                records.append(
                    {
                        "desc_source_model": obj.get("model_id", "unknown_vlm"),
                        "desc_prompt_idx": desc_prompt_idx,
                        "desc_prompt": desc_prompt_text,
                        "row_id": row_id,
                        "y_true": y_true,
                        "text_model": tm.name,
                        "text_model_id": tm.model_id,
                        "prompt_variant": variant_name,
                        "raw_generation": raw,
                        "parsed_yesno": yn,
                        "y_pred": y_pred,
                    }
                )
                pbar.update(1)

    pbar.close()

    df = pd.DataFrame(records)
    out_long = os.path.join(args.out_dir, "skin_cancer_long.csv")
    df.to_csv(out_long, index=False)
    print(f"Saved long results: {out_long}")

    usable = df.dropna(subset=["y_true", "y_pred"])
    if len(usable) == 0:
        print("No usable predictions to score — check API outputs.")
        return

    summary = (
        usable.groupby(["text_model", "prompt_variant", "desc_prompt_idx"], dropna=False)
        .agg(
            n=("y_pred", "size"),
            acc=("y_pred", lambda s: (s.values == usable.loc[s.index, "y_true"].values).mean()),
            yes_rate=("y_pred", "mean"),
        )
        .reset_index()
    )
    out_sum = os.path.join(args.out_dir, "skin_cancer_summary.csv")
    summary.to_csv(out_sum, index=False)
    print(f"Saved summary: {out_sum}")

    topline = (
        usable.groupby(["text_model", "prompt_variant"])
        .agg(
            n=("y_pred", "size"),
            acc=("y_pred", lambda s: (s.values == usable.loc[s.index, "y_true"].values).mean()),
        )
        .reset_index()
        .sort_values(["acc", "n"], ascending=[False, False])
    )
    print("\nTopline accuracy:")
    print(topline.to_string(index=False))


if __name__ == "__main__":
    main()
