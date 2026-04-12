# create_dataset.py - build the unified sampling manifest.
#
# Loads each HuggingFace dataset, samples K negatives and K positives,
# and writes the metadata (task, row_id, label, label_name) to a CSV.
# Images are not stored here - they are loaded on the fly by downstream
# scripts using the row_id as a positional index into the HF DataFrame.
#
# Usage:
#   python create_dataset.py [--k 50] [--tasks brain_tumor pneumonia ...]
#                            [--output medical_vqa_dataset.csv]
import argparse

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from config import ALL_DATASETS, DatasetConfig, K_DEFAULT


def load_hf_df(cfg: DatasetConfig) -> pd.DataFrame:
    kwargs = {"revision": cfg.hf_revision} if cfg.hf_revision else {}
    ds = load_dataset(cfg.hf_dataset, **kwargs)
    split = "train" if "train" in ds else list(ds.keys())[0]
    return pd.DataFrame(ds[split])


def build_sample(df: pd.DataFrame, cfg: DatasetConfig, k: int) -> pd.DataFrame:
    if cfg.label_col:
        neg = df[df[cfg.label_col] == cfg.neg_value].sample(k, random_state=42).copy()
        pos = df[df[cfg.label_col] == cfg.pos_value].sample(k, random_state=42).copy()
    else:
        # CheXpert-style: separate columns
        neg = df[df[cfg.neg_col] == cfg.present_value].sample(k, random_state=42).copy()
        pos = df[df[cfg.pos_col] == cfg.present_value].sample(k, random_state=42).copy()

    neg["label"] = 0
    pos["label"] = 1

    sample = (
        pd.concat([neg, pos])
        .reset_index(drop=False)
        .rename(columns={"index": "row_id"})
    )
    sample["label_name"] = sample["label"].map({0: cfg.neg_name, 1: cfg.pos_name})
    sample["task"] = cfg.name
    return sample[["task", "row_id", "label", "label_name"]]


def main():
    parser = argparse.ArgumentParser(description="Build medical_vqa_dataset.csv")
    parser.add_argument("--k",      type=int, default=K_DEFAULT,
                        help="Samples per class per task (default: %(default)s)")
    parser.add_argument("--tasks",  nargs="+", default=None,
                        help="Subset of tasks to include (default: all)")
    parser.add_argument("--output", default="medical_vqa_dataset.csv",
                        help="Output CSV path (default: %(default)s)")
    args = parser.parse_args()

    tasks = ALL_DATASETS
    if args.tasks:
        missing = set(args.tasks) - set(ALL_DATASETS)
        if missing:
            raise ValueError(f"Unknown task(s): {missing}. Available: {list(ALL_DATASETS)}")
        tasks = {k: v for k, v in ALL_DATASETS.items() if k in args.tasks}

    parts = []
    for name, cfg in tqdm(tasks.items(), desc="tasks"):
        print(f"\nLoading {cfg.hf_dataset} ...")
        df = load_hf_df(cfg)
        print(f"  {len(df)} rows, sampling {args.k} neg + {args.k} pos")
        sample = build_sample(df, cfg, args.k)
        parts.append(sample)
        print(f"  label distribution:\n{sample['label_name'].value_counts().to_string()}")

    out = pd.concat(parts, ignore_index=True)
    out.to_csv(args.output, index=False)
    print(f"\nSaved {len(out)} rows -> {args.output}")
    print(out.groupby("task")["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
