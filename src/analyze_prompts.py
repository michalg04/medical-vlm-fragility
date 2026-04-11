# analyze_prompts.py - prompt sensitivity analysis.
#
# Reads the VQA results CSV from evaluate_vqa.py and computes per-prompt
# accuracy, sensitivity, specificity, F1, refusal rates, and variance.
#
# Usage:
#   python analyze_prompts.py --model medgemma --task pneumonia
#   python analyze_prompts.py --results_file results/medgemma_pneumonia_vqa_results.csv
import argparse
import os

import pandas as pd

from config import ALL_DATASETS, MODEL_ALIASES


def resolve_model_slug(model_arg: str) -> str:
    full_id = MODEL_ALIASES.get(model_arg, model_arg)
    return full_id.replace("/", "_")


def compute_metrics(group: pd.DataFrame) -> dict:
    total     = len(group)
    scored    = group[group["pred"].isin([0, 1])]
    ambiguous = total - len(scored)

    if scored.empty:
        return {"accuracy": None, "sensitivity": None, "specificity": None,
                "precision": None, "f1": None,
                "n_total": total, "n_scored": 0, "n_ambiguous": ambiguous,
                "tp": 0, "tn": 0, "fp": 0, "fn": 0}

    tp = int(((scored["label"] == 1) & (scored["pred"] == 1)).sum())
    tn = int(((scored["label"] == 0) & (scored["pred"] == 0)).sum())
    fp = int(((scored["label"] == 0) & (scored["pred"] == 1)).sum())
    fn = int(((scored["label"] == 1) & (scored["pred"] == 0)).sum())
    n  = len(scored)

    acc  = (tp + tn) / n
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1   = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0

    return {
        "accuracy": round(acc, 4), "sensitivity": round(sens, 4),
        "specificity": round(spec, 4), "precision": round(prec, 4), "f1": round(f1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_total": total, "n_scored": len(scored), "n_ambiguous": ambiguous,
    }


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, task, prompt_type, one_word, prompt), group in df.groupby(
        ["model", "task", "prompt_type", "one_word", "prompt"], sort=False
    ):
        m = compute_metrics(group)
        m.update({
            "model": model, "task": task,
            "prompt_type": prompt_type, "one_word": one_word,
            "prompt": prompt,
        })
        rows.append(m)
    return pd.DataFrame(rows)


def print_summary(metrics_df: pd.DataFrame) -> None:
    col_order = ["model", "task", "prompt_type", "one_word", "prompt",
                 "accuracy", "sensitivity", "specificity", "f1",
                 "n_scored", "n_ambiguous", "n_total"]
    display = metrics_df[[c for c in col_order if c in metrics_df.columns]]

    for (model, task), group in display.groupby(["model", "task"]):
        print(f"\n{'='*80}")
        print(f"Model: {model}   Task: {task}")
        print(f"{'='*80}")
        print(group.drop(columns=["model", "task"]).to_string(index=False))

        accs = group["accuracy"].dropna()
        if len(accs) > 1:
            print(f"\n  Accuracy across prompts:  mean={accs.mean():.4f}  "
                  f"std={accs.std():.4f}  range=[{accs.min():.4f}, {accs.max():.4f}]")
        ambig_rate = group["n_ambiguous"].sum() / group["n_total"].sum()
        print(f"  Refusal/ambiguous rate:   {ambig_rate:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Prompt sensitivity analysis")
    parser.add_argument("--model",        default=None,
                        help="Model alias or slug (e.g. medgemma). Ignored if --results_file is set.")
    parser.add_argument("--task",         default=None,
                        help="Task name (e.g. pneumonia). Ignored if --results_file is set.")
    parser.add_argument("--results_file", default=None,
                        help="Path to a specific VQA results CSV. "
                             "If omitted, inferred from --model and --task.")
    parser.add_argument("--results_dir",  default="results",
                        help="Directory to search for results CSVs (default: results/)")
    parser.add_argument("--out_dir",      default="results",
                        help="Where to save the metrics CSV (default: results/)")
    args = parser.parse_args()

    if args.results_file:
        csv_paths = [args.results_file]
    elif args.model and args.task:
        slug = resolve_model_slug(args.model)
        csv_paths = [os.path.join(args.results_dir, f"{slug}_{args.task}_vqa_results.csv")]
    else:
        import glob
        csv_paths = sorted(glob.glob(os.path.join(args.results_dir, "*_vqa_results.csv")))
        if not csv_paths:
            raise FileNotFoundError(f"No *_vqa_results.csv files in {args.results_dir}. "
                                    "Run evaluate_vqa.py first.")

    dfs = []
    for p in csv_paths:
        print(f"Loading {p}")
        dfs.append(pd.read_csv(p))

    df = pd.concat(dfs, ignore_index=True)

    if "pred" not in df.columns:
        raise ValueError("Results CSV has no 'pred' column. Re-run evaluate_vqa.py without --no_judge.")

    metrics_df = analyze(df)
    print_summary(metrics_df)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "prompt_sensitivity.csv")
    metrics_df.to_csv(out_path, index=False)
    print(f"\nSaved metrics -> {out_path}")


if __name__ == "__main__":
    main()
