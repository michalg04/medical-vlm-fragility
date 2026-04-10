import os
import re
import glob
import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
JUDGE_MODEL = "gpt-5"

JUDGE_SYSTEM_PROMPT = """\
You are a strict binary classifier. You will be given a response from a vision-language model
that was shown a chest X-ray and asked a question about it.

Your job: determine whether the model's response indicates the image is POSITIVE
(i.e., contains pneumonia / is abnormal / shows evidence of pneumonia) or NEGATIVE
(i.e., no pneumonia / normal / benign).

Rules:
- If the model says the X-ray IS abnormal, HAS pneumonia, DOES contain evidence of pneumonia,
  answers "A) Pneumonia" or "B) Pneumonia" (depending on answer order),
  or otherwise indicates a positive diagnosis then respond exactly: POSITIVE
- If the model says the X-ray is normal, negative, does NOT contain evidence of pneumonia,
  answers "A) No Pneumonia" or "B) No Pneumonia" (depending on answer order),
  or otherwise indicates no pneumonia then respond exactly: NEGATIVE
- If the response is truly ambiguous, hedged equally, or refuses to answer then respond exactly: AMBIGUOUS

You must respond with exactly one word: POSITIVE, NEGATIVE, or AMBIGUOUS. Nothing else."""

JUDGE_USER_TEMPLATE = """\
Original prompt given to the model:
{prompt}

Model's response:
{response}

Classification (one word):"""

# ── Helpers ─────────────────────────────────────────────────────────────────

def load_pth_files(results_dir: str) -> list[dict]:
    """Load all .pth files from a directory."""
    paths = sorted(glob.glob(os.path.join(results_dir, "*.pth")))
    if not paths:
        raise FileNotFoundError(f"No .pth files found in {results_dir}")

    data = []
    for p in paths:
        print(f"Loading {os.path.basename(p)} ...")
        obj = torch.load(p, map_location="cpu", weights_only=False)
        data.append(obj)
        print(f"  model_id: {obj['model_id']}, k={obj['k']}, prompts={len(obj['prompts'])}")
    return data


def judge_response(client: OpenAI, prompt: str, response: str) -> str:
    """Ask GPT-5 to classify a single VLM response as POSITIVE/NEGATIVE/AMBIGUOUS."""
    try:
        completion = client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                    prompt=prompt, response=response
                )},
            ],
        )

        raw = completion.choices[0].message.content.strip().upper()
        if len(repr(raw)) == 0:
            print("ERROR")

        if "POSITIVE" in raw:
            return "POSITIVE"
        elif "NEGATIVE" in raw:
            return "NEGATIVE"
        else:
            return "AMBIGUOUS"
    except Exception as e:
        print(f"  [ERROR] OpenAI call failed: {e}")
        return "ERROR"


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute accuracy, sensitivity, specificity, F1 from a df with 'label' and 'pred' cols."""
    scored = df[df["pred"].isin([0, 1])].copy()
    total = len(df)
    scored_n = len(scored)
    ambiguous_n = total - scored_n

    if scored_n == 0:
        return {"accuracy": 0, "sensitivity": 0, "specificity": 0, "f1": 0,
                "n_scored": 0, "n_ambiguous": ambiguous_n, "n_total": total}

    tp = ((scored["label"] == 1) & (scored["pred"] == 1)).sum()
    tn = ((scored["label"] == 0) & (scored["pred"] == 0)).sum()
    fp = ((scored["label"] == 0) & (scored["pred"] == 1)).sum()
    fn = ((scored["label"] == 1) & (scored["pred"] == 0)).sum()

    acc = (tp + tn) / scored_n if scored_n else 0
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0

    return {
        "accuracy": round(acc, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n_scored": scored_n,
        "n_ambiguous": ambiguous_n,
        "n_total": total,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Eval VLM chest x-ray responses with GPT-5 as judge")
    parser.add_argument("--results_dir", type=str, default=".",
                        help="Directory containing the .pth files")
    parser.add_argument("--output", type=str, default="eval_output_pneumonia.csv",
                        help="Path for the detailed per-response CSV output")
    parser.add_argument("--metrics_output", type=str, default="eval_metrics.csv",
                        help="Path for the summary metrics CSV")
    parser.add_argument("--cache", type=str, default="eval_cache.json",
                        help="JSON cache file to avoid re-judging on reruns")
    parser.add_argument("--judge_model", type=str, default=None,
                        help="Override the judge model (e.g. gpt-4o)")
    args = parser.parse_args()

    global JUDGE_MODEL
    if args.judge_model:
        JUDGE_MODEL = args.judge_model
        print(f"Using judge model: {JUDGE_MODEL}")

    client = OpenAI()  # uses OPENAI_API_KEY env var

    # Load cache if exists
    cache = {}
    if os.path.exists(args.cache):
        with open(args.cache, "r") as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached judgments from {args.cache}")

    all_data = load_pth_files(args.results_dir)

    all_rows = []
    all_metrics = []

    for obj in all_data:
        model_id = obj["model_id"]
        prompts = obj["prompts"]
        answers_df = obj["answers_df"]   # rows=prompts, cols=row_ids
        meta_df = obj["meta_df"]         # row_id, label, class_name

        label_map = dict(zip(meta_df["row_id"].tolist(), meta_df["label"].tolist()))

        row_ids = [c for c in answers_df.columns if c != "prompt"]

        print(f"\n{'='*80}")
        print(f"Evaluating: {model_id}")
        print(f"  {len(prompts)} prompts x {len(row_ids)} images = {len(prompts)*len(row_ids)} judgments")
        print(f"{'='*80}")

        for prompt_idx, prompt in enumerate(prompts):
            prompt_rows = []
            prompt_short = prompt.strip()[:80].replace("\n", " ")

            responses = answers_df.iloc[prompt_idx]

            for rid in tqdm(row_ids, desc=f"  P{prompt_idx} [{prompt_short}...]"):
                response_text = str(responses[rid])
                label = label_map[rid]

                cache_key = f"{model_id}||{prompt_idx}||{rid}"
                if cache_key in cache:
                    judgment = cache[cache_key]
                else:
                    judgment = judge_response(client, prompt, response_text)
                    cache[cache_key] = judgment
                    with open(args.cache, "w") as f:
                        json.dump(cache, f)

                pred = 1 if judgment == "POSITIVE" else (0 if judgment == "NEGATIVE" else -1)

                row = {
                    "model_id": model_id,
                    "prompt_idx": prompt_idx,
                    "prompt": prompt.strip(),
                    "row_id": rid,
                    "label": label,
                    "label_str": "pneumonia" if label == 1 else "no_pneumonia",
                    "response": response_text,
                    "gpt5_judgment": judgment,
                    "pred": pred,
                    "correct": int(pred == label) if pred in [0, 1] else None,
                }
                all_rows.append(row)
                prompt_rows.append(row)

            # Metrics for this model+prompt
            pdf = pd.DataFrame(prompt_rows)
            metrics = compute_metrics(pdf)
            metrics["model_id"] = model_id
            metrics["prompt_idx"] = prompt_idx
            metrics["prompt"] = prompt.strip()
            all_metrics.append(metrics)

            print(f"  P{prompt_idx}: acc={metrics['accuracy']}, "
                  f"sens={metrics['sensitivity']}, spec={metrics['specificity']}, "
                  f"f1={metrics['f1']}, ambig={metrics['n_ambiguous']}")

    # ── Save outputs ──
    detail_df = pd.DataFrame(all_rows)
    detail_df.to_csv(args.output, index=False)
    print(f"\nSaved detailed results -> {args.output}")

    metrics_df = pd.DataFrame(all_metrics)
    front_cols = ["model_id", "prompt_idx", "prompt", "accuracy", "sensitivity",
                  "specificity", "precision", "f1", "tp", "tn", "fp", "fn",
                  "n_scored", "n_ambiguous", "n_total"]
    metrics_df = metrics_df[[c for c in front_cols if c in metrics_df.columns]]
    metrics_df.to_csv(args.metrics_output, index=False)
    print(f"Saved metrics summary -> {args.metrics_output}")

    # ── Print summary ──
    print("\n" + "=" * 100)
    print("SUMMARY BY MODEL (averaged across prompts)")
    print("=" * 100)
    for mid in metrics_df["model_id"].unique():
        mdf = metrics_df[metrics_df["model_id"] == mid]
        print(f"\n  {mid}")
        print(f"    Accuracy:    {mdf['accuracy'].mean():.4f} (+/-{mdf['accuracy'].std():.4f})")
        print(f"    Sensitivity: {mdf['sensitivity'].mean():.4f} (+/-{mdf['sensitivity'].std():.4f})")
        print(f"    Specificity: {mdf['specificity'].mean():.4f} (+/-{mdf['specificity'].std():.4f})")
        print(f"    F1:          {mdf['f1'].mean():.4f} (+/-{mdf['f1'].std():.4f})")
        print(f"    Ambiguous:   {mdf['n_ambiguous'].mean():.1f} avg per prompt")


if __name__ == "__main__":
    main()
