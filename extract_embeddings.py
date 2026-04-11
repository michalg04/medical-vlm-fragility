# extract_embeddings.py - extract visual encoder embeddings from a VLM.
#
# For each image in the dataset manifest, runs it through the model's vision
# encoder and saves the mean-pooled features. Used by tsne_analysis.py.
#
# Output: results/{model}_{task}_embeddings.pt
#   keys: model, task, row_ids, labels, label_names, embeddings (Tensor[N, D])
#
# Usage:
#   python extract_embeddings.py --model gemma --task brain_tumor
#   python extract_embeddings.py --model medgemma --task pneumonia
import argparse
import os

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

from config import ALL_DATASETS, MODEL_ALIASES, DatasetConfig
from vlm_utils import to_pil


def _get_vision_encoder(model):
    for attr in ("vision_tower", "vision_model", "visual_encoder"):
        enc = getattr(model, attr, None)
        if enc is not None:
            return enc
    # Gemma3 / MedGemma structure
    if hasattr(model, "language_model"):
        for attr in ("vision_tower", "vision_model"):
            enc = getattr(model, attr, None)
            if enc is not None:
                return enc
    raise AttributeError(
        f"Cannot find vision encoder in {type(model).__name__}. "
        "Inspect model architecture and update _get_vision_encoder()."
    )


def load_hf_df(cfg: DatasetConfig) -> pd.DataFrame:
    kwargs = {"revision": cfg.hf_revision} if cfg.hf_revision else {}
    ds = load_dataset(cfg.hf_dataset, **kwargs)
    split = "train" if "train" in ds else list(ds.keys())[0]
    return pd.DataFrame(ds[split])


def resolve_model(model_arg: str) -> str:
    return MODEL_ALIASES.get(model_arg, model_arg)


def main():
    parser = argparse.ArgumentParser(description="Extract visual embeddings")
    parser.add_argument("--model",       required=True,
                        help=f"Model alias or HF ID. Aliases: {list(MODEL_ALIASES)}")
    parser.add_argument("--task",        required=True,
                        help=f"Task name. Available: {list(ALL_DATASETS)}")
    parser.add_argument("--dataset_csv", default="medical_vqa_dataset.csv")
    parser.add_argument("--out_dir",     default="results")
    parser.add_argument("--batch_size",  type=int, default=8)
    args = parser.parse_args()

    if args.task not in ALL_DATASETS:
        raise ValueError(f"Unknown task '{args.task}'. Available: {list(ALL_DATASETS)}")

    cfg      = ALL_DATASETS[args.task]
    model_id = resolve_model(args.model)
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = pd.read_csv(args.dataset_csv)
    task_df  = manifest[manifest["task"] == args.task].reset_index(drop=True)
    if task_df.empty:
        raise RuntimeError(f"No rows for task '{args.task}' in {args.dataset_csv}.")

    print(f"Task:   {args.task}  ({len(task_df)} images)")
    print(f"Model:  {model_id}")

    print("Loading HuggingFace dataset for images...")
    hf_df = load_hf_df(cfg)

    dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(model_id)
    model     = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()

    try:
        vision_enc = _get_vision_encoder(model)
    except AttributeError as e:
        print(f"[warn] {e}")
        print("Falling back to full model forward pass for embeddings.")
        vision_enc = None

    embeddings_list: list[torch.Tensor] = []
    row_ids, labels, label_names = [], [], []

    with torch.inference_mode():
        for _, row in tqdm(task_df.iterrows(), total=len(task_df), desc="embedding"):
            image = to_pil(hf_df.iloc[int(row["row_id"])][cfg.image_col])

            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if vision_enc is not None:
                out    = vision_enc(**inputs)
                # last_hidden_state: (1, seq_len, D)  ->  mean pool -> (D,)
                hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                emb    = hidden.mean(dim=1).squeeze(0).float().cpu()
            else:
                # Fallback: use pooler output or first output
                out = model(**inputs, output_hidden_states=True)
                hidden = out.hidden_states[-1] if hasattr(out, "hidden_states") else out[0]
                emb = hidden.mean(dim=1).squeeze(0).float().cpu()

            embeddings_list.append(emb)
            row_ids.append(int(row["row_id"]))
            labels.append(int(row["label"]))
            label_names.append(str(row["label_name"]))

    embeddings = torch.stack(embeddings_list)  # (N, D)
    print(f"Embeddings shape: {embeddings.shape}")

    model_slug = args.model.replace("/", "_")
    out_path   = os.path.join(args.out_dir, f"{model_slug}_{args.task}_embeddings.pt")
    torch.save({
        "model":       model_id,
        "task":        args.task,
        "row_ids":     row_ids,
        "labels":      labels,
        "label_names": label_names,
        "embeddings":  embeddings,
    }, out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
