# tsne_analysis.py - t-SNE visualization of visual encoder embeddings.
#
# Loads one or more .pt files from extract_embeddings.py and produces
# a scatter plot colored by ground-truth label.
#
# Usage:
#   python tsne_analysis.py --model gemma --task brain_tumor
#   python tsne_analysis.py --embeddings_files results/gemma_brain_tumor_embeddings.pt \
#                                              results/medgemma_brain_tumor_embeddings.pt
#   python tsne_analysis.py --all   # runs on every .pt file in results/
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from config import ALL_DATASETS, MODEL_ALIASES


def resolve_model_slug(model_arg: str) -> str:
    full_id = MODEL_ALIASES.get(model_arg, model_arg)
    return full_id.replace("/", "_")


def run_tsne(embeddings: np.ndarray, perplexity: int = 30, n_iter: int = 1000) -> np.ndarray:
    scaled = StandardScaler().fit_transform(embeddings)
    tsne   = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                  random_state=42, init="pca")
    return tsne.fit_transform(scaled)


def plot_tsne(coords: np.ndarray, labels: list[int], label_names: list[str],
              title: str, out_path: str) -> None:
    unique_labels = sorted(set(labels))
    colours       = plt.cm.tab10(np.linspace(0, 0.5, len(unique_labels)))
    label_to_name = {lbl: name for lbl, name in zip(labels, label_names)}

    fig, ax = plt.subplots(figsize=(8, 7))
    for lbl, colour in zip(unique_labels, colours):
        mask = np.array(labels) == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colour], label=label_to_name[lbl],
                   alpha=0.6, s=18, linewidths=0)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=2, framealpha=0.7)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


def process_file(pt_path: str, out_dir: str, perplexity: int) -> None:
    print(f"\nLoading {pt_path}")
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)

    embeddings  = obj["embeddings"].float().numpy()
    labels      = obj["labels"]
    label_names = obj["label_names"]
    model_id    = obj.get("model", "unknown")
    task        = obj.get("task", "unknown")

    print(f"  shape={embeddings.shape}, task={task}, model={model_id}")

    coords = run_tsne(embeddings, perplexity=perplexity)

    title    = f"t-SNE  |  {task}  |  {model_id.split('/')[-1]}"
    basename = os.path.splitext(os.path.basename(pt_path))[0]
    out_path = os.path.join(out_dir, f"{basename}_tsne.png")
    plot_tsne(coords, labels, label_names, title, out_path)


def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization of embeddings")
    parser.add_argument("--model",            default=None,
                        help="Model alias. Used to infer filename if --embeddings_files not set.")
    parser.add_argument("--task",             default=None,
                        help="Task name. Used to infer filename if --embeddings_files not set.")
    parser.add_argument("--embeddings_files", nargs="+", default=None,
                        help="Explicit list of .pt embedding files.")
    parser.add_argument("--all",              action="store_true",
                        help="Process all *_embeddings.pt files in --results_dir.")
    parser.add_argument("--results_dir",      default="results")
    parser.add_argument("--out_dir",          default="results")
    parser.add_argument("--perplexity",       type=int, default=30)
    args = parser.parse_args()

    if args.embeddings_files:
        pt_paths = args.embeddings_files
    elif args.all:
        import glob
        pt_paths = sorted(glob.glob(os.path.join(args.results_dir, "*_embeddings.pt")))
        if not pt_paths:
            raise FileNotFoundError(f"No *_embeddings.pt files in {args.results_dir}.")
    elif args.model and args.task:
        slug     = resolve_model_slug(args.model)
        pt_paths = [os.path.join(args.results_dir, f"{slug}_{args.task}_embeddings.pt")]
    else:
        raise ValueError("Specify --model+--task, --embeddings_files, or --all.")

    os.makedirs(args.out_dir, exist_ok=True)
    for pt_path in pt_paths:
        process_file(pt_path, args.out_dir, args.perplexity)


if __name__ == "__main__":
    main()
