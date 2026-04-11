# pip install -U transformers datasets accelerate scikit-learn matplotlib pillow pandas numpy torch

import argparse
import io
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoProcessor, AutoModelForImageTextToText


def pretty_model_name(model_key: str) -> str:
    mapping = {
        "medgemma": "MedGemma",
        "gemma": "Gemma-3",
        "llava": "LLaVA-1.6",
        "llava-med": "LLaVA-Med",
    }
    return mapping.get(model_key, model_key.replace("_", " ").title())


def pretty_dataset_name(dataset_key: str) -> str:
    mapping = {
        "brain_tumor": "Brain Tumor",
        "pneumonia": "Pneumonia",
        "idc": "Invasive Ductal Carcinoma (IDC)",
        "skin_cancer": "Skin Cancer",
    }
    return mapping.get(dataset_key, dataset_key.replace("_", " ").title())


def pretty_label(label: str) -> str:
    mapping = {
        "no_tumor": "No Tumor",
        "tumor": "Tumor",
        "no_pneumonia": "No Pneumonia",
        "pneumonia": "Pneumonia",
        "no_idc": "No IDC",
        "idc": "IDC",
        "benign": "Benign",
        "malignant": "Malignant",
    }
    return mapping.get(label, label.replace("_", " ").title())


def set_pretty_plot_style():
    # Try Georgia first; fall back gracefully if unavailable
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    if "Georgia" in available_fonts:
        plt.rcParams["font.family"] = "Georgia"
    else:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Times"]

    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["axes.titleweight"] = "semibold"
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 12


# =========================
# Configuration
# =========================

MODEL_REGISTRY = {
    "medgemma": {
        "hf_id": "google/medgemma-4b-it",
        "family": "gemma",
    },
    "gemma": {
        "hf_id": "google/gemma-3-4b-it",
        "family": "gemma",
    },
    "llava": {
        "hf_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "family": "llava",
    },
    "llava-med": {
        "hf_id": "wnkh/llava-med-v1.5-mistral-7b-hf",
        "family": "llava",
    },
}


DATASET_REGISTRY = {
    "brain_tumor": {
        "hf_id": "tanzuhuggingface/brainmri",
    },
    "pneumonia": {
        "hf_id": "danjacobellis/chexpert",
    },
    "idc": {
        "hf_id": "dbzadnen/breast-histopathology-images",
    },
    "skin_cancer":{
        "hf_id": "Falah/skin-cancer"
    },
}


@dataclass
class RunConfig:
    model_key: str
    dataset_key: str
    split: str = "train"
    max_per_class: Optional[int] = 300
    seed: int = 42
    batch_size: int = 4
    image_size: Optional[int] = None   # optional manual resize; None = processor default
    use_projected_features: bool = False   # False = raw vision encoder pooled features
    layer_idx: int = -1                # for vision tower hidden states if available
    tsne_perplexity: int = 30
    tsne_random_state: int = 42
    pca_dim: int = 50
    save_dir: str = "./embedding_runs"
    frontal_only_chexpert: bool = True  # recommended to reduce view confound
    device_map: str = "auto"


# =========================
# Reproducibility helpers
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Dataset loading
# =========================

def pil_to_rgb(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.fromarray(np.array(img)).convert("RGB")

def load_skin_cancer_df(split: str = "train") -> pd.DataFrame:
    splits = {
        "train": "data/train-00000-of-00001-7a2e46cfec0593e3.parquet",
        "test": "data/test-00000-of-00001-5e3e4ea58aecbdf9.parquet",
    }

    if split not in splits:
        raise ValueError(f"Unsupported split for skin_cancer: {split}")

    df_raw = pd.read_parquet(f"hf://datasets/Falah/skin-cancer/{splits[split]}")

    rows = []
    for _, ex in df_raw.iterrows():
        label = int(ex["label"])

        image_obj = ex["image"]
        if isinstance(image_obj, dict) and "bytes" in image_obj:
            image = Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")
        else:
            raise ValueError("Unexpected image format in skin_cancer parquet.")

        diagnosis = "malignant" if label == 1 else "benign"

        rows.append({
            "image": image,
            "diagnosis": diagnosis,
            "dataset": "skin_cancer",
            "meta": {"orig_label": label},
        })

    return pd.DataFrame(rows)

def load_brain_tumor_df(split: str = "train") -> pd.DataFrame:
    ds = load_dataset(DATASET_REGISTRY["brain_tumor"]["hf_id"], split=split)
    rows = []
    for ex in ds:
        label = ex["label"]
        # HF imagefolder class labels are often string names in features metadata,
        # but the actual examples may already be ints or strings depending on decode path.
        if isinstance(label, int):
            label_name = ds.features["label"].int2str(label)
        else:
            label_name = str(label)

        label_name = label_name.lower().strip()
        diagnosis = "tumor" if label_name in {"yes", "tumor", "1"} else "no_tumor"

        rows.append({
            "image": pil_to_rgb(ex["image"]),
            "diagnosis": diagnosis,
            "dataset": "brain_tumor",
            "meta": {"orig_label": label_name},
        })
    return pd.DataFrame(rows)


def load_chexpert_df(split: str = "train", frontal_only: bool = True) -> pd.DataFrame:
    ds = load_dataset(DATASET_REGISTRY["pneumonia"]["hf_id"], split=split)
    rows = []

    for ex in ds:
        # Pneumonia labels:
        # 0: unlabeled, 1: uncertain, 2: absent, 3: present
        pneumonia = ex["Pneumonia"] if "Pneumonia" in ex else ex["pneumonia"]
        if pneumonia not in [2, 3]:
            continue

        if frontal_only:
            fl = ex.get("Frontal/Lateral", None)
            # Viewer shows 0 = Frontal, 1 = Lateral
            if fl is not None and fl != 0:
                continue

        diagnosis = "pneumonia" if pneumonia == 3 else "no_pneumonia"

        img = ex.get("image", None)
        if img is None:
            continue

        rows.append({
            "image": pil_to_rgb(img),
            "diagnosis": diagnosis,
            "dataset": "pneumonia",
            "meta": {
                "pneumonia_raw": int(pneumonia),
                "sex": ex.get("Sex", None),
                "age": ex.get("Age", None),
                "view": ex.get("Frontal/Lateral", None),
                "ap_pa": ex.get("AP/PA", None),
                "path": ex.get("Path", None),
            },
        })

    return pd.DataFrame(rows)


def load_idc_df(split: str = "train") -> pd.DataFrame:
    ds = load_dataset(DATASET_REGISTRY["idc"]["hf_id"], split=split)
    rows = []
    for ex in ds:
        label = int(ex["label"])
        diagnosis = "idc" if label == 1 else "no_idc"
        rows.append({
            "image": pil_to_rgb(ex["image"]),
            "diagnosis": diagnosis,
            "dataset": "idc",
            "meta": {"orig_label": label},
        })
    return pd.DataFrame(rows)


def load_dataset_df(dataset_key: str, split: str, frontal_only_chexpert: bool = True) -> pd.DataFrame:
    if dataset_key == "brain_tumor":
        return load_brain_tumor_df(split=split)
    elif dataset_key == "pneumonia":
        return load_chexpert_df(split=split, frontal_only=frontal_only_chexpert)
    elif dataset_key == "idc":
        return load_idc_df(split=split)
    elif dataset_key == "skin_cancer":                  # NEW
        return load_skin_cancer_df(split=split)
    else:
        raise ValueError(f"Unknown dataset_key: {dataset_key}")


def balanced_subsample(df: pd.DataFrame, label_col: str, max_per_class: Optional[int], seed: int) -> pd.DataFrame:
    if max_per_class is None:
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    parts = []
    for label, group in df.groupby(label_col):
        n = min(max_per_class, len(group))
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts, axis=0).sample(frac=1, random_state=seed).reset_index(drop=True)


# =========================
# Model loading
# =========================

def load_model_and_processor(model_key: str, device_map: str = "auto"):
    model_info = MODEL_REGISTRY[model_key]
    hf_id = model_info["hf_id"]

    processor = AutoProcessor.from_pretrained(hf_id, use_fast=True)

    model = AutoModelForImageTextToText.from_pretrained(
        hf_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )
    model.eval()
    return model, processor, model_info["family"], hf_id


# =========================
# Vision feature extraction
# =========================

def move_to_model_device(batch: Dict, model):
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def preprocess_images_for_processor(
    images: List[Image.Image],
    image_size: Optional[int] = None
) -> List[Image.Image]:
    processed = []
    for img in images:
        img = pil_to_rgb(img)
        if image_size is not None:
            img = img.resize((image_size, image_size))
        processed.append(img)
    return processed


def get_default_prompt(family: str) -> str:
    if family == "gemma":
        return "<start_of_image> Describe this image in great detail and pay attention to any relevant pixels. Imagine a physician will use your description."
    elif family == "llava":
        return "Describe this image in great detail and pay attention to any relevant pixels. Imagine a physician will use your description."
    else:
        return "Describe this image."



@torch.no_grad()
def extract_llava_features_single(
    model,
    processor,
    image: Image.Image,
    layer_idx: int = -1,
    use_projected_features: bool = False,
):
    """
    Returns a single embedding of shape [D] for one image.

    Handles LLaVA processors that return either:
      [1, C, H, W]
    or
      [1, N, C, H, W]
    """
    image = preprocess_images_for_processor([image])[0]

    if hasattr(processor, "image_processor"):
        proc = processor.image_processor(images=image, return_tensors="pt")
    else:
        proc = processor(images=image, return_tensors="pt")

    pixel_values = proc["pixel_values"]

    # LLaVA variants sometimes return [B, N, C, H, W]
    if pixel_values.ndim == 5:
        # for a single image, take the first crop / patch group
        # shape: [1, N, C, H, W] -> [N, C, H, W]
        pixel_values = pixel_values[0]

        # if multiple crops remain, use the first one
        # shape: [N, C, H, W] -> [1, C, H, W]
        if pixel_values.ndim == 4:
            pixel_values = pixel_values[:1]

    elif pixel_values.ndim == 3:
        # just in case: [C, H, W] -> [1, C, H, W]
        pixel_values = pixel_values.unsqueeze(0)

    if pixel_values.ndim != 4:
        raise RuntimeError(f"Unexpected LLaVA pixel_values shape: {tuple(pixel_values.shape)}")

    vision_tower = None
    if hasattr(model, "vision_tower"):
        vision_tower = model.vision_tower
    elif hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        vision_tower = model.model.vision_tower

    if vision_tower is None:
        raise RuntimeError("Could not find vision_tower on this LLaVA model.")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    pixel_values = pixel_values.to(device=device, dtype=dtype)

    out = vision_tower(pixel_values, output_hidden_states=True)

    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        hs = out.hidden_states[layer_idx]
    elif hasattr(out, "last_hidden_state"):
        hs = out.last_hidden_state
    else:
        raise RuntimeError("Vision tower did not return hidden states.")

    if use_projected_features:
        projector = None
        if hasattr(model, "multi_modal_projector"):
            projector = model.multi_modal_projector
        elif hasattr(model, "model") and hasattr(model.model, "multi_modal_projector"):
            projector = model.model.multi_modal_projector
        elif hasattr(model, "multimodal_projector"):
            projector = model.multimodal_projector
        elif hasattr(model, "model") and hasattr(model.model, "multimodal_projector"):
            projector = model.model.multimodal_projector

        if projector is not None:
            hs = projector(hs)

    pooled = hs.mean(dim=1)
    return pooled[0].float().cpu().numpy()


@torch.no_grad()
def extract_gemma_features_single(
    model,
    processor,
    image: Image.Image,
    text: str = "<start_of_image> Describe this image in great detail and pay attention to any relevant pixels. Imagine a physician will use your description.",
    use_projected_features: bool = False,
):
    """
    Returns a single embedding of shape [D] for one image-text pair.
    """
    image = preprocess_images_for_processor([image])[0]

    proc = processor(
        text=[text],
        images=[[image]],
        return_tensors="pt",
        padding=True,
    )
    proc = move_to_model_device(proc, model)

    if hasattr(model, "get_image_features"):
        kwargs = {}
        if "pixel_values" in proc:
            kwargs["pixel_values"] = proc["pixel_values"]
        if "image_sizes" in proc:
            kwargs["image_sizes"] = proc["image_sizes"]

        feats = model.get_image_features(**kwargs)

        # unwrap model output objects (e.g. BaseModelOutputWithPooling)
        if not torch.is_tensor(feats):
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            elif hasattr(feats, "last_hidden_state"):
                feats = feats.last_hidden_state
            else:
                raise RuntimeError(f"Cannot extract tensor from get_image_features output: {type(feats)}")

        if feats.ndim == 3:
            pooled = feats.mean(dim=1)
        elif feats.ndim == 2:
            pooled = feats
        else:
            raise RuntimeError(f"Unexpected Gemma feature shape: {tuple(feats.shape)}")

        return pooled[0].float().cpu().numpy()

    vision_tower = None
    if hasattr(model, "vision_tower"):
        vision_tower = model.vision_tower
    elif hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        vision_tower = model.model.vision_tower

    if vision_tower is None:
        raise RuntimeError("Could not find get_image_features or vision_tower on this Gemma model.")

    vision_inputs = {}
    if "pixel_values" in proc:
        vision_inputs["pixel_values"] = proc["pixel_values"]
    if "image_sizes" in proc:
        vision_inputs["image_sizes"] = proc["image_sizes"]

    out = vision_tower(**vision_inputs, output_hidden_states=True)

    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        hs = out.hidden_states[-1]
    elif hasattr(out, "last_hidden_state"):
        hs = out.last_hidden_state
    else:
        raise RuntimeError("Gemma vision tower did not return hidden states.")

    pooled = hs.mean(dim=1)
    return pooled[0].float().cpu().numpy()


def extract_embedding_single(
    model,
    processor,
    family: str,
    image: Image.Image,
    text: Optional[str] = None,
    layer_idx: int = -1,
    use_projected_features: bool = False,
):
    if text is None:
        text = get_default_prompt(family)

    if family == "llava":
        return extract_llava_features_single(
            model=model,
            processor=processor,
            image=image,
            layer_idx=layer_idx,
            use_projected_features=use_projected_features,
        )
    elif family == "gemma":
        return extract_gemma_features_single(
            model=model,
            processor=processor,
            image=image,
            text=text,
            use_projected_features=use_projected_features,
        )
    else:
        raise ValueError(f"Unknown model family: {family}")


def embedding_extraction_one_by_one(
    model,
    processor,
    family: str,
    df: pd.DataFrame,
    layer_idx: int = -1,
    use_projected_features: bool = False,
):
    all_embs = []
    all_labels = []

    prompt = get_default_prompt(family)

    for _, row in df.iterrows():
        image = row["image"]
        label = row["diagnosis"]

        emb = extract_embedding_single(
            model=model,
            processor=processor,
            family=family,
            image=image,
            text=prompt,
            layer_idx=layer_idx,
            use_projected_features=use_projected_features,
        )

        all_embs.append(emb)
        all_labels.append(label)

    X = np.stack(all_embs, axis=0)
    y = np.array(all_labels)
    return X, y

# =========================
# t-SNE + plotting
# =========================

def compute_tsne(X: np.ndarray, pca_dim: int = 50, perplexity: int = 30, random_state: int = 42):
    # PCA first is usually much more stable for high-d features
    if X.shape[1] > pca_dim:
        X_reduced = PCA(n_components=min(pca_dim, X.shape[0] - 1), random_state=random_state).fit_transform(X)
    else:
        X_reduced = X

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5, X.shape[0] // 3)),
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(X_reduced)


def default_palette(labels: List[str]) -> Dict[str, str]:
    base = [
        "tab:blue", "tab:red", "tab:green", "tab:orange",
        "tab:purple", "tab:brown", "tab:pink", "tab:gray"
    ]
    uniq = sorted(set(labels))
    return {lab: base[i % len(base)] for i, lab in enumerate(uniq)}


def plot_tsne(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str,
    palette: Optional[Dict[str, str]] = None,
):
    set_pretty_plot_style()
    palette = palette or default_palette(labels.tolist())

    plt.figure(figsize=(10.5, 7.5))

    for lab in sorted(set(labels.tolist())):
        idx = labels == lab
        plt.scatter(
            coords[idx, 0],
            coords[idx, 1],
            label=pretty_label(lab),
            s=55,
            alpha=0.82,
            c=palette[lab],
            edgecolors="white",
            linewidths=0.6,
        )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(alpha=0.2, linewidth=0.6)
    plt.legend(loc="best", frameon=True, fancybox=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.show()


# =========================
# Full pipeline
# =========================

def run_embedding_pipeline(cfg: RunConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 1) Load dataset
    df = load_dataset_df(
        dataset_key=cfg.dataset_key,
        split=cfg.split,
        frontal_only_chexpert=cfg.frontal_only_chexpert,
    )
    df = balanced_subsample(df, label_col="diagnosis", max_per_class=cfg.max_per_class, seed=cfg.seed)

    print(f"Loaded {len(df)} samples")
    print(df["diagnosis"].value_counts(dropna=False))

    # 2) Load model
    model, processor, family, hf_id = load_model_and_processor(
        model_key=cfg.model_key,
        device_map=cfg.device_map,
    )
    print(f"Loaded model: {hf_id} | family={family}")

    # 3) Extract embeddings
    X, y = embedding_extraction_one_by_one(
        model=model,
        processor=processor,
        family=family,
        df=df,
        layer_idx=cfg.layer_idx,
        use_projected_features=cfg.use_projected_features,
    )
    print("Embedding matrix:", X.shape)

    # 4) Save embeddings
    tag = f"{cfg.model_key}__{cfg.dataset_key}"
    npz_path = os.path.join(cfg.save_dir, f"{tag}_embeddings.npz")
    np.savez_compressed(
        npz_path,
        embeddings=X,
        labels=y,
    )

    labels_csv = os.path.join(cfg.save_dir, f"{tag}_labels.csv")
    pd.DataFrame({"diagnosis": y}).to_csv(labels_csv, index=False)

    # 5) t-SNE
    coords = compute_tsne(
        X,
        pca_dim=cfg.pca_dim,
        perplexity=cfg.tsne_perplexity,
        random_state=cfg.tsne_random_state,
    )

    # 6) Plot
    fig_path = os.path.join(cfg.save_dir, f"{tag}_tsne.png")
    plot_tsne(
        coords=coords,
        labels=y,
        title=f"{pretty_model_name(cfg.model_key)} Vision Embeddings on {pretty_dataset_name(cfg.dataset_key)} Dataset",
        save_path=fig_path,
    )

    return {
        "df": df,
        "embeddings": X,
        "labels": y,
        "coords": coords,
        "embedding_file": npz_path,
        "labels_file": labels_csv,
        "plot_file": fig_path,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Run medical vision embedding + t-SNE pipeline"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model name (required unless --all is set)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=list(DATASET_REGISTRY.keys()),
        help="Task/dataset name (required unless --all is set)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-run t-SNE and plot for all existing embedding files in save_dir",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./embedding_runs",
        help="Directory for embeddings and plots (default: ./embedding_runs)",
    )
    args = parser.parse_args()

    if args.all:
        # Find all saved embedding .npz files and re-plot each one
        npz_files = [
            f for f in os.listdir(args.save_dir) if f.endswith("_embeddings.npz")
        ]
        if not npz_files:
            print(f"No *_embeddings.npz files found in {args.save_dir}")
            return
        for fname in sorted(npz_files):
            tag = fname.replace("_embeddings.npz", "")
            model_key, dataset_key = tag.split("__", 1)
            npz_path = os.path.join(args.save_dir, fname)
            print(f"\nProcessing {tag} ...")
            data = np.load(npz_path, allow_pickle=True)
            X, y = data["embeddings"], data["labels"]
            coords = compute_tsne(X)
            fig_path = os.path.join(args.save_dir, f"{tag}_tsne.png")
            plot_tsne(
                coords=coords,
                labels=y,
                title=f"{pretty_model_name(model_key)} Vision Embeddings on {pretty_dataset_name(dataset_key)} Dataset",
                save_path=fig_path,
            )
            print(f"Saved plot -> {fig_path}")
    else:
        if not args.model or not args.task:
            parser.error("--model and --task are required unless --all is set")
        cfg = RunConfig(
            model_key=args.model,
            dataset_key=args.task,
            save_dir=args.save_dir,
        )
        print(cfg)
        run_embedding_pipeline(cfg)


if __name__ == "__main__":
    main()
