# Medical VQA Evaluation Pipeline

| Task | Dataset | Positive class |
|---|---|---|
| `brain_tumor` | tanzuhuggingface/brainmri | tumor |
| `pneumonia` | danjacobellis/chexpert | pneumonia |
| `skin_cancer` | Falah/skin-cancer | malignant |
| `histo_idc` | dbzadnen/breast-histopathology-images | IDC |

Models available `medgemma`, `gemma`, `llava`, `llava-med`

---

## Requirements

```
pip install -r requirements.txt
```

Set your OpenAI API key for the judging/description-eval steps:

```
export OPENAI_API_KEY=sk-...
```

---

## Quick Start

Run a simple VQA evaluation:

```
python src/evaluate_vqa.py \
  --model gemma \
  --task pneumonia \
  --prompt_type ab \
  --one_word True
```

---

## Full Pipeline

### Step 1 - Create Dataset

Sample K images per class for each task and save the metadata manifest:

```
python src/create_dataset.py
```

Output:

```
medical_vqa_dataset.csv   # columns: task, row_id, label, label_name
```

Options:
- `--k 50`  - samples per class (default: 50)
- `--tasks brain_tumor pneumonia`  - subset of tasks

---

### Step 2 - Run Closed-Form VQA

Run a VLM on one task and judge responses with GPT:

```
python src/evaluate_vqa.py \
  --model medgemma \
  --task brain_tumor \
  --all_prompts True
```

Output:

```
results/{model}_{task}_vqa_results.csv
```

Options:
- `--prompt_type yesno|ab|all`  - which prompt category to use (default: all)
- `--one_word True`  - append "Answer in one word." to each prompt
- `--no_judge`  - skip GPT judging, save raw responses only
- `--judge_model gpt-5`  - GPT model for judging

---

### Step 3 - Prompt Sensitivity Analysis

Compute accuracy, sensitivity, specificity, F1, and variance across prompts:

```
python src/analyze_prompts.py \
  --model medgemma \
  --task pneumonia
```

Output:

```
results/prompt_sensitivity.csv
```

Computes:
- Per-prompt accuracy, sensitivity, specificity, F1
- Accuracy variance across prompts (stability metric)
- Refusal / ambiguous rates

---

### Step 4 - Description-Based Pipeline

**Generate descriptions:**

```
python src/generate_descriptions.py \
  --model gemma \
  --task pneumonia
```

Output:

```
results/{model}_{task}_descriptions.csv
```

**Evaluate descriptions:**

```
python src/evaluate_descriptions.py \
  --model gemma \
  --task pneumonia
```

Output:

```
results/{model}_{task}_desc_eval.csv
```

Pipeline:

```
image -> VLM description -> GPT Yes/No diagnosis -> accuracy metrics
```

Three GPT prompt variants are tested per description:
- `one_word` - role-aware specialist framing
- `tight` - minimal disease-specific framing
- `disease` - generic disease-presence framing

---

### Step 5 - Embedding Analysis

Extract visual encoder features and visualise with t-SNE:

```
python src/tsne_analysis.py \
  --model gemma \
  --task brain_tumor
```

Or re-run t-SNE plots for all existing embedding files at once:

```
python src/tsne_analysis.py --all
```

Output:

```
results/{model}_{task}_embeddings_tsne.png
```

---

Intermediate files are plain CSVs so every step is independently inspectable
and re-runnable. GPT API calls are cached to JSON to avoid redundant requests.
