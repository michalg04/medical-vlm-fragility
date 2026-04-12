# Fragility in Medically Fine-Tuned Vision–Language Models

This repository contains code for the paper:

**"Is There Knowledge Left to Extract? Evidence of Fragility in Medically Fine-Tuned Vision-Language Models"**

<img width="1112" height="358" alt="dataset_overview" src="https://github.com/user-attachments/assets/b1815f20-1b50-413a-ae40-a233d70542ef" />

---

We evaluate whether medical fine-tuning improves robustness and clinical reasoning in VLMs across:
- Closed-form VQA evaluation across prompt variants
- Prompt sensitivity analysis across prompt types
- Refusal-aware evaluation
- Description-based pipeline (image → text → diagnosis)
- Vision encoder embedding analysis (t-SNE separability)

---

## Models

| Alias | Model |
|---|---|
| `medgemma` | MedGemma 4B |
| `gemma` | Gemma-3 4B Instruct |
| `llava` | LLaVA v1.6 (Mistral 7B) |
| `llava-med` | LLaVA-Med v1.5 |

## Tasks

| Task | Modality |
|---|---|
| `brain_tumor` | Brain MRI |
| `pneumonia` | Chest X-ray |
| `skin_cancer` | Dermatoscopy |
| `histo_idc` | Histopathology |

## Metrics

- **Accuracy, sensitivity, specificity, F1** — per prompt
- **Refusal rate** — fraction of ambiguous / refused responses
- **Prompt variance** — accuracy std across prompt variants

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

## Citation

    @misc{mclaughlin2026fragilitymedicalvlms,
      title={Is There Knowledge Left to Extract? Evidence of Fragility in Medically Fine-Tuned Vision-Language Models},
      author={Oliver McLaughlin and Daniel Shubin and Carsten Eickhoff and Ritambhara Singh and William Rudman and Michal Golovanevsky},
      year={2026}
    }

