# Fragility in Medically Fine-Tuned Vision–Language Models

This repository contains code for the paper:

**"Is There Knowledge Left to Extract? Evidence of Fragility in Medically Fine-Tuned Vision-Language Models"**


<img width="1112" height="358" alt="dataset_overview" src="https://github.com/user-attachments/assets/b1815f20-1b50-413a-ae40-a233d70542ef" />

---

## Overview

We evaluate whether medical fine-tuning improves robustness and clinical reasoning in vision-language models (VLMs).

Key components:
- Closed-form VQA evaluation across prompt variants
- Prompt sensitivity analysis across prompt types
- Refusal-aware evaluation (F1_all vs F1_nr)
- Description-based pipeline (image → text → diagnosis)
- Vision encoder embedding analysis (t-SNE separability)

---

## Supported Models

- **llava** — LLaVA v1.6 (Mistral 7B)
- **llava-med** — LLaVA-Med v1.5
- **gemma** — Gemma-3 4B Instruct
- **medgemma** — MedGemma 4B

---

## Tasks

- Brain tumor detection (MRI)
- Pneumonia detection (Chest X-ray)
- Skin cancer classification (Dermatoscopy)
- Histopathology classification (IDC)

---


## Metrics

- **F1_all** — counts refusals as incorrect
- **F1_nr** — excludes refusals
- **Refusal rate**
- **Prompt variance (± std)**

---


## Citation

    @misc{mclaughlin2026fragilitymedicalvlms,
      title={Is There Knowledge Left to Extract? Evidence of Fragility in Medically Fine-Tuned Vision-Language Models},
      author={Oliver McLaughlin and Daniel Shubin and Carsten Eickhoff and Ritambhara Singh and William Rudman and Michal Golovanevsky},
      year={2026}
    }
