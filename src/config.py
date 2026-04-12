from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class DatasetConfig:
    # Identity
    name: str        # slug used in filenames/cache keys, e.g. "brain_tumor"
    class_name: str  # e.g. "BrainTumor"
    pos_name: str    # human label for positive class, e.g. "tumor"
    neg_name: str    # human label for negative class, e.g. "no_tumor"

    # HuggingFace dataset
    hf_dataset: str
    hf_revision: Optional[str]
    image_col: str = "image"

    # Label scheme - two modes:
    #   simple:   single column (label_col) with integer pos_value / neg_value
    #   chexpert: separate presence columns (pos_col, neg_col) where present_value = "present"
    label_col: Optional[str] = None
    pos_value: int = 1
    neg_value: int = 0
    pos_col: Optional[str] = None   # chexpert mode
    neg_col: Optional[str] = None   # chexpert mode
    present_value: int = 3          # chexpert mode

    # Prompts, split by type so evaluate_vqa.py can filter
    #   "yesno" - open-ended yes/no questions
    #   "ab"    - multiple-choice A/B questions (two orderings)
    yesno_prompts: List[str] = field(default_factory=list)
    ab_prompts: List[str] = field(default_factory=list)
    describe_prompts: List[str] = field(default_factory=list)

    # GPT judge config (direct VQA judging)
    judge_system_prompt: str = ""
    judge_pos_token: str = "POSITIVE"   # token GPT returns for a positive prediction
    judge_neg_token: str = "NEGATIVE"   # token GPT returns for a negative prediction

    # Desc eval prompt variants (templates with {desc} placeholder)
    desc_prompt_variants: Dict[str, str] = field(default_factory=dict)

BRAIN_TUMOR = DatasetConfig(
    name="brain_tumor",
    class_name="BrainTumor",
    pos_name="tumor",
    neg_name="no_tumor",
    hf_dataset="tanzuhuggingface/brainmri",
    hf_revision="refs/convert/parquet",
    label_col="label",
    pos_value=1,
    neg_value=0,
    yesno_prompts=[
        "Is this image abnormal?",
        "Is this image normal?",
        "Does this image contain evidence of tumors?",
        "Does this image contain evidence of tumors? Consider signs of irregular masses and other atypical morphology.",
    ],
    ab_prompts=[
        "Which does this image show? Choose one answer.\nA) Tumor\nB) No tumor",
        "Which does this image show? Choose one answer.\nA) No tumor\nB) Tumor",
    ],
    describe_prompts=[
        "Describe this image in great detail and pay attention to any relevant pixels. Imagine a physician will use your description.",
        "Describe this image in great detail and pay attention to any relevant pixels. Imagine a physician will use your description to diagnose brain cancer.",
    ],
    judge_system_prompt="""\
You are a strict binary classifier. You will be given a response from a vision-language model
that was shown a brain MRI image and asked a question about it.

Your job: determine whether the model's response indicates the image shows a TUMOR
(i.e., contains a brain tumor / is abnormal / shows a mass or lesion) or NO_TUMOR
(i.e., no tumor / normal / healthy brain).

Rules:
- If the model says the image IS abnormal, DOES contain a tumor, shows a mass/lesion,
  answers "A) Brain tumor", or otherwise indicates a tumor is present -> respond exactly: TUMOR
- If the model says the image is normal, healthy, no tumor, no mass, no abnormality,
  answers "B) No brain tumor", or otherwise indicates no tumor -> respond exactly: NO_TUMOR
- If the response is truly ambiguous, hedged equally, or refuses to answer -> respond exactly: AMBIGUOUS

You must respond with exactly one word: TUMOR, NO_TUMOR, or AMBIGUOUS. Nothing else.""",
    judge_pos_token="TUMOR",
    judge_neg_token="NO_TUMOR",
    desc_prompt_variants={
        "one_word": (
            "You are a careful radiologist.\n"
            "Given the following brain MRI image description, answer ONLY whether a brain tumor is present.\n"
            "Return a single word: Yes or No.\n\nDescription:\n{desc}\n"
        ),
        "tight": (
            "Task: Decide if a brain tumor is present.\n"
            "Output must be Yes or No.\n\nDescription:\n{desc}\n"
        ),
        "disease": (
            "Task: Decide if disease is present.\n"
            "Output must be Yes or No.\n\nDescription:\n{desc}\n"
        ),
    },
)

PNEUMONIA = DatasetConfig(
    name="pneumonia",
    class_name="Pneumonia",
    pos_name="pneumonia",
    neg_name="normal",
    hf_dataset="danjacobellis/chexpert",
    hf_revision=None,
    pos_col="Pneumonia",
    neg_col="No Finding",
    present_value=3,
    yesno_prompts=[
        "Is this image abnormal?",
        "Is this image normal?",
        "Does this image contain evidence of pneumonia?",
        "Does this image contain evidence of pneumonia? Consider signs of consolidation, air bronchograms, and silhouettes.",
    ],
    ab_prompts=[
        "Which does this image show? Choose one answer.\nA) Pneumonia\nB) No pneumonia",
        "Which does this image show? Choose one answer.\nA) No pneumonia\nB) Pneumonia",
    ],
    describe_prompts=[
        "Describe this image in great detail and pay attention to any relevant pixels. Imagine a physician will use your description.",
        "Describe this image in great detail and pay attention to any relevant pixels. Imagine a physician will use your description to diagnose pneumonia.",
    ],
    judge_system_prompt="""\
You are a strict binary classifier. You will be given a response from a vision-language model
that was shown a chest X-ray and asked a question about it.

Your job: determine whether the model's response indicates the image is POSITIVE
(i.e., contains pneumonia / is abnormal / shows evidence of pneumonia) or NEGATIVE
(i.e., no pneumonia / normal / benign).

Rules:
- If the model says the X-ray IS abnormal, HAS pneumonia, DOES contain evidence of pneumonia,
  answers "A) Pneumonia" or "B) Pneumonia" (depending on answer order),
  or otherwise indicates a positive diagnosis -> respond exactly: POSITIVE
- If the model says the X-ray is normal, negative, does NOT contain evidence of pneumonia,
  answers "A) No Pneumonia" or "B) No Pneumonia" (depending on answer order),
  or otherwise indicates no pneumonia -> respond exactly: NEGATIVE
- If the response is truly ambiguous, hedged equally, or refuses to answer -> respond exactly: AMBIGUOUS

You must respond with exactly one word: POSITIVE, NEGATIVE, or AMBIGUOUS. Nothing else.""",
    judge_pos_token="POSITIVE",
    judge_neg_token="NEGATIVE",
    desc_prompt_variants={
        "one_word": (
            "You are a careful clinician.\n"
            "Given the following chest X-ray description, answer ONLY whether pneumonia is present.\n"
            "Return a single word: Yes or No.\n\nDescription:\n{desc}\n"
        ),
        "tight": (
            "Task: Decide if pneumonia is present.\n"
            "Output must be Yes or No.\n\nDescription:\n{desc}\n"
        ),
        "disease": (
            "Task: Decide if disease is present.\n"
            "Output must be Yes or No.\n\nDescription:\n{desc}\n"
        ),
    },
)

SKIN_CANCER = DatasetConfig(
    name="skin_cancer",
    class_name="SkinCancer",
    pos_name="malignant",
    neg_name="benign",
    hf_dataset="Falah/skin-cancer",
    hf_revision="refs/convert/parquet",
    label_col="label",
    pos_value=1,
    neg_value=0,
    yesno_prompts=[
        "Is this skin lesion abnormal?",
        "Is this skin lesion normal?",
        "Does this image contain evidence of malignant skin cancer?",
        "Does this image contain evidence of malignant skin cancer? Consider signs of irregular borders, asymmetry, uneven pigmentation, and other atypical morphology.",
    ],
    ab_prompts=[
        "Which does this image show? Choose one answer.\nA) Malignant\nB) Benign",
        "Which does this image show? Choose one answer.\nA) Benign\nB) Malignant",
    ],
    describe_prompts=[
        "Describe this image in great detail and pay attention to any relevant pixels. Imagine a physician will use your description.",
        "Describe this image in great detail and pay attention to any relevant pixels. Imagine a physician will use your description to diagnose skin cancer.",
    ],
    judge_system_prompt="""\
You are a strict binary classifier. You will be given a response from a vision-language model
that was shown a skin lesion image and asked a question about it.

Your job: determine whether the model's response indicates the image is MALIGNANT
(i.e., contains skin cancer / is abnormal / shows malignancy) or BENIGN
(i.e., no skin cancer / normal / benign).

Rules:
- If the model says the lesion IS abnormal, IS malignant, DOES contain evidence of malignant skin cancer,
  answers "A) Malignant" or "B) Malignant" (depending on answer order), or otherwise indicates malignancy
  -> respond exactly: MALIGNANT
- If the model says the lesion is normal, benign, does NOT contain evidence of malignant skin cancer,
  answers "A) Benign" or "B) Benign" (depending on answer order), or otherwise indicates benign
  -> respond exactly: BENIGN
- If the response is truly ambiguous, hedged equally, or refuses to answer -> respond exactly: AMBIGUOUS

You must respond with exactly one word: MALIGNANT, BENIGN, or AMBIGUOUS. Nothing else.""",
    judge_pos_token="MALIGNANT",
    judge_neg_token="BENIGN",
    desc_prompt_variants={
        "one_word": (
            "You are a careful clinician.\n"
            "Given the following skin lesion description, answer ONLY whether skin cancer is present.\n"
            "Return a single word: Yes or No.\n\nDescription:\n{desc}\n"
        ),
        "tight": (
            "Task: Decide if skin cancer is present.\n"
            "Output must be Yes or No.\n\nDescription:\n{desc}\n"
        ),
        "disease": (
            "Task: Decide if disease is present.\n"
            "Output must be Yes or No.\n\nDescription:\n{desc}\n"
        ),
    },
)

HISTO_IDC = DatasetConfig(
    name="histo_idc",
    class_name="IDC",
    pos_name="malignant",
    neg_name="benign",
    hf_dataset="dbzadnen/breast-histopathology-images",
    hf_revision=None,
    label_col="label",
    pos_value=1,
    neg_value=0,
    yesno_prompts=[
        "Is this image abnormal?",
        "Is this image normal?",
        "Does this image contain evidence of invasive ductal carcinoma?",
        "Does this image contain evidence of invasive ductal carcinoma? Consider signs of irregular masses and other atypical cell morphology.",
    ],
    ab_prompts=[
        "Which does this image show? Choose one answer.\nA) Carcinoma\nB) No carcinoma",
        "Which does this image show? Choose one answer.\nA) No carcinoma\nB) Carcinoma",
    ],
    describe_prompts=[
        "Describe this image in great detail and pay attention to any relevant features. Imagine a pathologist will use your description.",
        "Describe this image in great detail and pay attention to any relevant features. Imagine a pathologist will use your description to diagnose invasive ductal carcinoma.",
    ],
    judge_system_prompt="""\
You are a strict binary classifier. You will be given a response from a vision-language model
that was shown a breast histopathology image and asked a question about it.

Your job: determine whether the model's response indicates the image is MALIGNANT
(i.e., contains invasive ductal carcinoma / is abnormal / shows carcinoma) or BENIGN
(i.e., no carcinoma / normal / benign).

Rules:
- If the model says the image IS abnormal, IS carcinoma, DOES contain IDC, answers "A) Carcinoma",
  or otherwise indicates malignancy -> respond exactly: MALIGNANT
- If the model says the image is normal, benign, no carcinoma, does NOT contain IDC,
  answers "B) No carcinoma", or otherwise indicates benign -> respond exactly: BENIGN
- If the response is truly ambiguous, hedged equally, or refuses to answer -> respond exactly: AMBIGUOUS

You must respond with exactly one word: MALIGNANT, BENIGN, or AMBIGUOUS. Nothing else.""",
    judge_pos_token="MALIGNANT",
    judge_neg_token="BENIGN",
    desc_prompt_variants={
        "one_word": (
            "You are a careful pathologist.\n"
            "Given the following histopathology image description, answer ONLY whether invasive ductal carcinoma is present.\n"
            "Return a single word: Yes or No.\n\nDescription:\n{desc}\n"
        ),
        "tight": (
            "Task: Decide if invasive ductal carcinoma is present.\n"
            "Output must be Yes or No.\n\nDescription:\n{desc}\n"
        ),
        "disease": (
            "Task: Decide if disease is present.\n"
            "Output must be Yes or No.\n\nDescription:\n{desc}\n"
        ),
    },
)

ALL_DATASETS: Dict[str, DatasetConfig] = {
    cfg.name: cfg for cfg in [BRAIN_TUMOR, PNEUMONIA, SKIN_CANCER, HISTO_IDC]
}

# Short aliases for --model flag
MODEL_ALIASES: Dict[str, str] = {
    "medgemma": "google/medgemma-4b-it",
    "gemma":    "google/gemma-3-4b-it",
    "llava":    "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-med":"wnkh/llava-med-v1.5-mistral-7b-hf",
}

ALL_MODELS: List[str] = list(MODEL_ALIASES.keys())

K_DEFAULT = 50
GPT_JUDGE_MODEL_DEFAULT = "gpt-5"
GPT_DESC_MODEL_DEFAULT  = "gpt-5.1"
