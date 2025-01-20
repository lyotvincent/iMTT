import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_PATH = os.path.join(ROOT, "data", "samples")
PSEUDO_SAMPLES_PATH = os.path.join(ROOT, "data", "pseudo_labeled_samples")

TOKEN_DIR_BASE = os.path.join(ROOT, "train_from_base", "content", "tokenizer")
MAX_LEN_BASE = 64
# tokenizer 手动先把 C-tracts 前后加上空格，再用BPE分割
TOKEN_DIR_BPE = os.path.join(ROOT, "train_from_bpe", "content", "tokenizer_bpe")
MAX_LEN_BPE = 22

# * BPE model
PRETRAINED_MODEL_DIR_BPE_4L8HT2 = os.path.join(ROOT, "train_from_bpe", "content", "imotifBERT_4L8HT2", "pretrained_model")
FINETUNING_MODEL_DIR_BPE_4L8HT2 = os.path.join(ROOT, "train_from_bpe", "content", "imotifBERT_4L8HT2", 'fine_tuned_model_P')

FINETUNING_MODEL_DIR_BPE_TEST = os.path.join(ROOT, "train_from_bpe", "content", "imotifBERT_4L8HT2", 'fine_tuned_model_TEST')


# * Base model
PRETRAINED_MODEL_DIR_BASE = os.path.join(ROOT, "train_from_base", "content", "imotifBERT", "pretrained_model")
FINETUNING_MODEL_DIR_BASE = os.path.join(ROOT, "train_from_base", "content", "imotifBERT", 'fine_tuned_model_P')

FINETUNING_MODEL_DIR_BASE_TEST = os.path.join(ROOT, "train_from_base", "content", "imotifBERT", 'fine_tuned_model_TEST')


# * Dual tower model
SAVED_MODEL_DIR_3D8H256D_P4 = os.path.join(ROOT, "train_two_tower_model", "content", "imotifTower_3D8H256D_P4")
SAVED_MODEL_DIR_SEMI_V2 = os.path.join(ROOT, "train_two_tower_model", "content", "imotifTower_semi_v2")

SAVED_MODEL_DIR_TEST = os.path.join(ROOT, "train_two_tower_model", "content", "imotifTower_TEST")

