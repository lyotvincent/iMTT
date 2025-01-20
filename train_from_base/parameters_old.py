import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_PATH = os.path.join(ROOT, "data", "samples")
TOKEN_DIR = os.path.join(ROOT, "train_from_base", "content", "tokenizer")
PSEUDO_SAMPLES_PATH = os.path.join(ROOT, "data", "pseudo_labeled_samples")

# model 4 layers 8 heads
PRETRAINED_MODEL_DIR = os.path.join(ROOT, "train_from_base", "content", "imotifBERT", "pretrained_model")
FINETUNING_MODEL_DIR = os.path.join(ROOT, "train_from_base", "content", "imotifBERT", 'fine_tuned_model')


# model 8 layers 16 heads
PRETRAINED_MODEL_DIR2 = os.path.join(ROOT, "train_from_base", "content", "imotifBERT2", "pretrained_model")
FINETUNING_MODEL_DIR2 = os.path.join(ROOT, "train_from_base", "content", "imotifBERT2", 'fine_tuned_model')

# model 8 layers 16 heads mask
# use PRETRAINED_MODEL_DIR2
FINETUNING_MODEL_DIR3 = os.path.join(ROOT, "train_from_base", "content", "imotifBERT3", 'fine_tuned_model')
