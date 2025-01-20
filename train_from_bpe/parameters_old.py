import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_PATH = os.path.join(ROOT, "data", "samples")
PSEUDO_SAMPLES_PATH = os.path.join(ROOT, "data", "pseudo_labeled_samples")

# tokenizer 用BPE自然分割的，C-tracts 和 spacers 会连在一起变成一个token
TOKEN_DIR = os.path.join(ROOT, "train_from_bpe", "content", "tokenizer")

# tokenizer 手动先把 C-tracts 前后加上空格，再用BPE分割
TOKEN_DIR2 = os.path.join(ROOT, "train_from_bpe", "content", "tokenizer2")

# * 老 tokenizer1
# model 4 layers 8 heads, TOEKNIZER1
PRETRAINED_MODEL_DIR = os.path.join(ROOT, "train_from_bpe", "content", "imotifBERT2", "pretrained_model")
FINETUNING_MODEL_DIR = os.path.join(ROOT, "train_from_bpe", "content", "imotifBERT2", 'fine_tuned_model')

# model 8 layers 16 heads, TOEKNIZER1
PRETRAINED_MODEL_DIR2 = os.path.join(ROOT, "train_from_bpe", "content", "imotifBERT3", "pretrained_model")
FINETUNING_MODEL_DIR2 = os.path.join(ROOT, "train_from_bpe", "content", "imotifBERT3", 'fine_tuned_model')

# model 8 layers 16 heads masked, TOEKNIZER1
# use PRETRAINED_MODEL_DIR2
FINETUNING_MODEL_DIR3 = os.path.join(ROOT, "train_from_bpe", "content", "imotifBERT4", 'fine_tuned_model')

# * 老 tokenizer1 + relative key
# model 4 layers 8 heads, relative key, TOEKNIZER1
PRETRAINED_MODEL_DIR4 = os.path.join(ROOT, "train_from_bpe", "content", "imotifBERT5", "pretrained_model")
FINETUNING_MODEL_DIR4 = os.path.join(ROOT, "train_from_bpe", "content", "imotifBERT5", 'fine_tuned_model')

# model 8 layers 16 heads, relative key, TOEKNIZER1
PRETRAINED_MODEL_DIR5 = os.path.join(ROOT, "train_from_bpe", "content", "imotifBERT6", "pretrained_model")
FINETUNING_MODEL_DIR5 = os.path.join(ROOT, "train_from_bpe", "content", "imotifBERT6", 'fine_tuned_model')

# * 新 tokenizer2 + relative key
# model 4 layer 8 heads, relative key, TOKENIZER2 (_4L8HT2)
PRETRAINED_MODEL_DIR_4L8HT2 = os.path.join(ROOT, "train_from_bpe", "content", "imotifBERT_4L8HT2", "pretrained_model")
FINETUNING_MODEL_DIR_4L8HT2 = os.path.join(ROOT, "train_from_bpe", "content", "imotifBERT_4L8HT2", 'fine_tuned_model')

# model 8 layer 16 heads, relative key, TOKENIZER2 (_8L16HT2)
PRETRAINED_MODEL_DIR_8L16HT2 = os.path.join(ROOT, "train_from_bpe", "content", "imotifBERT_8L16HT2", "pretrained_model")
FINETUNING_MODEL_DIR_8L16HT2 = os.path.join(ROOT, "train_from_bpe", "content", "imotifBERT_8L16HT2", 'fine_tuned_model')

# model 8 layer 16 heads, relative key, TOKENIZER2, masked (_8L16HT2M)
# use PRETRAINED_MODEL_DIR_8L16HT2
FINETUNING_MODEL_DIR_8L16HT2M = os.path.join(ROOT, "train_from_bpe", "content", "imotifBERT_8L16HT2M", 'fine_tuned_model')



