import os
from accelerate import Accelerator
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tokenizers.processors import BertProcessing
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from parameters import *


def train_tokenizer_example():
    # Step 3: Training a tokenizer
    # paths = [str(x) for x in Path(".").glob("**/*.txt")]
    POS_HEK293T_PATH = os.path.join(ROOT, "data", "samples", "positive_samples_HEK293T.csv")
    paths = [POS_HEK293T_PATH]

    file_contents = []
    for path in paths:
        try:
            with open(path, 'r', encoding='utf-8') as file:
                file_contents.append(file.read())
        except Exception as e:
            print(f"Error reading {path}: {e}")

    # Join the contents into a single string
    text = "\n".join(file_contents)

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()
    # Customize training
    tokenizer.train_from_iterator([text], vocab_size=1_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<mask>",
    ])

    # Step 4: Saving the files to disk
    token_dir = os.path.join(ROOT, "src", "content", "imotifBERT")
    if not os.path.exists(token_dir):
        os.makedirs(token_dir)
    tokenizer.save_model(token_dir)

def train_tokenizer_bpe():
    SAMPLES_PATH = os.path.join(ROOT, "data", "samples")
    paths = [str(x) for x in Path(SAMPLES_PATH).glob("**/*.csv")]
    file_contents = set()
    for path in paths:
        try:
            with open(path, 'r', encoding='utf-8') as file:
                while l := file.readline().strip():
                    file_contents.add(l)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    file_contents = list(file_contents)
    # 创建一个BPE模型
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    # 创建一个BPE训练器，并设置参数
    trainer = BpeTrainer(
        vocab_size=1000,  # 你可以根据需要调整词汇表的大小
        min_frequency=6,  # 最小频率，只有当subword出现次数大于等于这个值时，才会被添加到词汇表中
        special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],  # 特殊token，用于处理未知字符
        max_token_length=12
    )
    # 训练分词器
    tokenizer.train_from_iterator([file_contents], trainer=trainer)

    # Saving the files to disk
    if not os.path.exists(TOKEN_DIR):
        os.makedirs(TOKEN_DIR)
    tokenizer.model.save(TOKEN_DIR)
    tokenizer.save(os.path.join(TOKEN_DIR, "tokenizer.json"))

def load_tokenizer_example():
    # Load the tokenizerfrom tokenizers.implementations import ByteLevelBPETokenizer
    token_dir = os.path.join(ROOT, "src", "content", "imotifBERT")
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(token_dir, "vocab.json"),
        os.path.join(token_dir, "merges.txt"),
    )
    print(tokenizer.encode("CCCTAAACCCTAAACCCTAACCCTAACCCTAACCCTAACCC").tokens)
    print(tokenizer.encode("CCCTAAACCCTAAACCCTAACCCTAACCCTAACCCTAACCC"))
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=512)
    print(tokenizer.encode("CCCTAAACCCTAAACCCTAACCCTAACCCTAACCCTAACCC").tokens)
    print(tokenizer.encode("CCCTAAACCCTAAACCCTAACCCTAACCCTAACCCTAACCC"))


def load_tokenizer_bpe():
    # Load the tokenizerfrom tokenizers.implementations import ByteLevelBPETokenizer
    ROOT = Path(__file__).resolve().parent.parent
    tokenizer_path = os.path.join(TOKEN_DIR, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(tokenizer.encode("CCCTAAACCCTAAACCCTAACCCTAACCCTAACCCTAACCC").tokens)
    print(tokenizer.encode("CCCTAAACCCTAAACCCTAACCCTAACCCTAACCCTAACCC"))
    print(tokenizer.encode("CCCTAAA<mask>CCCTAACCCTAACCCTAACCCTAACCC").tokens)
    print(tokenizer.encode("CCCTAAA<mask>CCCTAACCCTAACCCTAACCCTAACCC"))
    return tokenizer

def combine_samples():
    SAMPLES_PATH = os.path.join(ROOT, "data", "samples")
    paths = [str(x) for x in Path(SAMPLES_PATH).glob("**/*.csv")]
    file_contents = list()
    for path in paths:
        try:
            with open(path, 'r', encoding='utf-8') as file:
                while l := file.readline().strip():
                    file_contents.append(l)
        except Exception as e:
            print(f"Error reading {path}: {e}")
    with open(os.path.join(SAMPLES_PATH, "imotif_repetitive.csv"), "w") as f:
        for line in file_contents:
            f.write(f"{line}\n")

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    # train_tokenizer_bpe()
    # load_tokenizer_bpe()
    combine_samples()


