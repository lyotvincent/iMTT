import os, sys
from accelerate import Accelerator
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tokenizers.processors import BertProcessing
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.parameters import *


def train_tokenizer_bpe():
    paths = [str(x) for x in Path(SAMPLES_PATH).glob("**/distinct_*.csv")]
    print(paths)
    file_contents = set()
    for path in paths:
        with open(path, 'r', encoding='utf-8') as file:
            while l := file.readline().strip():
                file_contents.add(l)

    file_contents = list(file_contents)
    # 创建一个BPE模型
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    # 创建一个BPE训练器，并设置参数
    trainer = BpeTrainer(
        vocab_size=9,  # 你可以根据需要调整词汇表的大小
        min_frequency=1,  # 最小频率，只有当subword出现次数大于等于这个值时，才会被添加到词汇表中
        special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>", "A", "T", "C", "G"],  # 特殊token，用于处理未知字符
        max_token_length=2,
    )
    # 训练分词器
    tokenizer.train_from_iterator([file_contents], trainer=trainer)

    # Saving the files to disk
    if not os.path.exists(TOKEN_DIR_BASE):
        os.makedirs(TOKEN_DIR_BASE)
    tokenizer.model.save(TOKEN_DIR_BASE)
    tokenizer.save(os.path.join(TOKEN_DIR_BASE, "tokenizer.json"))


def load_tokenizer_bpe():
    # Load the tokenizerfrom tokenizers.implementations import ByteLevelBPETokenizer
    ROOT = Path(__file__).resolve().parent.parent
    tokenizer_path = os.path.join(TOKEN_DIR_BASE, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(tokenizer.encode("CCCTAAACCCTAAACCCTAACCCTAACCCTAACCCTAACCC").tokens)
    print(tokenizer.encode("CCCTAAACCCTAAACCCTAACCCTAACCCTAACCCTAACCC"))
    print(tokenizer.encode("CCCTAAA<mask>CCCTAACCCTAACCCTAACCCTAACCC").tokens)
    print(tokenizer.encode("CCCTAAA<mask>CCCTAACCCTAACCCTAACCCTAACCC"))
    return tokenizer


if __name__ == "__main__":
    train_tokenizer_bpe()
    load_tokenizer_bpe()


