import os, sys
from pathlib import Path
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.parameters import *

def add_space_around_ctracts(seq):
    start, end, last_end = -1, -1, 0
    tokens_len = len(seq)
    splitted_seq = list()
    for ind in range(tokens_len):
        if seq[ind] == "C":
            if start == -1:
                start = ind
            else:
                end = ind
        else:
            if end - start >= 2:
                # print(start, end)
                splitted_seq.append(seq[last_end: start])
                splitted_seq.append(seq[start: end+1])
                last_end = end+1
            start, end = -1, -1
    if "" in splitted_seq:
        splitted_seq.remove("")
    splitted_seq.append(seq[last_end: start])
    splitted_seq.append(seq[start: end+1])
    # print("".join(splitted_seq)==seq)
    # print(" ".join(splitted_seq))
    # print(splitted_seq)
    return " ".join(splitted_seq)


def train_tokenizer_bpe():
    SAMPLES_PATH = os.path.join(ROOT, "data", "samples")
    paths = [str(x) for x in Path(SAMPLES_PATH).glob("**/distinct_*.csv")]
    print(paths)
    file_contents = set()
    for path in paths:
        with open(path, 'r', encoding='utf-8') as file:
            while l := file.readline().strip():
                file_contents.add(add_space_around_ctracts(l))

    file_contents = list(file_contents)
    # 创建一个BPE模型
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    # 设置pre_tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    # 创建一个BPE训练器，并设置参数
    trainer = BpeTrainer(
        vocab_size=10_000,  # 你可以根据需要调整词汇表的大小
        min_frequency=3,  # 最小频率，只有当subword出现次数大于等于这个值时，才会被添加到词汇表中
        special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>",
                        "CCC", "CCCC", "CCCCC",
                        "CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC", "CCCCCCCCCC",
                        "C"*11, "C"*12, "C"*13, "C"*14, "C"*15],  # 特殊token，用于处理未知字符
        max_token_length=12
    )
    # 训练分词器
    tokenizer.train_from_iterator([file_contents], trainer=trainer)

    # Saving the files to disk
    if not os.path.exists(TOKEN_DIR_BPE):
        os.makedirs(TOKEN_DIR_BPE)
    tokenizer.model.save(TOKEN_DIR_BPE)
    tokenizer.save(os.path.join(TOKEN_DIR_BPE, "tokenizer.json"))


def load_tokenizer_bpe():
    # Load the tokenizerfrom tokenizers.implementations import ByteLevelBPETokenizer
    ROOT = Path(__file__).resolve().parent.parent
    tokenizer_path = os.path.join(TOKEN_DIR_BPE, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(tokenizer.encode("CCCTAAACCCTAAACCCTAACCCTAACCCTAACCCTAACCC").tokens)
    print(tokenizer.encode("CCCTAAACCCTAAACCCTAACCCTAACCCTAACCCTAACCC"))
    print(tokenizer.encode("CCCTAAA<mask>CCCTAACCCTAACCCTAACCCTAACCC").tokens)
    print(tokenizer.encode("CCCTAAA<mask>CCCTAACCCTAACCCTAACCCTAACCC"))
    print(tokenizer.encode("CCCCCCGCTCCCCGTAAACCCCCGGTAACCCTAACCCTAACCCCCCTTAACCC").tokens)
    return tokenizer

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent

    # test_seq = "CCCTAAACCCCTAAACCCCCTAACCCTAACCCTAACCCCCCTAACCC"
    # print(test_seq)
    # add_space_around_ctracts(test_seq)
    
    # train_tokenizer_bpe()
    load_tokenizer_bpe()


