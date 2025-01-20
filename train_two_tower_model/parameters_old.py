import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_PATH = os.path.join(ROOT, "data", "samples")
PSEUDO_SAMPLES_PATH = os.path.join(ROOT, "data", "pseudo_labeled_samples")

TOKEN_DIR_BASE = os.path.join(ROOT, "train_from_base", "content", "tokenizer")
TOKEN_DIR_BPE = os.path.join(ROOT, "train_from_bpe", "content", "tokenizer2")

# model 4 Decoder layers 8 heads 512 hidden size (d_model), based on T5 module
SAVED_MODEL_DIR_4D8H512DT = os.path.join(ROOT, "train_dual_tower_model", "content", "imotifTower_4D8H512DT")
'''
config = T5Config(
    vocab_size_1=9,
    vocab_size_2=1000,
    d_model=512,       # 隐藏层的维度，通常为512
    num_labels=2,      # 分类任务的标签数量，您可以根据需要更改
    num_heads=8,       # 多头注意力机制的头数
    relative_attention_num_buckets=64,
    num_decoder_layers=4,      # Transformer Decoder层数
    d_ff=2048,         # 前馈网络的隐藏层维度
    dropout_rate=0.1   # dropout率
)
'''

# model 4 Decoder layers 8 heads 256 hidden size (d_model), based on T5 module
SAVED_MODEL_DIR_4D8H256DT = os.path.join(ROOT, "train_dual_tower_model", "content", "imotifTower_4D8H256DT")
'''
config = T5Config(
    vocab_size_1=9,
    vocab_size_2=1000,
    d_model=256,       # 隐藏层的维度，通常为512
    num_labels=2,      # 分类任务的标签数量，您可以根据需要更改
    num_heads=8,       # 多头注意力机制的头数
    relative_attention_num_buckets=64,
    num_decoder_layers=4,      # Transformer Decoder层数
    d_ff=2048,         # 前馈网络的隐藏层维度
    dropout_rate=0.1   # dropout率
)
'''


# model 3 Decoder layers 8 heads 256 hidden size (d_model), based on Roberta module, custom outlayer
SAVED_MODEL_DIR_3D8H256DRC1 = os.path.join(ROOT, "train_dual_tower_model", "content", "imotifTower_3D8H256DRC1")
'''
config = RobertaConfig(
    vocab_size_1=9,
    vocab_size_2=1000,
    hidden_size=256, # embedding size
    max_position_embeddings=128,
    num_decoder_layers=3,
    num_attention_heads=8,
    intermediate_size=3072,
    type_vocab_size=1,
    pad_token_id=3,
    bos_token_id= 0,
    eos_token_id= 1,
    position_embedding_type="relative_key",
)
'''

# model 3 Decoder layers 8 heads 256 hidden size (d_model), based on Roberta module, custom RobertaClassificationHead outlayer
SAVED_MODEL_DIR_3D8H256DRC2 = os.path.join(ROOT, "train_dual_tower_model", "content", "imotifTower_3D8H256DRC2")
'''
config 同上
'''

# ! pretrained model 4 layers 8 heads
PRETRAINED_MODEL_DIR = os.path.join(ROOT, "train_from_base", "content", "imotifBERT", "pretrained_model")
# ! pretrained model 4 layer 8 heads, relative key, TOKENIZER2 (_4L8HT2)
PRETRAINED_MODEL_DIR_4L8HT2 = os.path.join(ROOT, "train_from_bpe", "content", "imotifBERT_4L8HT2", "pretrained_model")

# model 3 Decoder layers 8 heads 256 hidden size (d_model), based on Roberta module, custom RobertaClassificationHead outlayer, load pretrained model
SAVED_MODEL_DIR_3D8H256D_P1 = os.path.join(ROOT, "train_dual_tower_model", "content", "imotifTower_3D8H256D_P1")
'''
config 同上
此模型把 Embedding 改成了 RobertaEmbeddings，并且加载了 Pretrained Model
由于改动较大，并且决定后续均使用Roberta(因为要双向模型)，项目编号省略“R”，更新编号后缀为 _3D8H256D_P
'''

# model 3 Decoder layers 8 heads 256 hidden size (d_model), based on Roberta module, custom RobertaClassificationHead outlayer
SAVED_MODEL_DIR_3D8H256D_ = os.path.join(ROOT, "train_dual_tower_model", "content", "imotifTower_3D8H256D_")
'''
config 同上
此模型同上，但没有加载 Pretrained Model
'''

# model 3 Decoder layers 8 heads 256 hidden size (d_model), based on Roberta module, custom RobertaClassificationHead outlayer, load pretrained model
SAVED_MODEL_DIR_3D8H256D_P2 = os.path.join(ROOT, "train_dual_tower_model", "content", "imotifTower_3D8H256D_P2")
'''
config 和 model 同 SAVED_MODEL_DIR_3D8H256D_P1
区别就是还加载了 cross-attention，P2的意思是 load_pretrained_weights_v2 加载。
'''

# // SAVED_MODEL_DIR_3D8H256DM_P2 = os.path.join(ROOT, "train_dual_tower_model", "content", "imotifTower_3D8H256DM_P2")
'''
config 和 model 同 SAVED_MODEL_DIR_3D8H256D_P2
只加了 tokenizer 添加 <pad> 时自动对<pad> 加的 MASK，之前 MASK 直接是 None
'''

SAVED_MODEL_DIR_3D8H256D_U = os.path.join(ROOT, "train_dual_tower_model", "content", "imotifTower_3D8H256D_U")
'''
在 SAVED_MODEL_DIR_3D8H256D_P2 模型上，加入 监督学习&半监督学习 的数据 *从头* 进行训练
'''

SAVED_MODEL_DIR_3D8H256D_I = os.path.join(ROOT, "train_dual_tower_model", "content", "imotifTower_3D8H256D_I")
"""
在 SAVED_MODEL_DIR_3D8H256D_P2 模型上，增量训练(Incremental Training)或继续训练(Continued Training)
但增量结果 (SAVED_MODEL_DIR_3D8H256D_I) 和从头训练 (SAVED_MODEL_DIR_3D8H256D_U) 的指标基本都一样。
"""

SAVED_MODEL_DIR_3D8H256D_P4 = os.path.join(ROOT, "train_dual_tower_model", "content", "imotifTower_3D8H256D_P4")
FINETUNING_MODEL_DIR = os.path.join(ROOT, "train_from_base", "content", "imotifBERT", 'fine_tuned_model')
FINETUNING_MODEL_DIR_4L8HT2 = os.path.join(ROOT, "train_from_bpe", "content", "imotifBERT_4L8HT2", 'fine_tuned_model')
"""
目前尝试的两个半监督学习都效果反而不如 监督学习的 SAVED_MODEL_DIR_3D8H256D_P2，
尝试直接将 base 和 bpe 两个模型最优的结果的参数加载，然后训练。
# ? 理由: base 和 bpe 作为双塔模型的两个塔，他们独自训练时在不同的周期达到最优，
# ? 在双塔里一起训练时，可能无法达到最优，会有木桶的短板。
# ? 所以直接拿最优参数，再训练，看看是否在两个塔都是最优的情况下，双塔模型效果会更好？
用的 load_pretrained_weights_v4 所以叫 P4
"""

SAVED_MODEL_DIR_TEST = os.path.join(ROOT, "train_dual_tower_model", "content", "imotifTower_TEST")


