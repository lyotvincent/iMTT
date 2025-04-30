# from transformers.models.t5.modeling_t5 import T5LayerSelfAttention, T5LayerCrossAttention, T5LayerFF, T5LayerNorm, T5ClassificationHead
from transformers import RobertaConfig, RobertaTokenizer, Trainer, TrainingArguments
import torch
import torch.nn as nn
from torch.nn import Dropout, ReLU, GELU
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset
import random
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, recall_score, precision_score, f1_score
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_two_tower_model.two_tower_models import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

tokenizer1 = RobertaTokenizer.from_pretrained(TOKEN_DIR_BASE)
tokenizer2 = RobertaTokenizer.from_pretrained(TOKEN_DIR_BPE)


# Function to load and preprocess the dataset
def load_and_preprocess_data_v1(positive_file, negative_file, pseudo_pos_file=None, pseudo_neg_file=None, add_pseudo=False):
    # Load positive and negative samples
    positive_samples = pd.read_csv(os.path.join(SAMPLES_PATH, positive_file), header=None, names=["sequence"])
    negative_samples = pd.read_csv(os.path.join(SAMPLES_PATH, negative_file), header=None, names=["sequence"])

    # Add labels
    positive_samples["labels"] = 1
    negative_samples["labels"] = 0

    # Concatenate the datasets
    data = pd.concat([positive_samples, negative_samples], ignore_index=True)

    # Split the dataset into train and test sets with stratification
    train_data, test_data = train_test_split(data, test_size=0.2, stratify=data["labels"], random_state=42)
    print(f"train:test = {len(train_data)}:{len(test_data)} = {len(train_data) / len(test_data)}")

    # Balance the training dataset by oversampling positive samples
    positive_train_samples = train_data[train_data["labels"] == 1]
    negative_train_samples = train_data[train_data["labels"] == 0]
    print(f"train pos:neg = {len(positive_train_samples)}:{len(negative_train_samples)} = {len(positive_train_samples) / len(negative_train_samples)}")
    positive_train_samples_upsampled = positive_train_samples.sample(n=len(negative_train_samples), replace=True, random_state=42)
    balanced_train_data = pd.concat([positive_train_samples_upsampled, negative_train_samples], ignore_index=True)
    if add_pseudo:
        pseudo_pos_samples = pd.read_csv(os.path.join(PSEUDO_SAMPLES_PATH, pseudo_pos_file), header=None, names=["sequence"])
        pseudo_neg_samples = pd.read_csv(os.path.join(PSEUDO_SAMPLES_PATH, pseudo_neg_file), header=None, names=["sequence"])
        pseudo_pos_samples["labels"] = 1
        pseudo_neg_samples["labels"] = 0
        print(f"pseudo pos:neg = {len(pseudo_pos_samples)}:{len(pseudo_neg_samples)} = {len(pseudo_pos_samples) / len(pseudo_neg_samples)}")
        # pseudo_pos_samples_upsampled = pseudo_pos_samples.sample(n=len(pseudo_neg_samples), replace=True, random_state=42)
        balanced_train_data = pd.concat([balanced_train_data, pseudo_pos_samples, pseudo_neg_samples], ignore_index=True)

    # Balance the validation set
    positive_test_samples = test_data[test_data["labels"] == 1]
    negative_test_samples = test_data[test_data["labels"] == 0]
    print(f"test pos:neg = {len(positive_test_samples)}:{len(negative_test_samples)} = {len(positive_test_samples) / len(negative_test_samples)}")
    min_samples = min(len(positive_test_samples), len(negative_test_samples))
    balanced_test_data = pd.concat([positive_test_samples.sample(n=min_samples, random_state=42), 
                                    negative_test_samples.sample(n=min_samples, random_state=42)], 
                                   ignore_index=True)
    print(f"balanced train:test = {len(balanced_train_data)}:{len(balanced_test_data)} = {len(balanced_train_data) / len(balanced_test_data)}")

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(balanced_train_data)
    test_dataset = Dataset.from_pandas(balanced_test_data)

    # Tokenize the sequences
    def tokenize_function(examples):
        tokens1 = tokenizer1(examples["sequence"], padding="max_length", max_length=MAX_LEN_BASE)
        tokens2 = tokenizer2(examples["sequence"], padding="max_length", max_length=MAX_LEN_BPE)
        return {"input_ids_1": tokens1["input_ids"], "input_ids_2": tokens2["input_ids"]}
        # return {"input_ids_1": tokens1["input_ids"], "input_ids_2": tokens2["input_ids"], "attention_mask_1": tokens1["attention_mask"], "attention_mask_2": tokens2["attention_mask"]}

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    return tokenized_train_dataset, tokenized_test_dataset

def load_and_preprocess_data_v2(positive_file, negative_file, pseudo_pos_file=None, pseudo_neg_file=None):
    '''
    用positive伪标签数据，补足原本标注数据的正例数量，同时减少标注负例数量，使得加上伪标签负例后，正负例数量相等
    '''
    # Load positive and negative samples
    positive_samples = pd.read_csv(os.path.join(SAMPLES_PATH, positive_file), header=None, names=["sequence"])
    negative_samples = pd.read_csv(os.path.join(SAMPLES_PATH, negative_file), header=None, names=["sequence"])

    # Add labels
    positive_samples["labels"] = 1
    negative_samples["labels"] = 0

    # Concatenate the datasets
    data = pd.concat([positive_samples, negative_samples], ignore_index=True)

    # Split the dataset into train and test sets with stratification
    train_data, test_data = train_test_split(data, test_size=0.2, stratify=data["labels"], random_state=42)
    print(f"train:test = {len(train_data)}:{len(test_data)} = {len(train_data) / len(test_data)}")

    # Balance the training dataset by oversampling positive samples
    positive_train_samples = train_data[train_data["labels"] == 1]
    negative_train_samples = train_data[train_data["labels"] == 0]
    print(f"train pos:neg = {len(positive_train_samples)}:{len(negative_train_samples)} = {len(positive_train_samples) / len(negative_train_samples)}")
    pseudo_pos_samples = pd.read_csv(os.path.join(PSEUDO_SAMPLES_PATH, pseudo_pos_file), header=None, names=["sequence"])
    pseudo_neg_samples = pd.read_csv(os.path.join(PSEUDO_SAMPLES_PATH, pseudo_neg_file), header=None, names=["sequence"])
    pseudo_pos_samples["labels"] = 1
    pseudo_neg_samples["labels"] = 0
    print(f"pseudo pos:neg = {len(pseudo_pos_samples)}:{len(pseudo_neg_samples)} = {len(pseudo_pos_samples) / len(pseudo_neg_samples)}")
        # pseudo_pos_samples_upsampled = pseudo_pos_samples.sample(n=len(pseudo_neg_samples), replace=True, random_state=42)
    neg_samples_downsampled = negative_train_samples.sample(
        n=len(positive_train_samples)+len(pseudo_pos_samples)-len(pseudo_neg_samples),
        random_state=42
    )
    balanced_train_data = pd.concat([positive_train_samples, pseudo_pos_samples, neg_samples_downsampled, pseudo_neg_samples], ignore_index=True)

    # Balance the validation set
    positive_test_samples = test_data[test_data["labels"] == 1]
    negative_test_samples = test_data[test_data["labels"] == 0]
    print(f"test pos:neg = {len(positive_test_samples)}:{len(negative_test_samples)} = {len(positive_test_samples) / len(negative_test_samples)}")
    min_samples = min(len(positive_test_samples), len(negative_test_samples))
    balanced_test_data = pd.concat([positive_test_samples.sample(n=min_samples, random_state=42), 
                                    negative_test_samples.sample(n=min_samples, random_state=42)], 
                                   ignore_index=True)
    print(f"balanced train:test = {len(balanced_train_data)}:{len(balanced_test_data)} = {len(balanced_train_data) / len(balanced_test_data)}")

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(balanced_train_data)
    test_dataset = Dataset.from_pandas(balanced_test_data)

    # Tokenize the sequences
    def tokenize_function(examples):
        tokens1 = tokenizer1(examples["sequence"], padding="max_length", max_length=MAX_LEN_BASE)
        tokens2 = tokenizer2(examples["sequence"], padding="max_length", max_length=MAX_LEN_BPE)
        return {"input_ids_1": tokens1["input_ids"], "input_ids_2": tokens2["input_ids"]}
        # return {"input_ids_1": tokens1["input_ids"], "input_ids_2": tokens2["input_ids"], "attention_mask_1": tokens1["attention_mask"], "attention_mask_2": tokens2["attention_mask"]}

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    return tokenized_train_dataset, tokenized_test_dataset


train_dataset, test_dataset = load_and_preprocess_data_v2(
    "distinct_positive_samples_HEK293T.csv",
    "distinct_negative_samples_HEK293T.csv",
    "pseudo_positive_samples_HEK293T.csv",
    "pseudo_negative_samples_HEK293T.csv",
    # add_pseudo=True,
)
print(train_dataset)

config = RobertaConfig(
    vocab_size_1=9,
    vocab_size_2=10_000,
    hidden_size=256, # embedding size
    max_position_embeddings=128,
    num_decoder_layers=3,
    num_attention_heads=8,
    intermediate_size=3072,
    type_vocab_size=1,
    pad_token_id=3,
    bos_token_id=0,
    eos_token_id=1,
    position_embedding_type="relative_key",
)

model = DualTowerForSequenceClassification_visual(config, is_pretrained=True, pretrained_version=3)
# print(model)

# Define training arguments
NUM_EPOCHS = 100
BATCH_SIZE = 512 # 最开始设置为 BATCH_SIZE=32, 为了维持steps数量相同，和 NUM_EPOCHS=100 同比变化
training_args = TrainingArguments(
    output_dir=SAVED_MODEL_DIR_TEST,
    overwrite_output_dir=True,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    save_steps=0.5/NUM_EPOCHS, # `save_steps` must be a round multiple of `eval_steps`
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=0.5/NUM_EPOCHS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=0.25/NUM_EPOCHS,
    logging_dir=os.path.join(SAVED_MODEL_DIR_TEST, 'logs'),  # Directory for storing logs
    report_to="tensorboard",  # Report to TensorBoard
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)
    probs = pred.predictions[0][:, 1]

    acc = accuracy_score(labels, preds)
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)

    return {
        'accuracy': acc,
        'auroc': auroc,
        'auprc': auprc,
        'recall': recall,
        'precision': precision,
        'f1': f1,
    }

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    # callbacks=[loss_history_callback],
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

trainer.save_model(SAVED_MODEL_DIR_TEST)

# 对验证集进行评估
eval_results = trainer.evaluate(eval_dataset=test_dataset)

# 输出评估结果
print("Evaluation results:", eval_results)

# 保存评估结果
with open(os.path.join(SAVED_MODEL_DIR_TEST, "best_model_eval_metrics.txt"), "w") as f:
    f.write(str(eval_results))
