import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from transformers import DataCollatorWithPadding
from transformers import TrainerCallback
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
import random
import numpy as np
import sys, os

from models import CustomRobertaForSequenceClassification

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.parameters import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained(TOKEN_DIR_BASE)

# Function to load and preprocess the dataset
def load_and_preprocess_data(samples_path, positive_file, negative_file, fold_i):
    # Load positive and negative samples
    positive_samples = pd.read_csv(os.path.join(samples_path, positive_file), header=None, names=["sequence"])
    negative_samples = pd.read_csv(os.path.join(samples_path, negative_file), header=None, names=["sequence"])

    # Add labels
    positive_samples["labels"] = 1
    negative_samples["labels"] = 0

    # Concatenate the datasets
    data = pd.concat([positive_samples, negative_samples], ignore_index=True)

    # Split the dataset into train and test sets with stratification
    origin_train_data, origin_test_data = train_test_split(data, test_size=0.2, stratify=data["labels"], random_state=42)

    assert 0 <= fold_i < 5, "fold_i must be between 0 and 4"
    if fold_i == 0:
        train_data, test_data = origin_train_data, origin_test_data
    else: # fold_i in 1~4
        folds = list()
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

        for train_idx, val_idx in skf.split(origin_train_data, origin_train_data["labels"]):
            inner_val = origin_train_data.iloc[val_idx].reset_index(drop=True)

            folds.append(inner_val)

        test_data = folds[fold_i - 1]
        # 除去当前验证集之后的训练部分（= 3 折） + 原来的 origin_test_data
        other_folds = [folds[i] for i in range(4) if i != (fold_i - 1)]
        train_data = pd.concat(other_folds + [origin_test_data], ignore_index=True)

    # Balance the training dataset by oversampling positive samples
    positive_train_samples = train_data[train_data["labels"] == 1]
    negative_train_samples = train_data[train_data["labels"] == 0]
    positive_train_samples_upsampled = positive_train_samples.sample(n=len(negative_train_samples), replace=True, random_state=42)
    balanced_train_data = pd.concat([positive_train_samples_upsampled, negative_train_samples], ignore_index=True)

    # Balance the validation set
    positive_test_samples = test_data[test_data["labels"] == 1]
    negative_test_samples = test_data[test_data["labels"] == 0]
    min_samples = min(len(positive_test_samples), len(negative_test_samples))
    balanced_test_data = pd.concat([positive_test_samples.sample(n=min_samples, random_state=42), 
                                    negative_test_samples.sample(n=min_samples, random_state=42)], 
                                   ignore_index=True)

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(balanced_train_data)
    test_dataset = Dataset.from_pandas(balanced_test_data)

    # Tokenize the sequences
    def tokenize_function(examples):
        tokenized_batch = tokenizer(examples["sequence"], padding="max_length", max_length=MAX_LEN_BASE)
        # 基于attention_mask 创建一个新 loss mask 把 0 index 和 最后一个 1 变成 0
        loss_mask = list()
        for i in tokenized_batch["attention_mask"]:
            temp = i.copy()
            temp[0] = 0.
            temp[sum(i)-1] = 0.
            loss_mask.append(temp)
        tokenized_batch["loss_mask"] = loss_mask
        return tokenized_batch

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    return tokenized_train_dataset, tokenized_test_dataset

# Load and preprocess the dataset
train_dataset, test_dataset = load_and_preprocess_data(SAMPLES_PATH,
                                                       "distinct_positive_samples_HEK293T.csv",
                                                       "distinct_negative_samples_HEK293T.csv",
                                                       fold_i=4)

# Define the model
# from transformers import RobertaConfig
# config = RobertaConfig(
#     vocab_size=9,
#     hidden_size=256,
#     max_position_embeddings=128,
#     num_attention_heads=8,
#     num_hidden_layers=4,
#     type_vocab_size=1,
#     position_embedding_type="relative_key",
#     num_labels=2,
# )
# model = RobertaForSequenceClassification(config=config)
model = RobertaForSequenceClassification.from_pretrained(PRETRAINED_MODEL_DIR_BASE, num_labels=2)

# Define training arguments
NUM_EPOCHS = 800
BATCH_SIZE = 512 # 最开始是 Batch size 为 512, epoch 为 400
training_args = TrainingArguments(
    output_dir=FINETUNING_MODEL_DIR_BASE_TEST,
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
    logging_dir=os.path.join(FINETUNING_MODEL_DIR_BASE_TEST, 'logs'),  # Directory for storing logs
    report_to="tensorboard",  # Report to TensorBoard
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions[:, 1]

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

trainer.save_model(FINETUNING_MODEL_DIR_BASE_TEST)

# 对验证集进行评估
eval_results = trainer.evaluate(eval_dataset=test_dataset)

# 输出评估结果
print("Evaluation results:", eval_results)

# 保存评估结果
with open(os.path.join(FINETUNING_MODEL_DIR_BASE_TEST, "best_model_eval_metrics.txt"), "w") as f:
    f.write(str(eval_results))

