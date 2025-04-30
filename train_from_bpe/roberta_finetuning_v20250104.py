import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import RobertaTokenizer, Trainer
from transformers import TrainingArguments
from datasets import Dataset
from transformers import TrainerCallback
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, recall_score, precision_score, f1_score
import random
import numpy as np
import os, sys

from models import CustomRobertaForSequenceClassification

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.parameters import *
from custom_modules import CustomTrainer

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
tokenizer = RobertaTokenizer.from_pretrained(TOKEN_DIR_BPE)

CTRACT_DICT = {
    "CCC": 5,
    "CCCC": 6,
    "CCCCC": 7,
    "CCCCCC": 8,
    "CCCCCCC": 9,
    "CCCCCCCC": 10,
    "CCCCCCCCC": 11,
    "CCCCCCCCCC": 12,
    "CCCCCCCCCCC": 13,
    "CCCCCCCCCCCC": 14,
    "CCCCCCCCCCCCC": 15,
    "CCCCCCCCCCCCCC": 16,
    "CCCCCCCCCCCCCCC": 17,
}
CTRACT_ID_SET = set(CTRACT_DICT.values())

# Function to load and preprocess the dataset
def load_and_preprocess_data(samples_path, positive_file, negative_file):
    # Load positive and negative samples
    positive_samples = pd.read_csv(os.path.join(samples_path, positive_file), header=None, names=["sequence"])
    negative_samples = pd.read_csv(os.path.join(samples_path, negative_file), header=None, names=["sequence"])

    # Add labels
    positive_samples["labels"] = 1
    negative_samples["labels"] = 0

    # Concatenate the datasets
    data = pd.concat([positive_samples, negative_samples], ignore_index=True)

    # Split the dataset into train and test sets with stratification
    train_data, test_data = train_test_split(data, test_size=0.2, stratify=data["labels"], random_state=42)

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
        tokenized_batch = tokenizer(examples["sequence"], padding="max_length", max_length=MAX_LEN_BPE)
        # 基于attention_mask 创建一个新 loss mask 把 0 index 和 最后一个 1 变成 0
        loss_mask = list()
        for i in tokenized_batch["attention_mask"]:
            temp = i.copy()
            temp[0] = 0.
            temp[sum(i)-1] = 0.
            loss_mask.append(temp)
        tokenized_batch["loss_mask"] = loss_mask
        ctract_labels = list()
        for i in tokenized_batch["input_ids"]:
            temp = [0.] * MAX_LEN_BPE
            for j in range(MAX_LEN_BPE):
                if i[j] in CTRACT_ID_SET:
                    temp[j] = 1.
            ctract_labels.append(temp)
        tokenized_batch["ctract_signs"] = ctract_labels
        return tokenized_batch

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    return tokenized_train_dataset, tokenized_test_dataset

# Load and preprocess the dataset
train_dataset, test_dataset = load_and_preprocess_data(SAMPLES_PATH, "distinct_positive_samples_HEK293T.csv", "distinct_negative_samples_HEK293T.csv")
print(f"Train dataset length: {len(train_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")

class LossHistoryCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.epochs1 = []
        self.epochs2 = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
                self.epochs1.append(state.epoch)
                print(">>>>>")
                print(state)
                print("<<=>>")
                print(logs)
                print("<<<<<")
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])
                self.epochs2.append(state.epoch)

# Initialize the loss history callback
loss_history_callback = LossHistoryCallback()

# Define the model
model = CustomRobertaForSequenceClassification.from_pretrained(PRETRAINED_MODEL_DIR_BPE_4L8HT2, num_labels=2)

# Define training arguments
NUM_EPOCHS = 100
BATCH_SIZE = 512
training_args = TrainingArguments(
    output_dir=FINETUNING_MODEL_DIR_BPE_TEST,
    overwrite_output_dir=True,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    save_steps=0.5/NUM_EPOCHS,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=0.5/NUM_EPOCHS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss_cel",
    greater_is_better=False,
    logging_steps=0.25/NUM_EPOCHS,
    logging_dir=os.path.join(FINETUNING_MODEL_DIR_BPE_TEST, 'logs'),  # Directory for storing logs
    report_to="tensorboard",  # Report to TensorBoard
)

def compute_metrics(pred):
    labels = pred.label_ids
    # loss_cel, loss_bc, loss_4c, loss_adj, logits = pred.predictions
    loss_cel, loss_bc, logits = pred.predictions
    preds = logits.argmax(-1)
    probs = logits[:, 1]

    acc = accuracy_score(labels, preds)
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)

    return {
        'loss_cel': np.mean(loss_cel),
        'loss_bc': np.mean(loss_bc),
        # 'loss_4c': np.mean(loss_4c),
        # 'loss_adj': np.mean(loss_adj),
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

trainer.save_model(FINETUNING_MODEL_DIR_BPE_TEST)

# 对验证集进行评估
eval_results = trainer.evaluate(eval_dataset=test_dataset)

# 输出评估结果
print("Evaluation results:", eval_results)

# 保存评估结果
with open(os.path.join(FINETUNING_MODEL_DIR_BPE_TEST, "best_model_eval_metrics.txt"), "w") as f:
    f.write(str(eval_results))

