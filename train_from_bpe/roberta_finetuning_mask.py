import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from transformers import DataCollatorWithPadding
from transformers import TrainerCallback
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import random
import numpy as np

from parameters import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

"""
请你帮我写一下load_dataset的代码，输入数据文件在SAMPLES_PATH下，正例样本的文件名是positive_samples_HEK293T.csv，反例样本的文件名是negative_samples_HEK293T.csv，文件内每一行是一个样本DNA序列。
并且样本要经过tokenizer处理再输入模型，这部分代码也要补齐，tokenizer的路径是TOKEN_DIR。
上面读取的文件中，我的正例有15357个，反例有31889个。请你帮我通过增加正例的方式，变成正例:反例=1:1的形式。
"""
# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained(TOKEN_DIR)

# Function to load and preprocess the dataset
def load_and_preprocess_data(samples_path, positive_file, negative_file):
    # Load positive and negative samples
    positive_samples = pd.read_csv(os.path.join(samples_path, positive_file), header=None, names=["sequence"])
    negative_samples = pd.read_csv(os.path.join(samples_path, negative_file), header=None, names=["sequence"])

    # Add labels
    positive_samples["label"] = 1
    negative_samples["label"] = 0

    # Concatenate the datasets
    data = pd.concat([positive_samples, negative_samples], ignore_index=True)

    # Split the dataset into train and test sets with stratification
    train_data, test_data = train_test_split(data, test_size=0.2, stratify=data["label"], random_state=42)

    # Balance the training dataset by oversampling positive samples
    positive_train_samples = train_data[train_data["label"] == 1]
    negative_train_samples = train_data[train_data["label"] == 0]
    positive_train_samples_upsampled = positive_train_samples.sample(n=len(negative_train_samples), replace=True, random_state=42)
    balanced_train_data = pd.concat([positive_train_samples_upsampled, negative_train_samples], ignore_index=True)

    # Balance the validation set
    positive_test_samples = test_data[test_data["label"] == 1]
    negative_test_samples = test_data[test_data["label"] == 0]
    min_samples = min(len(positive_test_samples), len(negative_test_samples))
    balanced_test_data = pd.concat([positive_test_samples.sample(n=min_samples, random_state=42), 
                                    negative_test_samples.sample(n=min_samples, random_state=42)], 
                                   ignore_index=True)

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(balanced_train_data)
    test_dataset = Dataset.from_pandas(balanced_test_data)
    
    # Tokenize the sequences
    def tokenize_function(examples):
        MAX_LEN = 16
        tokenized = tokenizer(examples["sequence"], padding="max_length", max_length=MAX_LEN)
        # Create custom attention mask
        attention_mask = []
        for input_ids in tokenized["input_ids"]:
            tokens_len = len(input_ids)
            mask = torch.ones(tokens_len, tokens_len)
            for i in range(tokens_len):
                if 1 < i < tokens_len-1:
                    mask[i, i-1] = 0
                if 0 < i < tokens_len-2:
                    mask[i, i+1] = 0
            # mask[0,1:tokens_len] = 0
            # mask[1:tokens_len-1,0] = 0
            # mask[-1,0:tokens_len-1] = 0
            # mask[1:tokens_len-1,-1] = 0
            padded_mask = torch.zeros((MAX_LEN, MAX_LEN))
            padded_mask[:tokens_len, :tokens_len] = mask
            for i in range(tokens_len, MAX_LEN):
                padded_mask[i,i] = 1
            attention_mask.append(padded_mask)
        tokenized["attention_mask"] = attention_mask
        return tokenized

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
    # print(tokenized_dataset)

    # Calculate token lengths
    # df = tokenized_train_dataset.to_pandas()
    # lengths = df['input_ids'].apply(len)
    # max_length = lengths.quantile(1)  # Choose 95th percentile as max_length
    # print(f"max token length: {max_length}")

    return tokenized_train_dataset, tokenized_test_dataset

# Load and preprocess the dataset
train_dataset, test_dataset = load_and_preprocess_data(SAMPLES_PATH, "positive_samples_HEK293T.csv", "negative_samples_HEK293T.csv")

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
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])
                self.epochs2.append(state.epoch)

# Initialize the loss history callback
loss_history_callback = LossHistoryCallback()

# Define the model
model = RobertaForSequenceClassification.from_pretrained(PRETRAINED_MODEL_DIR2, num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir=FINETUNING_MODEL_DIR3,
    overwrite_output_dir=True,
    per_device_train_batch_size=32,
    num_train_epochs=40,
    save_steps=0.0125,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=0.0125,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=0.00625,
    logging_dir=os.path.join(FINETUNING_MODEL_DIR3, 'logs'),  # Directory for storing logs
    report_to="tensorboard",  # Report to TensorBoard
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions[:, 1]

    acc = accuracy_score(labels, preds)
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)

    return {
        'accuracy': acc,
        'auroc': auroc,
        'auprc': auprc,
    }

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[loss_history_callback],
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

trainer.save_model(FINETUNING_MODEL_DIR3)

# Plot the loss curves
# plt.plot(loss_history_callback.epochs1, loss_history_callback.train_losses, label="Training Loss")
# plt.plot(loss_history_callback.epochs2, loss_history_callback.eval_losses, label="Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training and Validation Loss Curve")
# plt.legend()
# plt.savefig(os.path.join(FINETUNING_MODEL_DIR3, "loss_vs_valloss_curve.png"), format="png", dpi=300, bbox_inches="tight")

