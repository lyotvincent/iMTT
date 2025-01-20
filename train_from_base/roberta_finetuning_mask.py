import torch
from sklearn.model_selection import train_test_split
import pandas as pd
# import matplotlib.pyplot as plt
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
# from transformers import DataCollatorWithPadding
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

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained(TOKEN_DIR)

def get_ctract_mask(input_ids):
    """
    input_ids is a list of ids，一维数组，每个元素是一个id
    C: 7
    """
    # mask = torch.zeros_like(input_ids) # [0] * len(input_ids)
    i, j = 0, 0
    ctract_regions = []
    tokens_len = len(input_ids)
    for ind in range(tokens_len):
        if input_ids[ind] == 7:
            # print("*", i, j)
            if i == 0:
                i = ind
            else:
                j = ind
        else:
            if j - i >= 2:
                # mask[i:j+1] = 1
                ctract_regions.append([i,j])
                # print("@", i, j)
            i, j = 0, 0
    # print(mask)
    # print(ctract_regions)
    
    mask = torch.ones(tokens_len, tokens_len)
    ctract_region_num = len(ctract_regions)
    for i in range(ctract_region_num):
        if i < ctract_region_num-1:
            region1, region2 = ctract_regions[i], ctract_regions[i+1]
            mask[region1[0]:region1[1]+1, region2[0]:region2[1]+1] = 0.
        if i > 0:
            region1, region2 = ctract_regions[i], ctract_regions[i-1]
            mask[region1[0]:region1[1]+1, region2[0]:region2[1]+1] = 0.

    input_ids = torch.Tensor(input_ids)
    pad_len = torch.sum(input_ids==3).item()
    # print(pad_len)
    mask[-pad_len:, :] = 0.
    mask[:, -pad_len:] = 0.
    for i in range(pad_len):
        mask[-1-i, -1-i] = 1.
    # print(mask)
    # for i in mask:
    #     print([int(j) for j in i.tolist()])
    return mask


# test_input_ids = torch.Tensor([0,7,7,7,7,5,5,7,5,7,7,5,7,7,7,2,7,7,7,2,7,7,2,7,7,7,1,3,3,3,3]).float()
# print(test_input_ids)
# get_ctract_mask(test_input_ids)
# exit()

# Tokenize the sequences
def tokenize_function(examples):
    MAX_LEN = 64
    tokenized = tokenizer(examples["sequence"], padding="max_length", max_length=MAX_LEN)

    attention_mask = []
    for input_ids in tokenized["input_ids"]:
        padded_mask = get_ctract_mask(input_ids)
        attention_mask.append(padded_mask)
    tokenized["attention_mask"] = attention_mask
    return tokenized

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
print(train_dataset)

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
# loss_history_callback = LossHistoryCallback()

# Define the model
model = RobertaForSequenceClassification.from_pretrained(PRETRAINED_MODEL_DIR2, num_labels=2)

# Define training arguments
NUM_EPOCHS = 100
training_args = TrainingArguments(
    output_dir=FINETUNING_MODEL_DIR3,
    overwrite_output_dir=True,
    per_device_train_batch_size=32,
    num_train_epochs=NUM_EPOCHS,
    save_steps=1/NUM_EPOCHS, # `save_steps` must be a round multiple of `eval_steps`
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=0.5/NUM_EPOCHS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=0.25/NUM_EPOCHS,
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
    # callbacks=[loss_history_callback],
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

