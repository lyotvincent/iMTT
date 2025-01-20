
import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.parameters import *

positive_file = "distinct_positive_samples_HEK293T.csv"
negative_file = "distinct_negative_samples_HEK293T.csv"

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
positive_train_samples_upsampled = positive_train_samples.sample(n=len(negative_train_samples), replace=True, random_state=42)
print(f"train pos:neg = {len(positive_train_samples)}:{len(negative_train_samples)} = {len(positive_train_samples) / len(negative_train_samples)}")
print(f'upsampled positive samples: {len(positive_train_samples_upsampled)}')

# 输出positive train samples 和 negative train samples 到两个文件, 只输出sequence列
positive_train_samples_upsampled = positive_train_samples_upsampled.drop(columns=["labels"])
positive_train_samples_upsampled.to_csv("./origin_data/positive_train_samples.csv", header=False, index=False)
negative_train_samples = negative_train_samples.drop(columns=["labels"])
negative_train_samples.to_csv("./origin_data/negative_train_samples.csv", header=False, index=False)

# Balance the validation set
positive_test_samples = test_data[test_data["labels"] == 1]
negative_test_samples = test_data[test_data["labels"] == 0]
print(f"test pos:neg = {len(positive_test_samples)}:{len(negative_test_samples)} = {len(positive_test_samples) / len(negative_test_samples)}")
min_samples = min(len(positive_test_samples), len(negative_test_samples))
negative_test_samples = negative_test_samples.sample(n=min_samples, random_state=42)
print(f'downsampled negative samples: {len(negative_test_samples)}')

#
positive_test_samples = positive_test_samples.drop(columns=["labels"])
positive_test_samples.to_csv("./origin_data/positive_test_samples.csv", header=False, index=False)
negative_test_samples = negative_test_samples.drop(columns=["labels"])
negative_test_samples.to_csv("./origin_data/negative_test_samples.csv", header=False, index=False)



