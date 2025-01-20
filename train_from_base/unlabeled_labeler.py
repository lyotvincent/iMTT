from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification
from safetensors import safe_open
import torch
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.parameters import *

def load_unlabeled_data():
    UNLABELED_SAMPLE = os.path.join(SAMPLES_PATH, "distinct_unlabeled_samples_HEK293T.csv")
    file_contents = set()
    with open(UNLABELED_SAMPLE, 'r', encoding='utf-8') as file:
        while l := file.readline().strip():
            file_contents.add(l)

    file_contents = list(file_contents)
    return file_contents

def label_unlabeled_data(input_seqs):

    # * base model
    base_model = RobertaForSequenceClassification.from_pretrained(FINETUNING_MODEL_DIR_BASE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)

    tokenizer1 = RobertaTokenizer.from_pretrained(TOKEN_DIR_BASE)

    def tokenize_function(examples):
        tokens1 = tokenizer1(examples, return_tensors='pt', padding="max_length", max_length=MAX_LEN_BASE)
        return {"input_ids": tokens1["input_ids"]}

    inputs = tokenize_function(input_seqs)
    
    inputs["input_ids"] = inputs["input_ids"].to(device)
    # Perform prediction
    BATCH_SIZE = 1024
    i = 0
    outputs_base = torch.Tensor([])
    outputs_base = outputs_base.to(device)
    print("input samples num:", inputs["input_ids"].shape)
    with torch.no_grad():
        while batched_inputs := {"input_ids": inputs["input_ids"][i*BATCH_SIZE:(i+1)*BATCH_SIZE]}:
            if len(batched_inputs["input_ids"]) == 0:
                break
            outputs_base = torch.cat((outputs_base, base_model(**batched_inputs)[0]), dim=0)
            i += 1
    # print(outputs)
    print(f"outputs_base shape: {outputs_base.shape}")

    # predictions = torch.argmax(outputs_base, dim=-1)
    # pos_num = torch.sum(predictions)
    # neg_num = len(predictions) - pos_num
    # print(f"prediction pos:neg = {pos_num}:{neg_num} = {pos_num/neg_num}")
    # # 取结果为1的索引为正例，按input_seqs取索引位置的序列，并存入PSEUDO_LABELS_PATH
    # with open(PSEUDO_SAMPLES_PATH+"/pseudo_positive_samples_HEK293T.base_only.csv", 'w', encoding='utf-8') as file:
    #     for i, p in enumerate(predictions):
    #         if p == 1:
    #             file.write(input_seqs[i] + "\n")
    # # 取结果为0的索引为负例，按input_seqs取索引位置的序列，并存入PSEUDO_LABELS_PATH
    # with open(PSEUDO_SAMPLES_PATH+"/pseudo_negative_samples_HEK293T.base_only.csv", 'w', encoding='utf-8') as file:
    #     for i, p in enumerate(predictions):
    #         if p == 0:
    #             file.write(input_seqs[i] + "\n")
    # # return predictions

    # 取 索引 1 处的概率值为预测概率，
    preds = outputs_base[:, 1]
    with open(PSEUDO_SAMPLES_PATH+"/pseudo_preds_samples_HEK293T.base_only.csv", 'w', encoding='utf-8') as file:
        for i, p in enumerate(preds):
            file.write(f"{input_seqs[i]}\t{p}\n")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    input_seqs = load_unlabeled_data()
    predictions = label_unlabeled_data(input_seqs)

    
