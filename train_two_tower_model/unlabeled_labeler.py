from transformers import RobertaTokenizer, RobertaConfig
from safetensors import safe_open
import torch
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.parameters import *
from train_two_tower_model.two_tower_models import DualTowerForSequenceClassification

def load_unlabeled_data():
    UNLABELED_SAMPLE = os.path.join(SAMPLES_PATH, "distinct_unlabeled_samples_HEK293T.csv")
    file_contents = set()
    with open(UNLABELED_SAMPLE, 'r', encoding='utf-8') as file:
        while l := file.readline().strip():
            file_contents.add(l)

    file_contents = list(file_contents)
    return file_contents

def label_unlabeled_data(input_seqs):
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
    # * tower model
    tower_model = DualTowerForSequenceClassification(config)
    # print(tower_model.embedding_1.state_dict())
    tensors = {}
    with safe_open(SAVED_MODEL_DIR_3D8H256D_P4+"/model.safetensors", framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    # print(tensors.keys())
    tower_model.load_state_dict(tensors)
    # print(tower_model.embedding_1.state_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tower_model.to(device)

    tokenizer1 = RobertaTokenizer.from_pretrained(TOKEN_DIR_BASE)
    tokenizer2 = RobertaTokenizer.from_pretrained(TOKEN_DIR_BPE)

    def tokenize_function(examples):
        tokens1 = tokenizer1(examples, return_tensors='pt', padding="max_length", max_length=MAX_LEN_BASE)
        tokens2 = tokenizer2(examples, return_tensors='pt', padding="max_length", max_length=MAX_LEN_BPE)
        return {"input_ids_1": tokens1["input_ids"], "input_ids_2": tokens2["input_ids"]}


    inputs = tokenize_function(input_seqs)

    inputs["input_ids_1"] = inputs["input_ids_1"].to(device)
    inputs["input_ids_2"] = inputs["input_ids_2"].to(device)
    # Perform prediction
    BATCH_SIZE = 1024
    i = 0
    outputs_tower = torch.Tensor([])
    outputs_tower = outputs_tower.to(device)
    print("input samples num:", inputs["input_ids_1"].shape)
    with torch.no_grad():
        while batched_inputs := {"input_ids_1": inputs["input_ids_1"][i*BATCH_SIZE:(i+1)*BATCH_SIZE], "input_ids_2": inputs["input_ids_2"][i*BATCH_SIZE:(i+1)*BATCH_SIZE]}:
            if len(batched_inputs["input_ids_1"]) == 0:
                break
            outputs_tower = torch.cat((outputs_tower, tower_model(**batched_inputs)[-1].to(device)), dim=0)
            i += 1
    # print(outputs)
    print(f"outputs_tower shape: {outputs_tower.shape}")

    # predictions = torch.argmax(outputs_tower, dim=-1)
    # pos_num = torch.sum(predictions)
    # neg_num = len(predictions) - pos_num
    # print(f"prediction1 pos:neg = {pos_num}:{neg_num} = {pos_num/neg_num}")
    # # 取结果为1的索引为正例，按input_seqs取索引位置的序列，并存入PSEUDO_LABELS_PATH
    # with open(PSEUDO_SAMPLES_PATH+"/pseudo_positive_samples_HEK293T.tower_only.csv", 'w', encoding='utf-8') as file:
    #     for i, p in enumerate(predictions):
    #         if p == 1:
    #             file.write(input_seqs[i] + "\n")
    # # 取结果为0的索引为负例，按input_seqs取索引位置的序列，并存入PSEUDO_LABELS_PATH
    # with open(PSEUDO_SAMPLES_PATH+"/pseudo_negative_samples_HEK293T.tower_only.csv", 'w', encoding='utf-8') as file:
    #     for i, p in enumerate(predictions):
    #         if p == 0:
    #             file.write(input_seqs[i] + "\n")
    # return predictions

    # 取 索引 1 处的概率值为预测概率，
    preds = outputs_tower[:, 1]
    with open(PSEUDO_SAMPLES_PATH+"/pseudo_preds_samples_HEK293T.tower_only.csv", 'w', encoding='utf-8') as file:
        for i, p in enumerate(preds):
            file.write(f"{input_seqs[i]}\t{p}\n")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    input_seqs = load_unlabeled_data()
    predictions = label_unlabeled_data(input_seqs)

    
