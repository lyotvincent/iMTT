import os
from bertviz import head_view, model_view
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer
from transformers import RobertaForMaskedLM


from parameters import *


# model = BertModel.from_pretrained(output_dir)
model = RobertaForMaskedLM.from_pretrained(PRETRAINED_MODEL_DIR, attn_implementation="eager")
tokenizer = RobertaTokenizer.from_pretrained(TOKEN_DIR)
# tokenizer = BertTokenizer.from_pretrained(TOKEN_DIR)


sentence_a = "CCCTAAACCCTAAACCCTAACCCTAACCCTAACCCTAACCC"
sentence_b = "CCCTCAGCCGGCCCGCCCGCCC"
print(tokenizer.tokenize(sentence_a))
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence_a)))
print(tokenizer.tokenize(sentence_b))
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence_b)))

# inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
# print(inputs)

# input_ids = inputs['input_ids']
# # token_type_ids = inputs['token_type_ids']
# token_type_ids = [0] * len(input_ids[0])
# print(token_type_ids)
# token_type_ids = [0] * (input_ids[0].tolist().index(tokenizer.sep_token_id) + 1)
# token_type_ids += [1] * (len(input_ids[0]) - len(token_type_ids))
# print(token_type_ids)

# attention = model(input_ids, token_type_ids=token_type_ids)[-1]
# sentence_b_start = token_type_ids[0].tolist().index(1)
# input_id_list = input_ids[0].tolist() # Batch index 0
# tokens = tokenizer.convert_ids_to_tokens(input_id_list)

# head_view(attention, tokens)

# Tokenize the input sentence
inputs = tokenizer.encode_plus(sentence_a, return_tensors='pt')
print(inputs)

# Get the model's attention weights
outputs = model(**inputs, output_attentions=True)
attention = outputs.attentions

# Use BertViz to visualize the attention
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
head_view(attention, tokens)

