from transformers.models.t5.modeling_t5 import T5Block, T5LayerSelfAttention, T5LayerCrossAttention, T5LayerFF, T5LayerNorm, T5ClassificationHead
from transformers.models.roberta.modeling_roberta import RobertaLayer, RobertaEmbeddings
from transformers.utils import ModelOutput
from transformers import RobertaForSequenceClassification
from safetensors import safe_open
import torch
import torch.nn as nn
from torch.nn import Dropout
import os, sys
from typing import Optional, Tuple
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.parameters import *


class DualTowerForSequenceClassification_v20241216(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # * Tower 1
        self.embedding_1 = nn.Embedding(num_embeddings=config.vocab_size_1, embedding_dim=config.d_model)
        self.encoder_1 = nn.ModuleList([
            T5LayerSelfAttention(config=config,has_relative_attention_bias=True),
            T5LayerFF(config=config),
        ])
        # need n decoders
        self.decoder_list_1 = nn.ModuleList([
            nn.ModuleList([
                T5LayerSelfAttention(config=config),
                T5LayerCrossAttention(config=config),
                T5LayerFF(config=config),
            ])
            for _ in range(config.num_decoder_layers)
        ])
        self.tower_output_1 = nn.ModuleList([
            T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon),
            Dropout(config.dropout_rate),
        ])
        # * Tower 2
        self.embedding_2 = nn.Embedding(num_embeddings=config.vocab_size_2, embedding_dim=config.d_model)
        self.encoder_2 = nn.ModuleList([
            T5LayerSelfAttention(config=config, has_relative_attention_bias=True),
            T5LayerFF(config=config),
        ])
        self.decoder_list_2 = nn.ModuleList([
            nn.ModuleList([
                T5LayerSelfAttention(config=config),
                T5LayerCrossAttention(config=config),
                T5LayerFF(config=config),
            ])
            for _ in range(config.num_decoder_layers)
        ])
        self.tower_output_2 = nn.ModuleList([
            T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon),
            Dropout(config.dropout_rate),
        ])
        # * Output
        config.d_model = config.d_model * 2
        self.classification_head = T5ClassificationHead(config)

    def forward(
            self,
            input_ids_1,
            input_ids_2,
            attention_mask_1=None,
            attention_mask_2=None,
            labels=None,):
        # * Tower 1 Encoder
        hidden_states_1 = self.embedding_1(input_ids_1)
        self_attention_1, feed_forward_1 = self.encoder_1
        # // self_attention_outputs_1 = self_attention_1(hidden_states_1, attention_mask=attention_mask_1)
        # // hidden_states, present_key_value_state = self_attention_outputs_1[:2]
        # // attention_outputs = self_attention_outputs_1[2:]
        hidden_states_1 = self_attention_1(hidden_states_1, attention_mask=attention_mask_1)[0]
        hidden_states_1 = feed_forward_1(hidden_states_1)

        # * Tower 2 Encoder
        hidden_states_2 = self.embedding_2(input_ids_2)
        self_attention_2, feed_forward_2 = self.encoder_2
        hidden_states_2 = self_attention_2(hidden_states_2, attention_mask=attention_mask_2)[0]
        hidden_states_2 = feed_forward_2(hidden_states_2)

        # * Tower 1 Decoder & Tower 2 Decoder
        for modules_1, modules_2 in zip(self.decoder_list_1, self.decoder_list_2):
            self_attention_1, cross_attention_1, feed_forward_1 = modules_1
            hidden_states_1 = self_attention_1(hidden_states_1, attention_mask=attention_mask_1)[0]
            hidden_states_1 = cross_attention_1(hidden_states_1, hidden_states_2)
            hidden_states_1 = feed_forward_1(hidden_states_1)

            self_attention_2, cross_attention_2, feed_forward_2 = modules_2
            hidden_states_2 = self_attention_2(hidden_states_2, attention_mask=attention_mask_2)[0]
            hidden_states_2 = cross_attention_2(hidden_states_2, hidden_states_1)
            hidden_states_2 = feed_forward_2(hidden_states_2)

        # * Tower 1 Output
        for layer in self.tower_output_1:
            hidden_states_1 = layer(hidden_states_1)
        # * Tower 2 Output
        for layer in self.tower_output_2:
            hidden_states_2 = layer(hidden_states_2)
        # * Output
        hidden_states = torch.cat([hidden_states_1, hidden_states_2], dim=-1)
        logits = self.classification_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return loss, logits


class DualTowerForSequenceClassification_v20241217(nn.Module):
    '''
    SAVED_MODEL_DIR_4D8H512DT
    SAVED_MODEL_DIR_4D8H256DT
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        # ! Encoder
        # * Tower 1
        self.embedding_1 = nn.Embedding(num_embeddings=config.vocab_size_1, embedding_dim=config.d_model)
        self.encoder_1 = T5Block(config=config, has_relative_attention_bias=True)
        # * Tower 2
        self.embedding_2 = nn.Embedding(num_embeddings=config.vocab_size_2, embedding_dim=config.d_model)
        self.encoder_2 = T5Block(config=config, has_relative_attention_bias=True)

        # ! Decoder
        config.is_decoder = True
        # * Tower 1
        self.decoder_list_1 = nn.ModuleList([
            T5Block(config=config, has_relative_attention_bias=False)
            for _ in range(config.num_decoder_layers)
        ])
        self.tower_output_1 = nn.ModuleList([
            T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon),
            Dropout(config.dropout_rate),
        ])
        # * Tower 2
        self.decoder_list_2 = nn.ModuleList([
            T5Block(config=config, has_relative_attention_bias=False)
            for _ in range(config.num_decoder_layers)
        ])
        self.tower_output_2 = nn.ModuleList([
            T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon),
            Dropout(config.dropout_rate),
        ])
        # * Output
        self.outlayer1 = nn.Sequential(
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(86, 1, bias=True),
            nn.Tanh(),
        )
        self.outlayer2 = nn.Sequential(
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(config.d_model, config.num_labels, bias=True),
            # nn.Softmax(dim=-1),
        )

    def forward(
            self,
            input_ids_1,
            input_ids_2,
            attention_mask_1=None,
            attention_mask_2=None,
            labels=None,):
        # * Tower 1 Encoder
        hidden_states_1 = self.embedding_1(input_ids_1)
        hidden_states_1 = self.encoder_1(hidden_states_1)[0]

        # * Tower 2 Encoder
        hidden_states_2 = self.embedding_2(input_ids_2)
        hidden_states_2 = self.encoder_2(hidden_states_2)[0]

        # * Tower 1 Decoder & Tower 2 Decoder
        for decoder_1, decoder_2 in zip(self.decoder_list_1, self.decoder_list_2):
            temp_hidden_states_1 = decoder_1(
                hidden_states=hidden_states_1,
                encoder_hidden_states=hidden_states_2
            )
            temp_hidden_states_2 = decoder_2(
                hidden_states=hidden_states_2,
                encoder_hidden_states=hidden_states_1
            )
            hidden_states_1, hidden_states_2 = temp_hidden_states_1[0], temp_hidden_states_2[0]

        # * Tower 1 Output
        for layer in self.tower_output_1:
            hidden_states_1 = layer(hidden_states_1)
        # * Tower 2 Output
        for layer in self.tower_output_2:
            hidden_states_2 = layer(hidden_states_2)
        # * Output
        hidden_states = torch.cat([hidden_states_1, hidden_states_2], dim=1)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.outlayer1(hidden_states)
        hidden_states = hidden_states.squeeze(2)
        logits = self.outlayer2(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return loss, logits


class DualTowerForSequenceClassification_v20241218_1(nn.Module):
    '''
    相比下个模型，此模型自定义了输出层。
    SAVED_MODEL_DIR_3D8H256DRC1
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        # ! Encoder
        # * Tower 1
        self.embedding_1 = nn.Embedding(num_embeddings=config.vocab_size_1, embedding_dim=config.hidden_size)
        self.encoder_1 = RobertaLayer(config=config)
        # * Tower 2
        self.embedding_2 = nn.Embedding(num_embeddings=config.vocab_size_2, embedding_dim=config.hidden_size)
        self.encoder_2 = RobertaLayer(config=config)

        # ! Decoder
        config.add_cross_attention = True
        config.is_decoder = True
        # * Tower 1
        self.decoder_list_1 = nn.ModuleList([
            RobertaLayer(config=config)
            for _ in range(config.num_decoder_layers)
        ])
        # * Tower 2
        self.decoder_list_2 = nn.ModuleList([
            RobertaLayer(config=config)
            for _ in range(config.num_decoder_layers)
        ])
        # * Output
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.outlayer1 = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(86, 1, bias=True),
            nn.Tanh(),
        )
        self.outlayer2 = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(config.hidden_size, config.num_labels, bias=True),
            # nn.Softmax(dim=-1),
        )

    def forward(
            self,
            input_ids_1,
            input_ids_2,
            attention_mask_1=None,
            attention_mask_2=None,
            labels=None,):
        # * Tower 1 Encoder
        hidden_states_1 = self.embedding_1(input_ids_1)
        hidden_states_1 = self.encoder_1(hidden_states_1)[0]

        # * Tower 2 Encoder
        hidden_states_2 = self.embedding_2(input_ids_2)
        hidden_states_2 = self.encoder_2(hidden_states_2)[0]

        # * Tower 1 Decoder & Tower 2 Decoder
        for decoder_1, decoder_2 in zip(self.decoder_list_1, self.decoder_list_2):
            temp_hidden_states_1 = decoder_1(
                hidden_states=hidden_states_1,
                encoder_hidden_states=hidden_states_2
            )
            temp_hidden_states_2 = decoder_2(
                hidden_states=hidden_states_2,
                encoder_hidden_states=hidden_states_1
            )
            hidden_states_1, hidden_states_2 = temp_hidden_states_1[0], temp_hidden_states_2[0]

        # * Output
        hidden_states = torch.cat([hidden_states_1, hidden_states_2], dim=1)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.outlayer1(hidden_states)
        hidden_states = hidden_states.squeeze(2)
        logits = self.outlayer2(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return loss, logits

class RobertaClassificationHead2(nn.Module):
    """Custom Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features1, features2, **kwargs):
        x = features1[:, 0, :] + features2[:, 0, :]  # take <s> token from two sequences (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class DualTowerForSequenceClassification_v20241218_2(nn.Module):
    '''
    SAVED_MODEL_DIR_3D8H256DRC2
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        # ! Encoder
        # * Tower 1
        self.embedding_1 = nn.Embedding(num_embeddings=config.vocab_size_1, embedding_dim=config.hidden_size)
        self.encoder_1 = RobertaLayer(config=config)
        # * Tower 2
        self.embedding_2 = nn.Embedding(num_embeddings=config.vocab_size_2, embedding_dim=config.hidden_size)
        self.encoder_2 = RobertaLayer(config=config)

        # ! Decoder
        config.add_cross_attention = True
        config.is_decoder = True
        # * Tower 1
        self.decoder_list_1 = nn.ModuleList([
            RobertaLayer(config=config)
            for _ in range(config.num_decoder_layers)
        ])
        # * Tower 2
        self.decoder_list_2 = nn.ModuleList([
            RobertaLayer(config=config)
            for _ in range(config.num_decoder_layers)
        ])
        # * Output
        self.classifier = RobertaClassificationHead2(config)

    def forward(
            self,
            input_ids_1,
            input_ids_2,
            attention_mask_1=None,
            attention_mask_2=None,
            labels=None,):
        # * Tower 1 Encoder
        hidden_states_1 = self.embedding_1(input_ids_1)
        hidden_states_1 = self.encoder_1(hidden_states_1)[0]

        # * Tower 2 Encoder
        hidden_states_2 = self.embedding_2(input_ids_2)
        hidden_states_2 = self.encoder_2(hidden_states_2)[0]

        # * Tower 1 Decoder & Tower 2 Decoder
        for decoder_1, decoder_2 in zip(self.decoder_list_1, self.decoder_list_2):
            temp_hidden_states_1 = decoder_1(
                hidden_states=hidden_states_1,
                encoder_hidden_states=hidden_states_2
            )
            temp_hidden_states_2 = decoder_2(
                hidden_states=hidden_states_2,
                encoder_hidden_states=hidden_states_1
            )
            hidden_states_1, hidden_states_2 = temp_hidden_states_1[0], temp_hidden_states_2[0]

        # * Output
        logits = self.classifier(hidden_states_1, hidden_states_2)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return loss, logits


class DualTowerForSequenceClassification_base(nn.Module):
    '''
    SAVED_MODEL_DIR_3D8H256D_P1
    SAVED_MODEL_DIR_3D8H256D_
    SAVED_MODEL_DIR_3D8H256D_P2
    SAVED_MODEL_DIR_3D8H256D_U
    SAVED_MODEL_DIR_3D8H256D_I
    SAVED_MODEL_DIR_3D8H256D_P4
    '''
    def __init__(self, config, is_pretrained=False, pretrained_version=2, fold_i=None):
        super().__init__()
        self.config = config
        # ! Encoder
        # * Tower 1
        config.vocab_size = config.vocab_size_1
        self.embedding_1 = RobertaEmbeddings(config)
        self.encoder_1 = RobertaLayer(config=config)
        # * Tower 2
        config.vocab_size = config.vocab_size_2
        self.embedding_2 = RobertaEmbeddings(config)
        self.encoder_2 = RobertaLayer(config=config)

        # ! Decoder
        config.add_cross_attention = True
        config.is_decoder = True
        # * Tower 1
        self.decoder_list_1 = nn.ModuleList([
            RobertaLayer(config=config)
            for _ in range(config.num_decoder_layers)
        ])
        # * Tower 2
        self.decoder_list_2 = nn.ModuleList([
            RobertaLayer(config=config)
            for _ in range(config.num_decoder_layers)
        ])
        # * Output
        self.classifier = RobertaClassificationHead2(config)

        if is_pretrained:
            if pretrained_version == 1:
                self.load_pretrained_weights_v1(PRETRAINED_MODEL_DIR_BASE, PRETRAINED_MODEL_DIR_BPE_4L8HT2)
            elif pretrained_version == 2:
                self.load_pretrained_weights_v2(PRETRAINED_MODEL_DIR_BASE, PRETRAINED_MODEL_DIR_BPE_4L8HT2)
            elif pretrained_version == 3:
                self.load_pretrained_weights_v3(FINETUNING_MODEL_DIR_BASE+f"_fold{fold_i}", FINETUNING_MODEL_DIR_BPE_4L8HT2+f"_fold{fold_i}")
            elif pretrained_version == 4:
                self.load_pretrained_weights_v4(FINETUNING_MODEL_DIR_BASE+f"_fold{fold_i}", FINETUNING_MODEL_DIR_BPE_4L8HT2+f"_fold{fold_i}")
            else:
                raise ValueError("Invalid pretrained_version")

    def load_pretrained_weights_v1(self, model_path_1, model_path_2):
        """
        之前预训练了配置如下的RobertaForMaskedLM模型，现在想要将其加载到 Tower 1 的 Encoder 和 Decoder 中。
        存储在 PRETRAINED_MODEL_DIR
        config = RobertaConfig(
            vocab_size=9,
            hidden_size=256,
            max_position_embeddings=128,
            num_attention_heads=8,
            num_hidden_layers=4,
            type_vocab_size=1,
            position_embedding_type="relative_key",
        )
        之前预训练了配置如下的RobertaForMaskedLM模型，现在想要将其加载到 Tower 2 的 Encoder 和 Decoder 中。
        存储在 PRETRAINED_MODEL_DIR_4L8HT2
        config = RobertaConfig(
            vocab_size=1_000,
            hidden_size=256, # embedding size
            max_position_embeddings=128,
            num_attention_heads=8,
            num_hidden_layers=4,
            type_vocab_size=1,
            position_embedding_type="relative_key",
        )
        由于 RobertaForMaskedLM 均由 Encoder 构成，所以有些层不完全兼容，所以进行如下加载即可：
        (1) 加载 PRETRAINED_MODEL_DIR 的 Embedding 到 Tower 1 的 Embedding, 加载 PRETRAINED_MODEL_DIR_4L8HT2 的 Embedding 到 Tower 2 的 Embedding;
        (2) 加载 PRETRAINED_MODEL_DIR 的 Encoder 到 Tower 1 的 Encoder, 加载 PRETRAINED_MODEL_DIR_4L8HT2 的 Encoder 到 Tower 2 的 Encoder;
        (3) 加载 PRETRAINED_MODEL_DIR 的 Encoder 的 self-attention 和 全连接层 到 Tower 1 的 Decoder 的 self-attention 和 全连接层;
        (4) 加载 PRETRAINED_MODEL_DIR_4L8HT2 的 Encoder 的 self-attention 和 全连接层 到 Tower 2 的 Decoder 的 self-attention 和 全连接层;
        """
        pretrained_model_1 = RobertaForSequenceClassification.from_pretrained(model_path_1, num_labels=2)
        self.embedding_1.load_state_dict(pretrained_model_1.roberta.embeddings.state_dict())
        self.encoder_1.load_state_dict(pretrained_model_1.roberta.encoder.layer[0].state_dict())
        for i, decoder in enumerate(self.decoder_list_1):
            decoder.attention.load_state_dict(pretrained_model_1.roberta.encoder.layer[i+1].attention.state_dict())
            decoder.intermediate.load_state_dict(pretrained_model_1.roberta.encoder.layer[i+1].intermediate.state_dict())
            decoder.output.load_state_dict(pretrained_model_1.roberta.encoder.layer[i+1].output.state_dict())
  
        pretrained_model_2 = RobertaForSequenceClassification.from_pretrained(model_path_2, num_labels=2)
        self.embedding_2.load_state_dict(pretrained_model_2.roberta.embeddings.state_dict())
        self.encoder_2.load_state_dict(pretrained_model_2.roberta.encoder.layer[0].state_dict())
        for i, decoder in enumerate(self.decoder_list_2):
            decoder.attention.load_state_dict(pretrained_model_2.roberta.encoder.layer[i+1].attention.state_dict())
            decoder.intermediate.load_state_dict(pretrained_model_2.roberta.encoder.layer[i+1].intermediate.state_dict())
            decoder.output.load_state_dict(pretrained_model_2.roberta.encoder.layer[i+1].output.state_dict())

        print("Pretrained weights loaded successfully.")

    def load_pretrained_weights_v2(self, model_path_1, model_path_2):
        """
        在v1的基础上，由于 cross-attention 无预训练参数，而预训练参数可能提升效果，所以尝试如下加载：
        将 pretrained_model_1 的 第n层 Encoder 的 self-attention 的参数加载到 Tower 2 的 第n层 Decoder 的 cross-attention 中
        将 pretrained_model_2 的 第n层 Encoder 的 self-attention 的参数加载到 Tower 1 的 第n层 Decoder 的 cross-attention 中
        """
        pretrained_model_1 = RobertaForSequenceClassification.from_pretrained(model_path_1, num_labels=2)
        pretrained_model_2 = RobertaForSequenceClassification.from_pretrained(model_path_2, num_labels=2)

        def filter_state_dict(state_dict, exclude_key):
            return {k: v for k, v in state_dict.items() if k != exclude_key}

        self.embedding_1.load_state_dict(pretrained_model_1.roberta.embeddings.state_dict())
        self.encoder_1.load_state_dict(pretrained_model_1.roberta.encoder.layer[0].state_dict())
        for i, decoder in enumerate(self.decoder_list_1):
            decoder.attention.load_state_dict(pretrained_model_1.roberta.encoder.layer[i+1].attention.state_dict())
            filtered_state_dict = filter_state_dict(pretrained_model_2.roberta.encoder.layer[i+1].attention.state_dict(), 'self.distance_embedding.weight')
            decoder.crossattention.load_state_dict(filtered_state_dict)
            decoder.intermediate.load_state_dict(pretrained_model_1.roberta.encoder.layer[i+1].intermediate.state_dict())
            decoder.output.load_state_dict(pretrained_model_1.roberta.encoder.layer[i+1].output.state_dict())
  
        self.embedding_2.load_state_dict(pretrained_model_2.roberta.embeddings.state_dict())
        self.encoder_2.load_state_dict(pretrained_model_2.roberta.encoder.layer[0].state_dict())
        for i, decoder in enumerate(self.decoder_list_2):
            decoder.attention.load_state_dict(pretrained_model_2.roberta.encoder.layer[i+1].attention.state_dict())
            filtered_state_dict = filter_state_dict(pretrained_model_1.roberta.encoder.layer[i+1].attention.state_dict(), 'self.distance_embedding.weight')
            decoder.crossattention.load_state_dict(filtered_state_dict)
            decoder.intermediate.load_state_dict(pretrained_model_2.roberta.encoder.layer[i+1].intermediate.state_dict())
            decoder.output.load_state_dict(pretrained_model_2.roberta.encoder.layer[i+1].output.state_dict())

        print("Pretrained weights loaded successfully.")

    def load_pretrained_weights_v3(self, model_path_1, model_path_2):
        pretrained_model_1 = RobertaForSequenceClassification.from_pretrained(model_path_1, num_labels=2)
        pretrained_model_2 = RobertaForSequenceClassification.from_pretrained(model_path_2, num_labels=2)

        self.embedding_1.load_state_dict(pretrained_model_1.roberta.embeddings.state_dict())
        self.encoder_1.load_state_dict(pretrained_model_1.roberta.encoder.layer[0].state_dict())
        for i, decoder in enumerate(self.decoder_list_1):
            decoder.attention.load_state_dict(pretrained_model_1.roberta.encoder.layer[i+1].attention.state_dict())
            decoder.intermediate.load_state_dict(pretrained_model_1.roberta.encoder.layer[i+1].intermediate.state_dict())
            decoder.output.load_state_dict(pretrained_model_1.roberta.encoder.layer[i+1].output.state_dict())
  
        self.embedding_2.load_state_dict(pretrained_model_2.roberta.embeddings.state_dict())
        self.encoder_2.load_state_dict(pretrained_model_2.roberta.encoder.layer[0].state_dict())
        for i, decoder in enumerate(self.decoder_list_2):
            decoder.attention.load_state_dict(pretrained_model_2.roberta.encoder.layer[i+1].attention.state_dict())
            decoder.intermediate.load_state_dict(pretrained_model_2.roberta.encoder.layer[i+1].intermediate.state_dict())
            decoder.output.load_state_dict(pretrained_model_2.roberta.encoder.layer[i+1].output.state_dict())

        new_state_dict = {}
        for k in pretrained_model_1.classifier.state_dict().keys():
            new_state_dict[k] = (pretrained_model_1.classifier.state_dict()[k] + pretrained_model_2.classifier.state_dict()[k]) / 2
        self.classifier.load_state_dict(new_state_dict)

        print("Pretrained weights loaded successfully. (v3)")

    def load_pretrained_weights_v4(self, model_path_1, model_path_2):
        pretrained_model_1 = RobertaForSequenceClassification.from_pretrained(model_path_1, num_labels=2)
        pretrained_model_2 = RobertaForSequenceClassification.from_pretrained(model_path_2, num_labels=2)

        def filter_state_dict(state_dict, exclude_key):
            return {k: v for k, v in state_dict.items() if k != exclude_key}

        self.embedding_1.load_state_dict(pretrained_model_1.roberta.embeddings.state_dict())
        self.encoder_1.load_state_dict(pretrained_model_1.roberta.encoder.layer[0].state_dict())
        for i, decoder in enumerate(self.decoder_list_1):
            decoder.attention.load_state_dict(pretrained_model_1.roberta.encoder.layer[i+1].attention.state_dict())
            filtered_state_dict = filter_state_dict(pretrained_model_2.roberta.encoder.layer[i+1].attention.state_dict(), 'self.distance_embedding.weight')
            decoder.crossattention.load_state_dict(filtered_state_dict)
            decoder.intermediate.load_state_dict(pretrained_model_1.roberta.encoder.layer[i+1].intermediate.state_dict())
            decoder.output.load_state_dict(pretrained_model_1.roberta.encoder.layer[i+1].output.state_dict())
  
        self.embedding_2.load_state_dict(pretrained_model_2.roberta.embeddings.state_dict())
        self.encoder_2.load_state_dict(pretrained_model_2.roberta.encoder.layer[0].state_dict())
        for i, decoder in enumerate(self.decoder_list_2):
            decoder.attention.load_state_dict(pretrained_model_2.roberta.encoder.layer[i+1].attention.state_dict())
            filtered_state_dict = filter_state_dict(pretrained_model_1.roberta.encoder.layer[i+1].attention.state_dict(), 'self.distance_embedding.weight')
            decoder.crossattention.load_state_dict(filtered_state_dict)
            decoder.intermediate.load_state_dict(pretrained_model_2.roberta.encoder.layer[i+1].intermediate.state_dict())
            decoder.output.load_state_dict(pretrained_model_2.roberta.encoder.layer[i+1].output.state_dict())

        new_state_dict = {}
        for k in pretrained_model_1.classifier.state_dict().keys():
            new_state_dict[k] = (pretrained_model_1.classifier.state_dict()[k] + pretrained_model_2.classifier.state_dict()[k]) / 2
        self.classifier.load_state_dict(new_state_dict)

        print("Pretrained weights loaded successfully. (v4)")

    def load_pretrained_weights_v5(self, model_path):
        tensors = {}
        with safe_open(model_path+"/model.safetensors", framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        # print(tensors.keys())
        self.load_state_dict(tensors)
        print("Pretrained weights loaded successfully. (v5)")

    def forward(
            self,
            input_ids_1,
            input_ids_2,
            attention_mask_1=None,
            attention_mask_2=None,
            labels=None,):
        # * Tower 1 Encoder
        hidden_states_1 = self.embedding_1(input_ids_1)
        hidden_states_1 = self.encoder_1(hidden_states=hidden_states_1, attention_mask=attention_mask_1)[0]

        # * Tower 2 Encoder
        hidden_states_2 = self.embedding_2(input_ids_2)
        hidden_states_2 = self.encoder_2(hidden_states=hidden_states_2, attention_mask=attention_mask_2)[0]

        # * Tower 1 Decoder & Tower 2 Decoder
        for decoder_1, decoder_2 in zip(self.decoder_list_1, self.decoder_list_2):
            temp_hidden_states_1 = decoder_1(
                hidden_states=hidden_states_1,
                attention_mask=attention_mask_1,
                encoder_hidden_states=hidden_states_2,
                encoder_attention_mask=attention_mask_2,
            )
            temp_hidden_states_2 = decoder_2(
                hidden_states=hidden_states_2,
                attention_mask=attention_mask_2,
                encoder_hidden_states=hidden_states_1,
                encoder_attention_mask=attention_mask_1,
            )
            hidden_states_1, hidden_states_2 = temp_hidden_states_1[0], temp_hidden_states_2[0]

        # * Output
        logits = self.classifier(hidden_states_1, hidden_states_2)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return loss, logits


class DualTowerForSequenceClassification_semi(nn.Module):
    '''
    用于在已训练好的P4模型上，进一步半监督学习，对应main_v20241228
    '''
    def __init__(self, config, is_pretrained=True, pretrained_version=5, fold_i=None):
        super().__init__()
        self.config = config
        # ! Encoder
        # * Tower 1
        config.vocab_size = config.vocab_size_1
        self.embedding_1 = RobertaEmbeddings(config)
        self.encoder_1 = RobertaLayer(config=config)
        # * Tower 2
        config.vocab_size = config.vocab_size_2
        self.embedding_2 = RobertaEmbeddings(config)
        self.encoder_2 = RobertaLayer(config=config)

        # ! Decoder
        config.add_cross_attention = True
        config.is_decoder = True
        # * Tower 1
        self.decoder_list_1 = nn.ModuleList([
            RobertaLayer(config=config)
            for _ in range(config.num_decoder_layers)
        ])
        # * Tower 2
        self.decoder_list_2 = nn.ModuleList([
            RobertaLayer(config=config)
            for _ in range(config.num_decoder_layers)
        ])
        # * Output
        self.classifier = RobertaClassificationHead2(config)

        if is_pretrained:
            if pretrained_version == 5:
                self.load_pretrained_weights_v5(SAVED_MODEL_DIR_P3+f"_fold{fold_i}")
            else:
                raise ValueError("Invalid pretrained_version")

    def load_pretrained_weights_v5(self, model_path):
        tensors = {}
        with safe_open(model_path+"/model.safetensors", framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        # print(tensors.keys())
        self.load_state_dict(tensors)
        print("Pretrained weights loaded successfully. (v5)")

    def forward(
            self,
            input_ids_1,
            input_ids_2,
            attention_mask_1=None,
            attention_mask_2=None,
            labels=None,):
        # * Tower 1 Encoder
        hidden_states_1 = self.embedding_1(input_ids_1)
        encoder_output = self.encoder_1(hidden_states=hidden_states_1,
                                         attention_mask=attention_mask_1,
                                         output_attentions=True,)
        hidden_states_1 = encoder_output[0]

        # * Tower 2 Encoder
        hidden_states_2 = self.embedding_2(input_ids_2)
        encoder_output = self.encoder_2(hidden_states=hidden_states_2,
                                         attention_mask=attention_mask_2,
                                         output_attentions=True,)
        hidden_states_2 = encoder_output[0]

        # * Tower 1 Decoder & Tower 2 Decoder
        for decoder_1, decoder_2 in zip(self.decoder_list_1, self.decoder_list_2):
            temp_hidden_states_1 = decoder_1(
                hidden_states=hidden_states_1,
                attention_mask=attention_mask_1,
                encoder_hidden_states=hidden_states_2,
                encoder_attention_mask=attention_mask_2,
                output_attentions=True,
            )
            temp_hidden_states_2 = decoder_2(
                hidden_states=hidden_states_2,
                attention_mask=attention_mask_2,
                encoder_hidden_states=hidden_states_1,
                encoder_attention_mask=attention_mask_1,
                output_attentions=True,
            )
            hidden_states_1, hidden_states_2 = temp_hidden_states_1[0], temp_hidden_states_2[0]

        # * Output
        logits = self.classifier(hidden_states_1, hidden_states_2)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return loss, logits


@dataclass
class CustomSequenceClassifierOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class DualTowerForSequenceClassification_visual(nn.Module):
    '''
    尝试加入output_attentions
    '''
    def __init__(self, config, is_pretrained=True, pretrained_version=5):
        super().__init__()
        self.config = config
        # ! Encoder
        # * Tower 1
        config.vocab_size = config.vocab_size_1
        self.embedding_1 = RobertaEmbeddings(config)
        self.encoder_1 = RobertaLayer(config=config)
        # * Tower 2
        config.vocab_size = config.vocab_size_2
        self.embedding_2 = RobertaEmbeddings(config)
        self.encoder_2 = RobertaLayer(config=config)

        # ! Decoder
        config.add_cross_attention = True
        config.is_decoder = True
        # * Tower 1
        self.decoder_list_1 = nn.ModuleList([
            RobertaLayer(config=config)
            for _ in range(config.num_decoder_layers)
        ])
        # * Tower 2
        self.decoder_list_2 = nn.ModuleList([
            RobertaLayer(config=config)
            for _ in range(config.num_decoder_layers)
        ])
        # * Output
        self.classifier = RobertaClassificationHead2(config)

        if is_pretrained:
            if pretrained_version == 5:
                self.load_pretrained_weights_v5(SAVED_MODEL_DIR_3D8H256D_P4)
            else:
                raise ValueError("Invalid pretrained_version")

    def load_pretrained_weights_v5(self, model_path):
        tensors = {}
        with safe_open(model_path+"/model.safetensors", framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        # print(tensors.keys())
        self.load_state_dict(tensors)
        print("Pretrained weights loaded successfully. (v5)")

    def forward(
            self,
            input_ids_1,
            input_ids_2,
            attention_mask_1=None,
            attention_mask_2=None,
            labels=None,):
        all_self_attentions = ()
        all_cross_attentions = ()
        # * Tower 1 Encoder
        hidden_states_1 = self.embedding_1(input_ids_1)
        encoder_output = self.encoder_1(hidden_states=hidden_states_1,
                                         attention_mask=attention_mask_1,
                                         output_attentions=True,)
        hidden_states_1 = encoder_output[0]
        all_self_attentions = all_self_attentions + (encoder_output[1],)

        # * Tower 2 Encoder
        hidden_states_2 = self.embedding_2(input_ids_2)
        encoder_output = self.encoder_2(hidden_states=hidden_states_2,
                                         attention_mask=attention_mask_2,
                                         output_attentions=True,)
        hidden_states_2 = encoder_output[0]
        all_self_attentions = all_self_attentions + (encoder_output[1],)

        # * Tower 1 Decoder & Tower 2 Decoder
        for decoder_1, decoder_2 in zip(self.decoder_list_1, self.decoder_list_2):
            temp_hidden_states_1 = decoder_1(
                hidden_states=hidden_states_1,
                attention_mask=attention_mask_1,
                encoder_hidden_states=hidden_states_2,
                encoder_attention_mask=attention_mask_2,
                output_attentions=True,
            )
            temp_hidden_states_2 = decoder_2(
                hidden_states=hidden_states_2,
                attention_mask=attention_mask_2,
                encoder_hidden_states=hidden_states_1,
                encoder_attention_mask=attention_mask_1,
                output_attentions=True,
            )
            hidden_states_1, hidden_states_2 = temp_hidden_states_1[0], temp_hidden_states_2[0]
            all_self_attentions = all_self_attentions + (temp_hidden_states_1[1],)
            all_cross_attentions = all_cross_attentions + (temp_hidden_states_1[2],)
            all_self_attentions = all_self_attentions + (temp_hidden_states_2[1],)
            all_cross_attentions = all_cross_attentions + (temp_hidden_states_2[2],)

        # * Output
        logits = self.classifier(hidden_states_1, hidden_states_2)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        # return loss, logits
        return CustomSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


if __name__ == "__main__":
    from transformers import T5Config, RobertaConfig
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
    model = DualTowerForSequenceClassification_base(config, is_pretrained=True, pretrained_version=4)
    
    print(model)
