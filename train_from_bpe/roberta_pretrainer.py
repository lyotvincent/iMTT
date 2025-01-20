import torch
import os
from pathlib import Path
from transformers import RobertaConfig
from transformers import RobertaTokenizer
# from token_processor import load_tokenizer_bpe
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from transformers import TrainerCallback
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.parameters import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train():
    print(f"is cuda available? {torch.cuda.is_available()}")

    # * Step 7: Defining the configuration of the model
    # config = RobertaConfig(
    #     vocab_size=1_000,
    #     hidden_size=256,
    #     max_position_embeddings=514,
    #     num_attention_heads=8,
    #     num_hidden_layers=4,
    #     type_vocab_size=1,
    # )
    config = RobertaConfig(
        vocab_size=10_000, # 最早是 1_000
        hidden_size=256, # embedding size
        max_position_embeddings=128,
        num_attention_heads=8,
        num_hidden_layers=4,
        type_vocab_size=1,
        position_embedding_type="relative_key",
    )
    print(config)

    ROOT = Path(__file__).resolve().parent.parent
    # * Step 8: Reloading the tokenizer in transformers
    # tokenizer_path = os.path.join(TOKEN_DIR2, "tokenizer")
    tokenizer = RobertaTokenizer.from_pretrained(TOKEN_DIR_BPE)
    # tokenizer = load_tokenizer_bpe()
    # 打印一些示例
    print(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}")
    print(f"tokenizer.mask_token_id: {tokenizer.mask_token_id}")
    print(tokenizer.tokenize("CCCTAAACCCTAAACCCTAACCCTAACCCTAACCCTAACCC"))
    print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("CCCTAAACCCTAAACCCTAACCCTAACCCTAACCCTAACCC")))

    # * Step 9: Initializing a model from scratch
    model = RobertaForMaskedLM(config=config)
    print(model.num_parameters())
    print(model)
    # exit()

    # * Step 10: Building the dataset
    SAMPLES_PATH = os.path.join(ROOT, "data", "samples")
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=os.path.join(SAMPLES_PATH, "imotif_repetitive.csv"),
        block_size=20,
    )
    print(f"dataset size: {len(dataset.examples)}")
    # dataset = load_dataset(
    #     path='text',
    #     data_files=os.path.join(SAMPLES_PATH, "imotif_repetitive.csv")
    # )

    # 打印数据集示例
    # print(dataset['train'][0:3])
    # print(dataset.shape)
    # exit()

    # * Step 11: Defining a data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # print(data_collator(dataset['train'][0:3]))
    # exit()

    
    class LossHistoryCallback(TrainerCallback):
        def __init__(self):
            self.losses = []
            self.epochs = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None and "loss" in logs:
                self.losses.append(logs["loss"])
                self.epochs.append(state.epoch)
    loss_history_callback = LossHistoryCallback()

    # * Step 12: Initializing the trainer

    training_args = TrainingArguments(
        output_dir=PRETRAINED_MODEL_DIR_BPE_4L8HT2,
        overwrite_output_dir=True,
        per_device_train_batch_size=1024,
        # num_train_epochs=10,
        # logging_steps=0.025,
        # save_steps=0.05,
        num_train_epochs=40,
        logging_steps=0.00625,
        save_steps=0.0125, # a checkpoint will be saved every 'save_steps' steps
        save_total_limit=2, # the maximum number of checkpoints that can be saved before the old ones are deleted.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        callbacks=[loss_history_callback],
    )

    # Step 13: Pretraining the model
    trainer.train()

    # Step 14: Saving the final model
    trainer.save_model(PRETRAINED_MODEL_DIR_BPE_4L8HT2)

    plt.plot(loss_history_callback.epochs, loss_history_callback.losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(PRETRAINED_MODEL_DIR_BPE_4L8HT2, "loss_curve.png"), format="png", dpi=300, bbox_inches="tight")

def encode():

    # 检查是否有可用的GPU
    device = 0 if torch.cuda.is_available() else -1

    tokenizer=RobertaTokenizer.from_pretrained(TOKEN_DIR_BPE)
    model=RobertaForMaskedLM.from_pretrained(PRETRAINED_MODEL_DIR_BPE_4L8HT2)

    fill_mask = pipeline(
        task="fill-mask",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    # ['CCCTAAA', 'CCCTAAA', 'CCCTAA', 'CCCTAA', 'CCCTAA', 'CCCTAACCC']
    # result = fill_mask("CCCTAAA<mask>CCCTAACCCTAACCCTAACCCTAACCC")
    result = fill_mask("CCCTAAA<mask>TAACCCTAACCC<mask>CCCTAACCCTAACCC")
    print(result)


if __name__ == "__main__":
    train()
    encode()



