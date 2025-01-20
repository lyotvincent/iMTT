from transformers import Trainer
from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)
from typing import Dict
import torch
import torch.nn as nn

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # custom_loss_dict = {"loss2": outputs[0], "loss_cel": outputs[1], "loss_4c": outputs[2], "loss_adj": outputs[3]}
        # self.state.log_history.append(custom_loss_dict)
        # self.state.log_history[-1]["loss2"] = outputs[0]
        # self.state.log_history[-1]["loss_cel"] = outputs[1]
        # self.state.log_history[-1]["loss_4c"] = outputs[2]
        # self.state.log_history[-1]["loss_adj"] = outputs[3]
        # print(outputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            # print("XXXXXXXXXXXX")
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
                # print("AAAAAAAAAAAA")
            else:
                model_name = unwrapped_model._get_name()
                # print("CCCCCCCCCCCCCC")
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
                # print("BBBBBBBBBBBBBBBB")
            else:
                loss = self.label_smoother(outputs, labels)
                # print("DDDDDDDDDDDDDDD")
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            # print("KKKKKKKKKKKK")

        # exit()
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


@DeprecationWarning
class FourCTractLoss(nn.Module):
    '''
    输入是Tokens，每个Token代表C-tract或者Spacer，模型输出是每个Token是C-tract的概率，
    由于成型的I-Motif中，一定有4个C-tracts，所以这个loss用来约束预测的C-tracts数量，
    但是潜在可以成为I-Motif的序列，有没有可能C-tracts数量越多越好？所以这个loss可能与二分类任务不完全一致？
    '''
    def __init__(self):
        super(FourCTractLoss, self).__init__()
        self.epsilon = 1E-6
        self.delta = 2

    def forward(self, predictions, mask):
        predictions = (predictions >= 0.).float()
        masked_preds = predictions * mask

        num_ones = masked_preds.sum(dim=-1)
        # loss = torch.abs(num_ones - 4)
        # loss = (num_ones - 4) ** 2
        # 使用Huber Loss进行平滑处理
        loss_4c = torch.where(torch.abs(num_ones - 4) <= self.delta,
                           torch.abs(num_ones - 4),
                           (torch.abs(num_ones - 4) - self.delta + 0.5)**2+self.delta)
        loss_4c += self.epsilon


        return torch.mean(loss_4c)


@DeprecationWarning
class NoAdjacentOnesLoss(nn.Module):
    '''
    模型输出1代表C-tract，0代表Spacer，这个loss用来约束C-tracts之间不能相邻
    '''
    def __init__(self):
        super(NoAdjacentOnesLoss, self).__init__()
        self.epsilon = 1E-6
        self.delta = 2

    def forward(self, predictions, mask):
        masked_preds = predictions * mask

        shifted_preds = masked_preds[:, 1:-1]
        shifted_preds_next = masked_preds[:, 2:]
        adjacent_ones = torch.sum((shifted_preds==1) & (shifted_preds_next==1), dim=-1).float()
        loss_adj = torch.where(adjacent_ones <= self.delta,
                           adjacent_ones,
                           (adjacent_ones - self.delta + 0.5)**2+self.delta)
        loss_adj += self.epsilon

        return torch.mean(loss_adj)


class BPECombinedLoss(nn.Module):
    def __init__(self):
        super(BPECombinedLoss, self).__init__()
        self.epsilon = 1E-6
        self.delta = 2

    def forward(self, predictions, mask):
        predictions = (predictions >= 0.).float()
        masked_preds = predictions * mask

        num_ones = masked_preds.sum(dim=-1)
        loss_4c = torch.where(torch.abs(num_ones - 4) <= self.delta,
                           torch.abs(num_ones - 4),
                           (torch.abs(num_ones - 4) - self.delta + 0.5)**2+self.delta)
        loss_4c += self.epsilon
        loss_4c = torch.mean(loss_4c)

        for i in range(predictions.shape[0]):
            # 获取当前向量的掩码后的非零元素
            non_zero_elements = predictions[i, mask[i] == 1]
            indices = non_zero_elements[:-1].gt(0).nonzero().flatten()
            adjacent_diff = torch.abs(non_zero_elements[indices] - non_zero_elements[indices+1])
            # 1/ln(x+1) 自变量x的定义域是 [0, +∞]，因变量y值域也是非负数，并且x越大，y越小，也就是邻居距离越大越好loss越小
            adjacent_diff = 1/(torch.log(adjacent_diff+1+self.epsilon))
        loss_adj = torch.mean(adjacent_diff)

        return loss_4c/10, loss_adj/10


if __name__ == "__main__":
    loss_4c = FourCTractLoss()
    preds = torch.Tensor([[1,1,1,0,0,1,0,1,2,3,3],
                         [1,1,0,0,0,1,2,3,3,3,3],
                         [1,1,1,1,0,0,0,2,3,3,3]])
    mask = torch.Tensor([[0,1,1,1,1,1,1,1,0,0,0],
                         [0,1,1,1,1,1,0,0,0,0,0],
                         [0,1,1,1,1,1,1,0,0,0,0]])
    print(loss_4c(preds, mask))
    loss_adj = NoAdjacentOnesLoss()
    print(loss_adj(preds, mask))
    combined_loss = BPECombinedLoss()
    print(combined_loss(preds, mask))

    print(preds.size(1))
