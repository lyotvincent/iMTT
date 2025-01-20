
import torch
import torch.nn as nn

class FourCTractLoss(nn.Module):
    '''
    模型输入是: [BATCH_SIZE, 64]，批次中每条样本是ATCG组成的64bp长度的碱基序列，
    模型输出是: [BATCH_SIZE，64]，批次中每条输出的预测值是预测该位置的碱基是属于 C-tract 区域还是属于 Spacer 区域的多标签二分类任务，连续的 1 表示 C-tract 区域，连续的 0 表示 Spacer 区域。
    每条序列对应的预测值应该符合如下规律: 
    (1) 由于成型的I-Motif中，一定有4个C-tracts，所以 loss 应该约束预测的 C-tracts 数量为4个，
    (2) 由于C-tract的长度是至少3个碱基，所以 loss 应该约束预测的 C-tracts 长度至少为3个碱基。长度不足3个的连续1应该被视为 Spacer 区域。
    根据上述要求实现该损失函数。
    '''
    def __init__(self):
        super(FourCTractLoss, self).__init__()

    def forward(self, predictions, mask):
        '''
        predictions: [BATCH_SIZE, 64] - Binary predictions for each base (0 for Spacer, 1 for C-tract)
        mask: [BATCH_SIZE, 64] - True labels (ground truth) for each base, 0 for Spacer, 1 for C-tract
        '''
        
        # Ensure predictions are between 0 and 1
        predictions = torch.sigmoid(predictions)*mask

        # Binary classification - round predictions to 0 or 1
        pred_labels = (predictions > 0.5).float()

        # Constraint 1: Count the number of C-tract blocks (sequences of 1s)
        def count_c_tract_blocks(sequence):
            # Detect the start and end of contiguous C-tracts
            changes = torch.diff(sequence)
            starts = torch.nonzero(changes == 1).squeeze()
            ends = torch.nonzero(changes == -1).squeeze()

            # Handle edge case where sequence starts or ends with a C-tract
            if sequence[0] == 1:
                starts = torch.cat([torch.tensor([0], device="cuda"), starts], dim=0)
            if sequence[-1] == 1:
                ends = torch.cat([ends, torch.tensor([len(sequence) - 1], device="cuda")], dim=0)

            c_tract_lengths = ends - starts + 1
            return c_tract_lengths

        # Compute the number of C-tracts and their lengths for each batch
        batch_size = predictions.size(0)
        num_c_tracts = []

        for i in range(batch_size):
            lengths = count_c_tract_blocks(pred_labels[i])
            valid_lengths = lengths[lengths >= 3]
            num_c_tracts.append(len(valid_lengths))

        # Convert list of results into tensors
        num_c_tracts = torch.tensor(num_c_tracts, dtype=torch.float)

        # Constraint 1: Penalize if number of C-tract blocks is not equal to 4
        c_tract_count_loss = torch.mean((num_c_tracts - 4) ** 2)

        return c_tract_count_loss/20
