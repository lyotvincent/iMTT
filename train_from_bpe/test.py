import sys, os
import torch
import torch.nn as nn

a = [
     [1,1,0,0,0,1,1,0,0],
     [1,1,0,1,0,1,1,0,0],
     [1,1,1,1,0,1,1,1,1],
     ]

labels = [
        [0,1,0,0,0,1,0,0,0],
        # [0,1,0,1,0,1,0,0,0],
        # [0,1,0,0,0,1,0,0,0],
        ]

mask = [[0,1,1,1,1,1,0,0,0],
        [0,1,1,1,1,0,0,0,0],
        [0,1,1,1,1,1,0,0,0],
        ]

a = torch.tensor(a, dtype=torch.float)
labels = torch.tensor(labels, dtype=torch.float)
mask = torch.tensor(mask, dtype=torch.float)

# loss_fct = nn.BCEWithLogitsLoss()
# loss_fct = nn.BCELoss()
# loss = loss_fct(a*mask, labels)
# print(loss)
for i in range(a.shape[0]):
    # 获取当前向量的掩码后的非零元素
    non_zero_elements = a[i, mask[i] == 1]
    print(non_zero_elements)
