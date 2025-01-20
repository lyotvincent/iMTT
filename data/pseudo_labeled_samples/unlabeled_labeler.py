# PSEUDO_POS_TOWER_PATH = './pseudo_positive_samples_HEK293T.tower_only.csv'
# PSEUDO_NEG_TOWER_PATH = './pseudo_negative_samples_HEK293T.tower_only.csv'
# PSEUDO_POS_BASE_PATH = './pseudo_positive_samples_HEK293T.base_only.csv'
# PSEUDO_NEG_BASE_PATH = './pseudo_negative_samples_HEK293T.base_only.csv'
# PSEUDO_POS_BPE_PATH = './pseudo_positive_samples_HEK293T.bpe_only.csv'
# PSEUDO_NEG_BPE_PATH = './pseudo_negative_samples_HEK293T.bpe_only.csv'

PSEUDO_PREDS_SAMPLES_BPE_PATH = './pseudo_preds_samples_HEK293T.bpe_only.csv'
PSEUDO_PREDS_SAMPLES_BASE_PATH = './pseudo_preds_samples_HEK293T.base_only.csv'
PSEUDO_PREDS_SAMPLES_TOWER_PATH = './pseudo_preds_samples_HEK293T.tower_only.csv'

from collections import defaultdict


THRESHOLD = 0.5
# POS_THRES = THRESHOLD
# NEG_THRES = -THRESHOLD
'''
用 下面的阈值，生成的正反例数量都是2923
POS_THRES = 0.7958
NEG_THRES = -0.2
'''
# POS_THRES = 0.7958
# NEG_THRES = -0.2
'''
标注数据的 pos:neg = 10966:22974   22974-10966=12008
'''
POS_THRES = 0.2
NEG_THRES = -0.5

all_samples = defaultdict(list)
pseudo_pos = list()
pseudo_neg = list()

# 统计三个模型对每个样本的给POS的次数，>1的认为是POS
with open(PSEUDO_PREDS_SAMPLES_BPE_PATH, 'r') as f:
    while l := f.readline().strip().split():
        all_samples[l[0]].append(float(l[1]))
with open(PSEUDO_PREDS_SAMPLES_BASE_PATH, 'r') as f:
    while l := f.readline().strip().split():
        all_samples[l[0]].append(float(l[1]))
with open(PSEUDO_PREDS_SAMPLES_TOWER_PATH, 'r') as f:
    while l := f.readline().strip().split():
        all_samples[l[0]].append(float(l[1]))

# 保存结果
pos_file = open('./pseudo_positive_samples_HEK293T.csv', 'w')
neg_file = open('./pseudo_negative_samples_HEK293T.csv', 'w')

discard = list()
pos_num = 0
neg_num = 0
for k, v in all_samples.items():
    # if v[0] >= POS_THRES and v[1] >= POS_THRES and v[2] >= POS_THRES: # 对应semi伪标签数据集构建方法1
    if v[2] >= POS_THRES: # 对应semi伪标签数据集构建方法2
        pos_file.write(k + '\n')
        pos_num += 1
    elif v[0] <= NEG_THRES and v[1] <= NEG_THRES and v[2] <= NEG_THRES:
    # elif v[2] <= NEG_THRES:
        neg_file.write(k + '\n')
        neg_num += 1
    else:
        discard.append(k)

pos_file.close()
neg_file.close()

print(f'pos num: {pos_num}; neg num: {neg_num}; discard {len(discard)} samples')



