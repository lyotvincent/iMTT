two_tower_main_v20241222.py是训练的监督学习加载fine-tuning模型参数的模型
> DualTowerForSequenceClassification_base(config, is_pretrained=True, pretrained_version=4)

two_tower_main_v20241228.py 是半监督学习，加载1222双塔模型参数的模型
> DualTowerForSequenceClassification_semi(config, is_pretrained=True, pretrained_version=3)

two_tower_main.py 的模型改变了输出预测的方式，为了 attention 可视化，这个文件只是试试这个模型是否可训练，没有实际训练
