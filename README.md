# iMTT: A Transformer-based Two-tower Model for Predicting i-Motif Folding States

## Overview

iMTT (i-Motif Transformer-based Two-tower model) is a deep learning model designed to predict the folding states of i-Motifs (iMs), which are four-stranded nucleic acid structures formed by cytosine-rich DNA sequences. The model leverages the Transformer architecture and employs a two-tower design with cross-attention mechanisms to process both fine-grained (e.g., character-level) and coarse-grained (e.g., subsequence-level) features simultaneously. iMTT uses semi-supervised learning to balance the dataset and improve prediction accuracy. Experimental results demonstrate that iMTT outperforms existing state-of-the-art models in predicting i-Motif folding states.

## Dataset

The dataset used for training and evaluation is based on CUT&Tag sequencing data from the NCBI GEO database (accession number GSE220882). The data includes biological replicates from two cell lines: human embryonic kidney (HEK293T) and 93T449 (WDLPS). The dataset is divided into positive, negative, and unlabeled samples.

### Data Location
The processed dataset for iMTT semi_v2 model can be found in the following directory:  
data/  
├── samples/  
│ ├── distinct_positive_samples_HEK293T.csv  
│ ├── distinct_negative_samples_HEK293T.csv  
│ ├── distinct_unlabeled_samples_HEK293T.csv  
│ ├── distinct_positive_samples_WDLPS.csv  
│ ├── distinct_negative_samples_WDLPS.csv  
│ └── distinct_unlabeled_samples_WDLPS.csv  
└── pseudo_labeled_samples/  
&nbsp;&nbsp;&nbsp;  ├── pseudo_positive_samples_HEK293T_v2.csv  
&nbsp;&nbsp;&nbsp;  └── pseudo_negative_samples_HEK293T_v2.csv    


## Model Training Stages

iMTT training consists of four main stages:

1. **Masked Language Model (MLM) Pre-training**:
   - **Code Location**: `train_from_base/roberta_pretrainer.py` (base-level tokenizer) and `train_from_bpe/roberta_pretrainer.py` (subsequence-level tokenizer)
   - **Description**: Two separate masked language models are trained using base-level and subsequence-level tokenizers. These models are used to enhance the performance of the subsequent classification models.

2. **Classification Model Fine-tuning**:
   - **Code Location**: `train_from_base/roberta_finetuning_v20241222.py` (base-level tokenizer) and `train_from_bpe/roberta_finetuning_v20241222.py` (subsequence-level tokenizer)
   - **Description**: The pre-trained MLM models are fine-tuned into binary classification models using labeled samples. The models are trained separately for base-level and subsequence-level tokenizers.

3. **Two-tower Model Training**:
   - **Code Location**: `train_two_tower_model/two_tower_main_v20241222.py`
   - **Description**: The two-tower model is initialized with the parameters from the fine-tuned classification models. The model is further fine-tuned using labeled data, and cross-attention mechanisms are introduced to enable information exchange between the two towers.

4. **Semi-supervised Learning**:
   - **Code Location**: `train_two_tower_model/two_tower_main_v20241228.py`
   - **Description**: The two-tower model is trained using a semi-supervised learning approach, incorporating pseudo-labeled samples to balance the dataset and improve model performance.

## Model Architecture

The iMTT model architecture consists of two towers, each processing sequences at different granularities (base-level and subsequence-level). The model incorporates self-attention and cross-attention layers to capture interactions between C-tracts and loop regions. The final classification head performs binary classification to predict whether the input sequence represents a folded or unfolded i-Motif.

### Code Location
The detailed implementation of the iMTT model can be found in:
`train_two_tower_model/two_tower_models.py`

## Usage

To reproduce the iMTT model, follow these steps:

1. **Preprocess the Data**:
   - Run the preprocessing scripts to prepare the dataset for training (already in `/data` directory):
   - **Code Location**:
   `data/*.py` and `unlabeled_labeler.py`

2. **Train the Masked Language Models**:
    - Train the base-level and subsequence-level MLM models.

3. **Fine-tune the Classification Models**:
    - Fine-tune the MLM models into binary classification models.

4. **Train the Two-tower Model**:
    - Train the two-tower model using the fine-tuned classification models.

5. **Perform Semi-supervised Learning**:
    - Train the model using semi-supervised learning with pseudo-labeled samples.

## Results

The model's performance is evaluated using metrics such as accuracy, AUROC, AUPRC, and F1-score. The results demonstrate that iMTT outperforms existing state-of-the-art models in predicting i-Motif folding states.

### Code Location
The implementation of the reproduced/retrained iM-Seeker classification model can be found in:
`references/iM-Seeker`

### Influence of putative C-tract number on prediction of folded i-Motif structures
The implementation of the experiment can be found in:
`test/test_ctract_num_influence.ipynb`

### Features importance for iM folding status
The implementation of the experiment can be found in:
`test/test_33feats.ipynb`

### Attention Visualization
The attention mechanisms in the iMTT model can be visualized using BertViz. The visualization scripts are located in:
`train_two_tower_model/visual_bertviz.ipynb`

