# Enconter
This is the official implementation of 2021 EACL paper
> ENCONTER: Entity Constrained Progressive Sequence Generation viaInsertion-based Transformer
## File description
```
train_CoNLL.sh: training args
train_enconter.py: main training script
preprocess_CoNLL.ipynb: preprocessing CoNLL dataset for training
dataset_utils.py, utils.py: some utilities
```
## How to use
1. Clone the repo
2. Run `preprocess_CoNLL.ipynb` to preprocess training and testing data
3. Run `train_CoNLL.sh` to train POINTER-E, Greedy Enconter, BBT Enconter
4. Run `predict_CoNLL.sh` to conduct generation
## Dependicies
```
tqdm==4.49.0
transformers==3.0.0
torch==1.5.0
numpy==1.19.2
matplotlib==3.2.1
yake==0.4.3
```
## Reference
