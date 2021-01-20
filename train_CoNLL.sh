
# POINTER-E
python train_enconter.py \
  --batch_size 8 \
  --save_dir pointer_e \
  --epoch 10 \
  --dataset CoNLL_pointer_e \
  --dataset_version CoNLL \
  --warmup \
  --save_epoch 5

# Greedy Enconter
python train_enconter.py \
  --batch_size 8 \
  --save_dir greedy_enconter \
  --epoch 10 \
  --dataset CoNLL_greedy_enconter \
  --dataset_version CoNLL \
  --warmup \
  --save_epoch 5

# BBT Enconter
python train_enconter.py \
  --batch_size 8 \
  --save_dir bbt_enconter \
  --epoch 10 \
  --dataset CoNLL_bbt_enconter \
  --dataset_version CoNLL \
  --warmup \
  --save_epoch 5