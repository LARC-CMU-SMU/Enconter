python predict_insertion_transformer_parallel.py \
  --save_dir "pointer_e" \
  --eval_dataset "./dataset/CoNLL_test" \
  --output_file "pointer_e"

python predict_insertion_transformer_parallel.py \
  --save_dir "pointer_e" \
  --eval_dataset "./dataset/CoNLL_test_esai" \
  --output_file "pointer_e_esai" \
  --inference_mode "esai"

python predict_insertion_transformer_parallel.py \
  --save_dir "greedy_enconter" \
  --eval_dataset "./dataset/CoNLL_test" \
  --output_file "greedy_encontery"

python predict_insertion_transformer_parallel.py \
  --save_dir "greedy_enconter" \
  --eval_dataset "./dataset/CoNLL_test_esai" \
  --output_file "greedy_enconter_esai" \
  --inference_mode "esai"

python predict_insertion_transformer_parallel.py \
  --save_dir "bbt_enconter" \
  --eval_dataset "./dataset/CoNLL_test" \
  --output_file "bbt_enconter"

python predict_insertion_transformer_parallel.py \
  --save_dir "bbt_enconter" \
  --eval_dataset "./dataset/CoNLL_test_esai" \
  --output_file "bbt_enconter_esai" \
  --inference_mode "esai"
