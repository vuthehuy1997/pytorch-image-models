CUDA_VISIBLE_DEVICES=1 python train_.py --data-dir ../datasets/flower102 --amp \
--model convnext_tiny_ablation_study -b 32 \
--experiment layernorm_f102  \
--model-kwargs remove_deepwise=False remove_shortcut=False remove_layernorm=True
