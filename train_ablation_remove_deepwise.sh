CUDA_VISIBLE_DEVICES=2 python train_.py ../datasets \
--dataset torch/flowers102 --dataset-download --amp \
--model convnext_tiny_ablation_study -b 32 \
--experiment deepwise_f102 \
--model-kwargs remove_deepwise=True remove_shortcut=False remove_layernorm=False
