CUDA_VISIBLE_DEVICES=1 python train_.py ../datasets --dataset torch/cifar100 --amp --model convnext_tiny_ablation_study -b 16  --experiment stem_cf100 --model-kwargs remove_deepwise=False remove_shortcut=False remove_layernorm=False remove_stem=True