CUDA_VISIBLE_DEVICES=0 python train.py datasets/food-101 --model convnext_tiny_ablation_study -b 32 --model-kwargs remove_deepwise=False remove_shocut=True remove_layernorm=False