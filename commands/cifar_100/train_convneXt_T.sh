CUDA_VISIBLE_DEVICES=1 python train_.py ../datasets --dataset torch/cifar100 --amp --model convnext_tiny -b 32 --experiment convnext_cf100
