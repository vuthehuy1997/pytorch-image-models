CUDA_VISIBLE_DEVICES=2 python train_.py ../datasets --dataset torch/cifar100 --amp --model swin_tiny_patch4_window7_224 -b 32 --experiment swinT_cf100
