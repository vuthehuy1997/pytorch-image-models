CUDA_VISIBLE_DEVICES=2 python train_.py ../datasets --dataset torch/cifar100 --amp --model efficientnet_b4 -b 32 --experiment effB4_cf100
