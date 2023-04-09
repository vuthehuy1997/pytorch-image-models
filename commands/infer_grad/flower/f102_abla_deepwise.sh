CUDA_VISIBLE_DEVICES=1 python inference_gradcam.py \
--data-dir ../datasets/flower102/grad_test \
--model convnext_tiny_ablation_study --model-kwargs remove_deepwise=True remove_shortcut=False remove_layernorm=False \
--checkpoint output/train/convnext_tiny_ablation_study-deepwise_f102-224_300ep/model_best.pth.tar \
--results-dir output/grad_cam/ \
--exp f102_abla_deepwise