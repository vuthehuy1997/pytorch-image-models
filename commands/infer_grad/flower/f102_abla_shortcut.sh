CUDA_VISIBLE_DEVICES=1 python inference_gradcam.py \
--data-dir ../datasets/flower102/grad_test \
--model convnext_tiny_ablation_study --model-kwargs remove_deepwise=False remove_shortcut=True remove_layernorm=False \
--checkpoint output/train/convnext_tiny_ablation_study-shortcut_f102-224_300ep/model_best.pth.tar \
--results-dir output/grad_cam/ \
--exp f102_abla_shortcut