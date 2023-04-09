CUDA_VISIBLE_DEVICES=1 python inference_gradcam.py \
--data-dir ../datasets/food-101/small_test \
--model convnext_tiny_ablation_study --model-kwargs remove_deepwise=False remove_shortcut=False remove_layernorm=False remove_stem=True \
--checkpoint output/train/convnext_tiny_ablation_study-stem_food-224_300ep/model_best.pth.tar \
--results-dir output/grad_cam/ \
--exp food_abla_stem