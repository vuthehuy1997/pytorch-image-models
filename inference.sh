CUDA_VISIBLE_DEVICES=1 python inference_custom.py \
--data-dir datasets/food-101/test/ \
--model convnext_tiny_ablation_study --model-kwargs remove_deepwise=True remove_shortcut=False remove_layernorm=False \
--checkpoint output/train/20230220-155555-convnext_tiny_ablation_study-224/model_best.pth.tar \
--results-dir output/inference/ \
--class-map datasets/food-101/meta/classes.txt \
--label-type None \
--include-index --exclude-output --fullname \