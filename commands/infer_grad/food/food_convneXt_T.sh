CUDA_VISIBLE_DEVICES=1 python inference_gradcam.py \
--data-dir ../datasets/food-101/small_test \
--model convnext_tiny \
--checkpoint output/train/convnext_tiny-convnext_food-224_300ep/model_best.pth.tar \
--results-dir output/grad_cam/ \
--exp food_convneXt_T