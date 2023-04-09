CUDA_VISIBLE_DEVICES=1 python inference_gradcam.py \
--data-dir ../datasets/food-101/small_test \
--checkpoint output/train/resnet50-resnet50_food-224_300ep/model_best.pth.tar \
--results-dir output/grad_cam/ \
--exp food_resnet50