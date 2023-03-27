CUDA_VISIBLE_DEVICES=1 python inference_gradcam.py \
--data-dir datasets/food-101/test/ \
--model resnet50 \
--pretrained \
--results-dir output/inference_cam_test/ \