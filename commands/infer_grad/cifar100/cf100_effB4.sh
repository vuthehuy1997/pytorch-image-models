CUDA_VISIBLE_DEVICES=1 python inference_gradcam.py \
--data-dir ../datasets/food-101/small_test \
--model efficientnet_b4 \
--checkpoint output/train/efficientnet_b4-effB4_cf100-320_300ep/model_best.pth.tar \
--results-dir output/grad_cam/ \
--exp cf100_effB4