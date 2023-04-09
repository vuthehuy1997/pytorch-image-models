CUDA_VISIBLE_DEVICES=1 python inference_gradcam.py \
--data-dir ../datasets/flower102/grad_test \
--model efficientnet_b4 \
--checkpoint output/train/efficientnet_b4-effB4_f102-320_300ep/model_best.pth.tar \
--results-dir output/grad_cam/ \
--exp f102_effB4