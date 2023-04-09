CUDA_VISIBLE_DEVICES=1 python inference_gradcam.py \
--data-dir ../datasets/flower102/grad_test \
--model swin_tiny_patch4_window7_224 \
--checkpoint output/train/swin_tiny_patch4_window7_224-swinT_f102-224_300ep/model_best.pth.tar \
--results-dir output/grad_cam/ \
--exp f102_swinT