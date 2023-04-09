CUDA_VISIBLE_DEVICES=1 python inference_gradcam.py \
--data-dir ../datasets/flower102/grad_test \
--model convnext_tiny \
--checkpoint output/train/convnext_tiny-convnext_f102-224_300ep/model_best.pth.tar \
--results-dir output/grad_cam/ \
--exp f102_convneXt_T