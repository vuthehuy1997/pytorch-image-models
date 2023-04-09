CUDA_VISIBLE_DEVICES=1 python inference_gradcam.py \
--data-dir ../datasets/flower102/grad_test \
--checkpoint output/train/resnet50-resnet50_f102-224_300ep/model_best.pth.tar \
--results-dir output/grad_cam/ \
--exp f102_resnet50