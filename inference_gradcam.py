#!/usr/bin/env python3
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import json
import logging
import os
import time
from contextlib import suppress
from functools import partial

import numpy as np
import pandas as pd
import torch

from timm.data import create_dataset, create_loader, resolve_data_config, ImageNetInfo, CustomDatasetInfo, infer_imagenet_subset
from timm.data.readers.class_map import load_class_map
from timm.layers import apply_test_time_pool
from timm.models import create_model
from timm.utils import AverageMeter, setup_default_logging, set_jit_fuser, ParseKwargs

from PIL import Image

import torchvision.transforms as tvf
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')


_FMT_EXT = {
    'json': '.json',
    'json-record': '.json',
    'json-split': '.json',
    'parquet': '.parquet',
    'csv': '.csv',
}

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (*deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--model', '-m', metavar='MODEL', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--in-chans', type=int, default=None, metavar='N',
                    help='Image input channels (default: None => 3)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default=None, type=str,
                    metavar='N', help='Input image crop mode (squash, border, center). Model default if None.')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")

parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)

parser.add_argument('--results-dir', type=str, default=None,
                    help='folder for output results')

parser.add_argument('--exp', type=str, default=None,
                    help='folder for output results')

def get_image_transforms(input_size):
    transforms = []
    transforms += [
        tvf.Resize([input_size, input_size]),
        tvf.ToTensor(),
        tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    transforms = tvf.Compose(transforms)
    return transforms

def main():
    setup_default_logging()
    args = parser.parse_args()
    # might as well try to do something useful...
    args.pretrained = args.pretrained or not args.checkpoint

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    # create model
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=in_chans,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint,
        **args.model_kwargs,
    )

    reshape_transform=None
    if 'convnext' in args.model:
        target_layers = [model.stages[-1]]
    elif 'resnet' in args.model:
        target_layers = [model.layer4[-1]]
    elif 'efficientnet' in args.model:
        target_layers = [model.blocks[-1][-1]]
    elif 'swin' in args.model:
        target_layers = [model.layers[-1].blocks[-1].norm1]
        def reshape_transform(tensor, height=7, width=7):
            result = tensor.reshape(tensor.size(0),
                height, width, tensor.size(2))

            # Bring the channels to the first dimension,
            # like in CNNs.
            result = result.transpose(2, 3).transpose(1, 2)
            return result
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    targets = [ClassifierOutputTarget(1)]

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    _logger.info(
        f'Model {args.model} created, param count: {sum([m.numel() for m in model.parameters()])}')
    print(args)
    
    model = model.to(device)
    model.eval()

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))  

    batch_time = AverageMeter()
    end = time.time()

    filenames = []
    for root, subdirs, files in os.walk(args.data_dir, topdown=False, followlinks=True):
        for f in files:
            # base, ext = os.path.splitext(f)
            # filenames.append(os.path.join(root, f).replace(args.data_dir, ''))
            filenames.append(f)
    # print("filenames: ", filenames)
    transforms = get_image_transforms(224)
    for idx, filename in enumerate(filenames):
        img_path = os.path.join(args.data_dir, filename)
        print("filename: ", filename)
        print("path: ", img_path)
        pil_image = Image.open(img_path).convert('RGB')
        image = transforms(pil_image)
        image.unsqueeze_(dim=0)
        image = image.to(device)
        grayscale_cam = cam(input_tensor=image, targets=targets)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        pil_image = pil_image.resize((224,224))
        visualization = show_cam_on_image(np.array(pil_image)/255, grayscale_cam, use_rgb=True)

        pil_out = Image.fromarray(visualization)

        save_dir = f"{args.results_dir}_{args.exp}"
        filename = os.path.join(save_dir, filename)
        os.makedirs(os.path.dirname(filename), exist_ok = True)
        pil_out.save(filename)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.log_freq == 0:
            _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                idx, len(filenames), batch_time=batch_time))


if __name__ == '__main__':
    main()
