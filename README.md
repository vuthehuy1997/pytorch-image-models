# PyTorch Image Models
- [Introduction](#Introduction)
- [Setup](#setup)
- [Licenses](#licenses)
- [Citing](#citing)

## Introduction
Dự án này chúng tôi kết hợp một số mã nguồn mở trên mạng:
* Pytorch-image-model (timm) (https://github.com/huggingface/pytorch-image-models)
* Grad-CAM (https://github.com/jacobgil/pytorch-grad-cam)

Đồng thời, chúng tôi tiến hành chỉnh sửa một số ít để phù hợp với yêu cầu của bài toán mà chúng tôi áp dụng:
* Định nghĩa thêm Ablation block để có thể huấn luyện ablation.
* Tích hợp grad-CAM để có thể trực quan hóa một số thông tin mà các model đã học dựa vào kết quả.

## Setup
1. Cài đặt môi trường
    * `pip install -r requirement.txt`
2. Dữ liệu sẽ được lưu tại thư mục cùng cấp với git, ở đây cụ thể là `../datasets/<dataset_name>`
    * Ở đây chúng tôi sử dụng 3 bộ dữ liệu: Flower102, food101 và cifar-100. 
    * Các bộ dữ liệu này có thể tìm thấy ở paperwithcode (https://paperswithcode.com/datasets?task=image-classification&page=1) hoặc đã được tích hợp sẵn trên pytorch.
3. Sau khi đã có 3 bộ dữ liệu, tiến hành train với các lệnh `bash` đã được cấu hình trước tại: `commands/train/<dataset>`
4. Sau khi các mô hình đã huấn luyện xong, có thể chạy grad-CAM với tập ảnh cho trước để xem xét khả năng học của mô hình
    * Có thể sử dụng các `bash` đã được thiết lập sẵn trong: `commands/infer_grad/<dataset>`.
    * Cần điều chỉnh tham số `--data-dir` trỏ đến folder chứa các ảnh cần infer. Ví dụ: `--data-dir ../datasets/food-101/small_test`

## Licenses

### Code
The code here is licensed Apache 2.0. I've taken care to make sure any third party code included or adapted has compatible (permissive) licenses such as MIT, BSD, etc. I've made an effort to avoid any GPL / LGPL conflicts. That said, it is your responsibility to ensure you comply with licenses here and conditions of any dependent licenses. Where applicable, I've linked the sources/references for various components in docstrings. If you think I've missed anything please create an issue.

### Pretrained Weights
So far all of the pretrained weights available here are pretrained on ImageNet with a select few that have some additional pretraining (see extra note below). ImageNet was released for non-commercial research purposes only (https://image-net.org/download). It's not clear what the implications of that are for the use of pretrained weights from that dataset. Any models I have trained with ImageNet are done for research purposes and one should assume that the original dataset license applies to the weights. It's best to seek legal advice if you intend to use the pretrained weights in a commercial product.

#### Pretrained on more than ImageNet
Several weights included or references here were pretrained with proprietary datasets that I do not have access to. These include the Facebook WSL, SSL, SWSL ResNe(Xt) and the Google Noisy Student EfficientNet models. The Facebook models have an explicit non-commercial license (CC-BY-NC 4.0, https://github.com/facebookresearch/semi-supervised-ImageNet1K-models, https://github.com/facebookresearch/WSL-Images). The Google models do not appear to have any restriction beyond the Apache 2.0 license (and ImageNet concerns). In either case, you should contact Facebook or Google with any questions.

## Citing

### BibTeX

```bibtex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```

### Latest DOI

[![DOI](https://zenodo.org/badge/168799526.svg)](https://zenodo.org/badge/latestdoi/168799526)
