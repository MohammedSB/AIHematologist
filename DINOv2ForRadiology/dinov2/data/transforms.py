# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

import numpy as np
import torch
from torchvision import transforms
# import torchxrayvision as xrv

class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)

    
class RescaleImage:
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            # Convert to tensor
            image = torch.from_numpy(image)
        elif torch.is_tensor(image):
            pass
        else:
            raise TypeError("Input should be of type numpy.ndarray or torch.Tensor")

        # Rescale the tensor to [0, 1]
        min_val = image.reshape(image.shape[0], -1).min(dim=1)[0].reshape(-1, 1, 1)
        max_val = image.reshape(image.shape[0], -1).max(dim=1)[0].reshape(-1, 1, 1)
        return (image - min_val) / (max_val - min_val)

class MaybeToTensor(transforms.PILToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        if isinstance(pic, np.ndarray):
            pic = torch.from_numpy(pic)
            return pic.permute(2, 0, 1) 
        return super().__call__(pic)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [
        transforms.RandomResizedCrop((crop_size, crop_size), scale=(0.75, 1), interpolation=interpolation),
    ]
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.extend(
        [
            MaybeToTensor(),
            RescaleImage(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize((resize_size, resize_size), interpolation=interpolation),
        transforms.CenterCrop((crop_size, crop_size)),
        MaybeToTensor(),
        RescaleImage(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)

def make_segmentation_train_transforms(
    *,
    resize_size: int = 448,
    vflip_prob: float = 0.25,
    hflip_prob: float = 0.25,
    rot_deg: float = 90,
    interpolation=transforms.InterpolationMode.BICUBIC,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    train_transforms_list = [
        transforms.Resize((resize_size, resize_size), interpolation=interpolation)
        ]
    target_transforms_list = [
        transforms.Resize((resize_size, resize_size), interpolation=transforms.InterpolationMode.NEAREST_EXACT)
        ]
    if vflip_prob > 0:
        train_transforms_list.append(transforms.RandomVerticalFlip(vflip_prob))
        target_transforms_list.append(transforms.RandomVerticalFlip(vflip_prob))
    if hflip_prob > 0:
        train_transforms_list.append(transforms.RandomVerticalFlip(hflip_prob))
        target_transforms_list.append(transforms.RandomVerticalFlip(hflip_prob))
    if rot_deg > 0:
        train_transforms_list.append(transforms.RandomRotation(rot_deg))
        target_transforms_list.append(transforms.RandomRotation(rot_deg))

    train_transforms_list.extend([    
        MaybeToTensor(),
        RescaleImage(),
        make_normalize_transform(mean=mean, std=std),
    ])
    target_transforms_list.append(MaybeToTensor())

    return (transforms.Compose(train_transforms_list),
            transforms.Compose(target_transforms_list))

def make_segmentation_eval_transforms(
    *,
    resize_size: int = 448,
    interpolation=transforms.InterpolationMode.BICUBIC,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    train_transforms_list = [
        transforms.Resize((resize_size, resize_size), interpolation=interpolation),
        MaybeToTensor(),
        RescaleImage(),
        make_normalize_transform(mean=mean, std=std)
    ]
    target_transform_list = [
        transforms.Resize((resize_size, resize_size), interpolation=transforms.InterpolationMode.NEAREST_EXACT),
        MaybeToTensor(),
    ] 
    return (transforms.Compose(train_transforms_list),
            transforms.Compose(target_transform_list))