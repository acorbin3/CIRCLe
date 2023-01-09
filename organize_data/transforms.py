import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

"""
Sourced from: https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
This allows to transform the mask and the image at the same time
"""


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, target_is_mask=True):
        image = F.resize(image, self.size)
        if target_is_mask:
            target = F.resize(target, self.size, interpolation=T.InterpolationMode.NEAREST)
        else:
            target = F.resize(target, self.size)

        return image, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target, target_is_mask=True):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        if target_is_mask:
            target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        else:
            target = F.resize(target, size)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob=.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target, target_is_mask=True):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, target, target_is_mask=True):
        random_degrees = random.randint(abs(self.degrees) * -1, abs(self.degrees))
        image = F.rotate(image, angle=random_degrees)
        target = F.rotate(target, angle=random_degrees)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, target_is_mask=True):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, target_is_mask=True):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class PILToTensor:
    def __call__(self, image, target, target_is_mask=True):
        image = F.pil_to_tensor(image)
        if target_is_mask:
            target = torch.as_tensor(np.array(target), dtype=torch.int64)
        else:
            target = F.pil_to_tensor(target)

        return image, target


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target, target_is_mask=True):
        image = F.convert_image_dtype(image, self.dtype)
        if not target_is_mask:
            target = F.convert_image_dtype(target, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target, target_is_mask=True):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if not target_is_mask:
            target = F.normalize(target, self.dtype)
        return image, target