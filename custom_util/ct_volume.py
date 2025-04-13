import torch
import cv2
import monai
import torchvision
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

import custom_util.misc as misc
from typing import Callable, Union

from PIL import Image
from torch.utils.data import Dataset
from monai import transforms as monai_transforms


def histogram_equalize_helper(
    vol: Union[np.ndarray, torch.Tensor], mean_comp: Callable, where_comp: Callable, keep_frames: Callable=lambda x: x > 0.025
) -> np.ndarray:
    assert len(vol.shape) == 3, f"Expected 3D volume, but got input with shape {vol.shape}"
    
    # first get the foreground
    transform = monai.transforms.CropForeground(keys=["pixel_values"], source_key="pixel_values")
    crop_vol = transform(vol)

    keep_frames = where_comp(keep_frames(mean_comp(mean_comp(crop_vol))))[0]
    crop_vol_ = crop_vol[keep_frames]
    W, H, F = crop_vol_.shape

    if (W * H * F) > 0:
        crop_vol = crop_vol_
    else:
        W, H, F = crop_vol.shape
    # next, equalize histogram and convert back to tensor
    proc_vol = cv2.equalizeHist(crop_vol.reshape(W, -1).astype(np.uint8)).reshape(W, H, F)
    return proc_vol


def histogram_equalize_numpy(vol: np.ndarray, keep_frames: Callable=lambda x: x > 0.025):
    """
    Perform histogram equalization for numpy volume input normalization.

    Parameters
    ----------
    vol: torch.Tensor
        3D input volume to normalize
    threshold: Callable = lambda x: x > 0.025
        Way to determine (and exclude) empty background frames based on the mean pixel value in a given frame;
        set to 0.025 threshold that matches UW pre-training data.
    """
    if np.max(vol) == 1:
        vol *= 255
    proc_vol = histogram_equalize_helper(
        vol,
        mean_comp=lambda x: np.mean(x, axis=-1),
        where_comp=lambda x: np.where(x),
        keep_frames=keep_frames
    )
    return proc_vol


def histogram_equalize_torch(vol: torch.Tensor, keep_frames: Callable=lambda x: x > 0.025):
    """
    Perform histogram equalization for tensor volume input normalization.

    Parameters
    ----------
    vol: torch.Tensor
        3D input volume to normalize
    threshold: Callable = lambda x: x > 0.025
        Way to determine (and exclude) empty background frames based on the mean pixel value in a given frame;
        set to 0.025 threshold that matches UW pre-training data.
    """
    if torch.max(vol) == 1:
        vol *= 255
    proc_vol = histogram_equalize_helper(
        vol,
        mean_comp=lambda x: torch.mean(x, dim=-1),
        where_comp=lambda x: torch.where(x),
        keep_frames=keep_frames,
    )
    proc_vol = torchvision.transforms.ToTensor()(proc_vol).permute(1, 2, 0)

    return proc_vol


def resize_image_keep_aspect_ratio(image: torch.Tensor, target_size: tuple):
    """
    Perform resizing of the image to the target size while keeping the aspect ratio. 
    The largest dimension of the image will be resized to the target size, while the other dimension will be padded.

    Parameters
    ----------
    image: torch.Tensor 
        3D input volume to normalize
    target_size: tuple
        Target size to resize the image to. Expect the target size is cubic. Example: [T, W, H]
    """
    assert target_size[0] == target_size[1] == target_size[2], f"Expected cubic target size, but got {target_size}"
    # get the current size of the image
    current_size = image.shape[-3:]
    # get the ratio of the current size to the target size
    ratio = target_size[0] / np.max(current_size)
    # get the new size of the image
    new_size = [int(ratio * c) for c in current_size]
    # resize the image
    image = monai.transforms.Resize(new_size)(image)
    # pad the image
    image = monai.transforms.ResizeWithPadOrCrop(target_size)(image)

    return image


class CTVolumeDataset(Dataset): # added by Hanwen to load csv data
    def __init__(self, 
                 input_filename, 
                 transform, 
                 img_key='images', 
                 input_size=128,
                 num_frames=16, 
                 histogram_equalize=False,
                 **kwargs):
        print(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename)

        self.images = df[img_key].tolist()
        self.transform = transform
        self.input_size = input_size
        self.num_frames = num_frames
        self.histogram_equalize = histogram_equalize

        print('Done loading {} images in total.'.format(len(self.images)))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        for _ in range(10):
            try:
                images = torch.load(self.images[idx])
                # histogram equalization
                if self.histogram_equalize:
                    images = histogram_equalize_torch(images.squeeze(0)).unsqueeze(0)
                # resize the images while keeping the aspect ratio
                images = resize_image_keep_aspect_ratio(images, (self.num_frames, self.input_size, self.input_size))

                if self.transform is not None:    
                    images = self.transform({"pixel_values": images})["pixel_values"]
                    
                return images, self.images[idx].split('/')[-1]

            except:
               print('Failed to load image {}, retrying...'.format(self.images[idx]))
               idx = np.random.randint(0, len(self.images))
               continue


class CTInpaintDataset(Dataset): # added by Hanwen to load csv data
    def __init__(self, 
                 input_filename, 
                 transform, 
                 img_key='images', 
                 mask_key='masks',
                 input_size=128,
                 num_frames=16, 
                 histogram_equalize=False,
                 **kwargs):
        print(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename)

        self.images = df[img_key].tolist()
        self.masks = df[mask_key].tolist()
        self.transform = transform
        self.input_size = input_size
        self.num_frames = num_frames

        print('Done loading {} images in total.'.format(len(self.images)))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        for _ in range(10):
            try:
                images = torch.from_numpy(np.load(self.images[idx]))
                masks = torch.from_numpy(np.load(self.masks[idx])).permute(1, 0, 2, 3)

                if images.max() > 1:
                    images = images / 255.0
                
                input_frames = images.shape[0]
                if input_frames < images.shape[1]:
                    # pad on the right
                    pad_r = torch.zeros(images.shape[2] - input_frames, 256, 256)
                    images = torch.cat([images, pad_r], dim=0)
                    pad_r = torch.zeros(2, masks.shape[2] - input_frames, 256, 256)
                    masks = torch.cat([masks, pad_r.clone()], dim=1)
                # print(images.shape, masks.shape)
                # make dummy masks
                # masks = torch.zeros_like(images)
                # masks[:, :, 64: 192, 64: 192] = 1
                images = images.unsqueeze(0)
                # import pdb; pdb.set_trace()
                if self.transform is not None:    
                    inputs = self.transform({"pixel_values": images, "masks": masks})
                    images = inputs["pixel_values"]
                    masks = inputs["masks"]
                    
                return images, masks, self.images[idx].split('/')[-1]

            except:
               print('Failed to load image {}, retrying...'.format(self.images[idx]))
               idx = np.random.randint(0, len(self.images))
               continue

    
def create_vol_transforms(RandZoomd_prob=0.0, RandRotate90d_prob=0.0, RandAffined_prob=0.0, **kwargs):
    
    # create the transform function
    train_transform = monai_transforms.Compose(
        [
            monai_transforms.RandZoomd(keys=["pixel_values"], prob=RandZoomd_prob, min_zoom=0.8, max_zoom=1.2),
            monai_transforms.RandRotate90d(keys=["pixel_values"], prob=RandRotate90d_prob, max_k=3, spatial_axes=[0, 1]),
            monai_transforms.RandRotate90d(keys=["pixel_values"], prob=RandRotate90d_prob, max_k=3, spatial_axes=[1, 2]),
            monai_transforms.RandAffined(keys=["pixel_values"], prob=RandAffined_prob, rotate_range=np.pi/3, translate_range=0.2),
            monai_transforms.NormalizeIntensityd(keys=["pixel_values"], subtrahend=misc.IMG_MEAN, divisor=misc.IMG_STD, nonzero=False)
        ]
    )
    
    val_transform = monai_transforms.Compose(
        [
            monai_transforms.RandZoomd(keys=["pixel_values"], prob=RandZoomd_prob, min_zoom=0.8, max_zoom=1.2),
            monai_transforms.RandRotate90d(keys=["pixel_values"], prob=RandRotate90d_prob, max_k=3, spatial_axes=[0, 1]),
            monai_transforms.RandRotate90d(keys=["pixel_values"], prob=RandRotate90d_prob, max_k=3, spatial_axes=[1, 2]),
            monai_transforms.RandAffined(keys=["pixel_values"], prob=RandAffined_prob, rotate_range=np.pi/3, translate_range=0.2),
            monai_transforms.NormalizeIntensityd(keys=["pixel_values"], subtrahend=misc.IMG_MEAN, divisor=misc.IMG_STD, nonzero=False)
        ]
    )

    return train_transform, val_transform


def create_inpaint_transforms(input_size=256, num_frames=256, RandAffined_prob=0.1, **kwargs):
    
    # create the transform function
    train_transform = monai_transforms.Compose(
        [
            monai_transforms.Resized(
                keys=["pixel_values", "masks"], spatial_size=(num_frames, input_size, input_size), mode=("bilinear")
            ),
            monai_transforms.RandAffined(keys=["pixel_values", "masks"], prob=RandAffined_prob, rotate_range=np.pi/18, translate_range=0.1, scale_range=0.1),
            monai_transforms.NormalizeIntensityd(keys=["pixel_values"], subtrahend=misc.IMG_MEAN, divisor=misc.IMG_STD, nonzero=True),
        ]
    )
    
    val_transform = monai_transforms.Compose(
        [
            monai_transforms.Resized(
                keys=["pixel_values", "masks"], spatial_size=(num_frames, input_size, input_size), mode=("bilinear")
            ),
            monai_transforms.NormalizeIntensityd(keys=["pixel_values"], subtrahend=misc.IMG_MEAN, divisor=misc.IMG_STD, nonzero=True),
        ]
    )

    return train_transform, val_transform