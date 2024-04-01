from typing import Tuple
from einops import rearrange
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as T
from torch import randn, zeros
from timm import create_model
from torch.nn.functional import l1_loss
from torch.optim import AdamW


class OxfordIIITPetsAugmented(torchvision.datasets.OxfordIIITPet):
    """
    Data augmentation class for the Oxford IIIT Pets dataset.
    
    This class is built on the torchvision.datasets.OxfordIIITPet class. It adds transforms such that
    one can resize and transform the images AND the segmentation masks (we want a constant image size
    to pass into the ViT).
    """
    def __init__(
        self,
        root: str,
        split: str,
        target_types="segmentation",
        download=False,
        pre_transform=None,
        post_transform=None,
        pre_target_transform=None,
        post_target_transform=None,
        common_transform=None,
    ):
        super().__init__(
            root=root,
            split=split,
            target_types=target_types,
            download=download,
            transform=pre_transform,
            target_transform=pre_target_transform,
        )
        self.post_transform = post_transform
        self.post_target_transform = post_target_transform
        self.common_transform = common_transform

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        (input, target) = super().__getitem__(idx)
        
        # Common transforms are performed on both the input and the labels
        # by creating a 4 channel image and running the transform on both.
        # Then the segmentation mask (4th channel) is separated out.
        if self.common_transform is not None:
            both = torch.cat([input, target], dim=0)
            both = self.common_transform(both)
            (input, target) = torch.split(both, 3, dim=0)
        
        if self.post_transform is not None:
            input = self.post_transform(input)
        if self.post_target_transform is not None:
            target = self.post_target_transform(target)

        return (input, target)
    

def get_device():
    """Get the device to be used for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
	

class ToDevice(torch.nn.Module):
    """
    Sends the input object to the device specified in the
    object's constructor by calling .to(device) on the object.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, img):
        """Send the input object to the device."""
        return img.to(self.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device})"