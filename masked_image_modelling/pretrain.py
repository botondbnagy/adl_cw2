import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets
import torchvision.transforms as T

import time
import matplotlib.pyplot as plt

# File imports
from vit import ViT
from simmim import SimMIM
from utils import OxfordIIITPetsAugmented, ToDevice, get_device

IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225])

# Dataset transforms
transform = T.Compose([
    T.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
    T.RandomResizedCrop(128, scale=(0.67, 1), ratio=(3. / 4., 4. / 3.)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
])

# Load the dataset
dataset = datasets.ImageNet(root='./data', split='val', transform=transform)
train_set, test_set = torch.utils.data.random_split(dataset, [45000, 5000])


dataloader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=True)


model = ViT(
    image_size = 128,
    patch_size = 32,
    num_classes = 10,
    dim = 128,
    depth = 12,
    heads = 8,
    mlp_dim = 384
)

# Print number of parameters
print('Number of parameters:', sum(p.numel() for p in model.parameters()))

mim = SimMIM(
    encoder = model,
    masking_ratio = 0.5  # they found 50% to yield the best results
)
optimizer = optim.AdamW(
		params=mim.parameters(),
		lr=8e-4,
		weight_decay=5e-2
)

n_epochs = 100
for i in range(n_epochs):
    j = 0
    running_loss = 0.0
    for images, _ in dataloader:
        print(f'Epoch {i} | Batch {j}')
        j += 1

        loss, pred, masks = mim(images)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

    print(f'Epoch {i} - Loss: {running_loss / len(dataloader)}')