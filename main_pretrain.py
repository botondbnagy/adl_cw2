# Description: Pre-train SimMIM model on a subset of the ImageNet1k dataset.
import time
from datetime import datetime

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets
import torchvision.transforms as T

# File imports
from models.vit import ViT
from models.simmim import SimMIM
from utils.utils import get_device, save_reconstructions, pretrain_transforms
from utils.configs import configs

# Command line arguments
TRAIN_SIZE = 100000
BATCH_SIZE = 64
CONFIG = 'vit_4M_pretrain'
RUN_PLOTS = False # plot reconstructions every 10 epochs

# Constants and configurations
config = configs[CONFIG]

# Device
device = get_device()
print(f'Using device: {device}')

# Dataset transforms
transform = pretrain_transforms(image_size=config['image_size'])

# Load the dataset and sample a random subset
print('Loading dataset (this may take a while)...')
dataset = datasets.ImageNet(root='./data', split='train', transform=transform)
train_idx = torch.randperm(len(dataset))[:TRAIN_SIZE] # random subset indices
train_set = torch.utils.data.Subset(dataset, train_idx)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

# Load the model
model = ViT(
    image_size = config['image_size'],
    patch_size = config['patch_size'],
    num_classes = config['num_classes'],
    dim = config['dim'],
    depth = config['depth'],
    heads = config['heads'],
    mlp_dim = config['mlp_dim']
).to(device)
    
# Print number of parameters
n_params = sum(p.numel() for p in model.parameters())
print('Number of parameters:', n_params)

# Model save name from parameters
savename = CONFIG + '_'
savename += 'data_' + str(TRAIN_SIZE // 10**3) + 'K_'
savename += datetime.now().strftime('%H-%M') # to avoid overwriting
print(f'Saving model as: {savename}')

mim = SimMIM(
    encoder = model,
    masking_ratio = config['masking_ratio'],
).to(device)
optimizer = optim.AdamW(
        params=mim.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
)
scheduler = MultiStepLR(optimizer, 
                        milestones=config['schedule_milestones'], 
                        gamma=config['schedule_gamma']
)

for i in range(config['epochs']):
    j = 0
    running_loss = 0.0
    epoch_start = time.time()
    print(f'Epoch {i} - Training...', end='\r')
    for images, _ in trainloader:
        j += 1

        images = images.to(device)
        loss, pred, masks = mim(images)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

    # Step the LR scheduler
    scheduler.step()

    print(f'Epoch {i} - Loss: {running_loss / len(trainloader):.5f} - '
        f'Time: {time.time() - epoch_start:.2f}s - '
        f'LR: {scheduler.get_last_lr()[0]:.2e}')

    # Save model (and plot reconstructions)
    if (i + 1) % 10 == 0:
        torch.save(mim.encoder.state_dict(), f'pretrained_encoder_{savename}.pth')
        torch.save(mim.state_dict(), f'pretrained_mim_{savename}.pth')
        if RUN_PLOTS:
            mim.plot_reconstructions(trainloader, mim, savename + f'_epoch_{i + 1}')