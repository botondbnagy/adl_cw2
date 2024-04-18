import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR


import torchvision
from torchvision import datasets
import torchvision.transforms as T

import time
import argparse
import matplotlib.pyplot as plt

# File imports
from vit import ViT
from simmim import SimMIM
from utils import ToDevice, get_device

# Device
device = get_device()
print(f'Using device: {device}')

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
print('Loading dataset...')
dataset = datasets.ImageNet(root='./data', split='train', transform=transform)
print('dataset size:', len(dataset))
n_train = 200000

#get random indices for train and test
train_idx = torch.randperm(len(dataset))[:n_train]
train_set = torch.utils.data.Subset(dataset, train_idx)

# Load dataset into memory for faster training
# train_set = list(train_set)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=8)

model = ViT(
    image_size = 128,
    patch_size = 16,
    num_classes = 2,
    dim = 128,
    depth = 12,
    heads = 8,
    mlp_dim = 512,
).to(device)

# Print number of parameters
n_params = sum(p.numel() for p in model.parameters())
print('Number of parameters:', n_params)

# Model name
savename = 'vit_'
if n_params > 10**6:
    savename += str(n_params // 10**6) + 'M_'
else:
    savename+= str(n_params // 10**3) + 'K_'
savename += 'data_' + str(n_train // 10**3) + 'K'


mim = SimMIM(
    encoder = model,
    masking_ratio = 0.5  # they found 50% to yield the best results
).to(device)
optimizer = optim.AdamW(
		params=mim.parameters(),
		lr=1e-4,
		weight_decay=0.05
)

scheduler = MultiStepLR(optimizer, milestones=[50, 85], gamma=0.1)

def display_reconstructions(testloader, mim, epoch=None):
    """Display 8 reconstructed patches and their corresponding ground truth patches."""
    test_images, test_targets = next(iter(testloader))
    test_images = test_images.to(device)
    # Evaluate model on test image
    test_loss, test_pred, test_masks = mim(test_images)

    num_patches = mim.encoder.pos_embedding.shape[-2:][0] - 1
    img_size = test_images.size(-1)
    patch_size = int(img_size / (num_patches ** 0.5))
    nrow = int(num_patches ** 0.5)

    # Select a random image
    plot_idx = torch.randint(0, test_images.size(0), (1,))
    pred_patches = test_pred[plot_idx].view(-1, patch_size, patch_size, 3).to(device)
    mask_patches = test_masks[plot_idx].view(-1, patch_size, patch_size, 3).to(device)

    # Get the original patches by passing through .to_patch() method
    org_patches = mim.to_patch(test_images[plot_idx]).view(-1, patch_size, patch_size, 3).to(device)

    # Unnormalize
    pred_patches = pred_patches * IMAGENET_DEFAULT_STD.to(device) + IMAGENET_DEFAULT_MEAN.to(device)
    mask_patches = mask_patches * IMAGENET_DEFAULT_STD.to(device) + IMAGENET_DEFAULT_MEAN.to(device)
    org_patches = org_patches * IMAGENET_DEFAULT_STD.to(device) + IMAGENET_DEFAULT_MEAN.to(device)

    # Replace patches with reconstructed patches
    masked_idx = []
    for i in range(num_patches):
        original_patch = org_patches[i] # get ith original patch
        original_patch = original_patch.view(-1) # Flatten

        # check against all the patches in mask_patches to see if the patch was masked
        for patch_id, mask_patch in enumerate(mask_patches):
            mask_patch = mask_patch.view(-1)
            # If the patch was masked, save the index
            if torch.allclose(original_patch, mask_patch):
                masked_idx.append((i, patch_id)) # (Original patch index, Masked patch index)
                break

    # Replace the masked patches with the reconstructed patches
    reconstruction = org_patches.clone()
    reconstruction[torch.tensor(masked_idx)[:, 0]] = pred_patches[torch.tensor(masked_idx)[:, 1]]

    # Make grid for plotting
    reconstruction = torchvision.utils.make_grid(reconstruction.permute(0, 3, 1, 2), nrow=nrow, padding=0)

    #plot the original image with masked patches grayed out
    grayed_out = org_patches.clone()
    grayed_out[torch.tensor(masked_idx)[:, 0]] = torch.tensor([0.0, 0.0, 0.0]).to(device)
    #remove border
    grayed_out = torchvision.utils.make_grid(grayed_out.permute(0, 3, 1, 2), nrow=nrow, padding=0)

    # Plot the original image with masked patches grayed out
    plt.imshow(grayed_out.permute(1, 2, 0).detach().cpu())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'figures/reconstruction_{int(plot_idx)}_grayed.pdf', bbox_inches='tight', pad_inches=0)

    # Plot the reconstructed patches
    plt.imshow(reconstruction.permute(1, 2, 0).detach().cpu())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'figures/reconstruction_{int(plot_idx)}.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

n_epochs = 100
for i in range(n_epochs):
    j = 0
    running_loss = 0.0
    epoch_start = time.time()
    print(f'Epoch {i}', end=' ')
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

    print(f'Epoch {i} - Loss: {running_loss / len(trainloader)} - Time: {time.time() - epoch_start} - LR: {scheduler.get_last_lr()}')

    # Optional: Display reconstructed images or log additional information
    if (i + 1) % 2 == 0:
        display_reconstructions(trainloader, mim, i+1)
        torch.save(mim.encoder.state_dict(), f'pretrained_encoder_{savename}.pth')
        torch.save(mim.state_dict(), f'pretrained_mim_{savename}.pth')
