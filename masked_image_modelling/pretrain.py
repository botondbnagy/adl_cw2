import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets
import torchvision.transforms as T

import time
import argparse
import matplotlib.pyplot as plt

# File imports
from vit import ViT
from simmim import SimMIM

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

trainloader = torch.utils.data.DataLoader(train_set, batch_size=500, shuffle=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=500, shuffle=False)

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


def display_reconstructions(testloader, mim):
    """Display 8 reconstructed patches and their corresponding ground truth patches."""
    test_images, test_targets = next(iter(testloader))
    # Evaluate model on test image
    test_loss, test_pred, test_masks = mim(test_images)

    # Plot an array of 8 masked patches reconstructed
    fig, axs = plt.subplots(2, 1, figsize=(20, 4))

    pred_patches = test_pred[0].view(-1, 32, 32, 3)
    mask_patches = test_masks[0].view(-1, 32, 32, 3)

    # Unnormalize
    pred_patches = pred_patches * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN
    mask_patches = mask_patches * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN

    # Make grid for plotting
    test_patches = torchvision.utils.make_grid(pred_patches.permute(0, 3, 1, 2), nrow=8)
    test_masks = torchvision.utils.make_grid(mask_patches.permute(0, 3, 1, 2), nrow=8)

    # Plot the reconstructed patches
    axs[0].imshow(test_patches.permute(1, 2, 0).detach().cpu())
    axs[0].set_title('Reconstructed patches')
    axs[0].axis('off')

    # Plot the ground truth masks
    axs[1].imshow(test_masks.permute(1, 2, 0).detach().cpu())
    axs[1].set_title('Ground truth masks')
    axs[1].axis('off')

    plt.show()


# display_reconstructions(testloader, mim)

n_epochs = 100
for i in range(n_epochs):
    j = 0
    running_loss = 0.0
    for images, _ in trainloader:
        print(f'Epoch {i} | Batch {j}')
        j += 1

        loss, pred, masks = mim(images)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

    # display_reconstructions(testloader, mim)
    print(f'Epoch {i} - Loss: {running_loss / len(trainloader)}')

# Save the model
torch.save(mim.state_dict(), 'mim.pth')