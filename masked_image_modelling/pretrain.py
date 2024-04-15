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
from utils import ToDevice, get_device

# Device
IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225])
device = get_device()

def load_imagenet_data():
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

    # Load dataset into memory for faster training
    train_set, test_set = list(train_set), list(test_set)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)
    return trainloader, testloader

def display_reconstructions(testloader, mim):
    """Display 8 reconstructed patches and their corresponding ground truth patches."""
    test_images, test_targets = next(iter(testloader))
    test_images = test_images.to(device)
    # Evaluate model on test image
    test_loss, test_pred, test_masks = mim(test_images)

    # Plot an array of 8 masked patches reconstructed
    fig, axs = plt.subplots(2, 1, figsize=(20, 4))

    patch_size = 16

    pred_patches = test_pred[0].view(-1, patch_size, patch_size, 3).to(device)
    mask_patches = test_masks[0].view(-1, patch_size, patch_size, 3).to(device)

    # Unnormalize
    pred_patches = pred_patches * IMAGENET_DEFAULT_STD.to(device) + IMAGENET_DEFAULT_MEAN.to(device)
    mask_patches = mask_patches * IMAGENET_DEFAULT_STD.to(device) + IMAGENET_DEFAULT_MEAN.to(device)

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

    # plt.show()
    plt.savefig('reconstructed_patches.png')

def train_model(model, trainloader, testloader=None, output_weight_name='pretrained_encoder.pth'):
    n_epochs = 1000
    
    optimizer = optim.AdamW(
        params=mim.parameters(),
        lr=8e-3,
        weight_decay=5e-2
    )

    for i in range(n_epochs):
        j = 0
        running_loss = 0.0
        epoch_start = time.time()
        print(f'Epoch {i}', end=' ')
        for images, _ in trainloader:
            # print(f'Epoch {i} | Batch {j}')
            j += 1

            images = images.to(device)
            loss, pred, masks = model(images)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

        # display_reconstructions(testloader, mim)
        print(f'Epoch {i} - Loss: {running_loss / len(trainloader)} - Time: {time.time() - epoch_start}')

        if testloader is not None:
            display_reconstructions(testloader, model)
            torch.save(model.encoder.state_dict(), output_weight_name)

    # Save the encoder
    torch.save(model.encoder.state_dict(), output_weight_name)


if __name__ == "__main__":
    device = get_device()
    print(f'Using device: {device}')

    model = ViT(
        image_size = 128,
        patch_size = 16,
        num_classes = 2,
        dim = 768,
        depth = 12,
        heads = 12,
        mlp_dim = 3072,
    ).to(device)

    # Print number of parameters
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))

    mim = SimMIM(
        encoder = model,
        masking_ratio = 0.5  # they found 50% to yield the best results
    ).to(device)

    trainloader, testloader = load_imagenet_data()
    train_model(mim, trainloader, testloader)