# Description: Fine-tune a pre-trained ViT model on the Oxford-IIIT Pet dataset for segmentation.
import time
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils
import torchvision.transforms as T
import torchvision

# File imports
from models.vit import ViT
from models.finetune import FineTune
from utils.utils import AugmentedOxfordIIITPet, get_device, finetune_transforms
from utils.configs import configs

# Command line arguments
TRAIN_SIZE = 6000
TEST_SIZE = 1000
BATCH_SIZE = 64
CONFIG = 'vit_4M_finetune'
TRAIN_SPLIT_SEED = 42
WEIGHTS_PATH = 'placeholder.pth'

# Constants and configurations
config = configs[CONFIG]

# Device
device = get_device()
print(f'Using device: {device}')

# Dataset transforms
transform = finetune_transforms()

# Download Oxford-IIIT Pet Dataset from PyTorch
trainset = AugmentedOxfordIIITPet(
	root='data',
	split="trainval",
	target_types="segmentation",
	download=True,
	**transform,
)
testset = AugmentedOxfordIIITPet(
	root='data',
	split="test",
	target_types="segmentation",
	download=True,
	**transform,
)
generator = torch.Generator().manual_seed(TRAIN_SPLIT_SEED)
full_dataset = torch.utils.data.ConcatDataset([trainset, testset])

# Resplit full dataset into train and test sets of desired sizes
splits = [TRAIN_SIZE, TEST_SIZE, len(full_dataset) - TRAIN_SIZE - TEST_SIZE]
trainset, testset, _ = torch.utils.data.random_split(full_dataset, splits, generator=generator)
trainset, testset = list(trainset), list(testset) # load the data into RAM

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

encoder = ViT(
	image_size = config['image_size'],
	patch_size = config['patch_size'],
	num_classes = config['num_classes'],
	dim = config['dim'],
	depth = config['depth'],
	heads = config['heads'],
	mlp_dim = config['mlp_dim'],
).to(device)

# Print number of parameters
n_params_enc = sum(p.numel() for p in encoder.parameters())
print('Number of parameters (encoder):', n_params_enc)

# Model save name from parameters
savename = CONFIG + '_'
savename += 'data_' + str(TRAIN_SIZE)
savename += datetime.now().strftime('%H-%M') # to avoid overwriting
print(f'Saving model as: {savename}')

model = FineTune(
	encoder = encoder,
	weights_path = WEIGHTS_PATH
).to(device)
optimizer = optim.AdamW(
        params=model.parameters(),
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
	model.train()
	for inputs, labels in trainloader:

		inputs, labels = inputs.to(device), labels.to(device)

		optimizer.zero_grad()
		loss, _, _ = model(inputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		j += 1

	# Step the LR scheduler
	scheduler.step()

	print(f'Epoch {i} - Loss: {running_loss / len(trainloader):.5f} - '
        f'Time: {time.time() - epoch_start:.2f}s - '
        f'LR: {scheduler.get_last_lr()[0]:.2e}')

	# Save model every 10 epochs
	if (i + 1) % 10 == 0:
		torch.save(model.state_dict(), f'finetuned_weights_{savename}_{i}.pth')