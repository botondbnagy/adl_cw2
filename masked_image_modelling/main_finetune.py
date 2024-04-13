# Description: Fine-tune a pre-trained ViT model on the Oxford-IIIT Pet dataset for segmentation.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torchvision.transforms as T
import torchvision

import time

from vit import ViT
from finetune import FineTune
from utils import OxfordIIITPetsAugmented, ToDevice, get_device

import matplotlib.pyplot as plt

if __name__ == '__main__':
	# Path to pre-trained weights
	pretrained_weights_path = 'pretrained_encoder.pth'

	# Set device
	device = get_device()
	print(f'Using device: {device}, {torch.cuda.get_device_name()}')

	# Define some transformations for the Oxford IIIT Pet dataset
	def tensor_trimap(t):
		x = t * 255
		x = x.to(torch.long)
		x = x - 1
		return x

	def args_to_dict(**kwargs):
		return kwargs
	
	# Need to normalise images the same way as the pre-trained model
	IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406])
	IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225])

	# Dataset transforms
	transform = T.Compose([
		T.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
		T.ToTensor(),
		T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
	])

	transform_dict = args_to_dict(
		pre_transform=transform,
		pre_target_transform=T.ToTensor(),
		common_transform=T.Compose([
			ToDevice(get_device()),
			T.Resize((128, 128), interpolation=T.InterpolationMode.NEAREST),
			# Random Horizontal Flip as data augmentation.
			T.RandomHorizontalFlip(p=0.5),
		]),
		post_transform=T.Compose([
			# Color Jitter as data augmentation.
			T.ColorJitter(contrast=0.3),
		]),
		post_target_transform=T.Compose([
			T.Lambda(tensor_trimap),
		]),
	)

	# Download Oxford-IIIT Pet Dataset - train and test sets
	trainset = OxfordIIITPetsAugmented(
		root='data',
		split="trainval",
		target_types="segmentation",
		download=True,
		**transform_dict,
	)
	testset = OxfordIIITPetsAugmented(
		root='data',
		split="test",
		target_types="segmentation",
		download=True,
		**transform_dict,
	)

	trainloader = torch.utils.data.DataLoader(
			trainset,
			batch_size=1024,
			shuffle=True,
	)
			
			
	testloader = torch.utils.data.DataLoader(
		testset,
		batch_size=128,
		shuffle=True,
	)

	# Instantiate encoder (ViT to be fine-tuned)
	encoder = ViT(
		image_size = 128,
		patch_size = 16,
		num_classes = 2,
		dim = 768,
		depth = 12,
		heads = 12,
		mlp_dim = 3072,
	).to(device)

	# Print number of parameters
	print('Number of parameters (encoder):', sum(p.numel() for p in encoder.parameters()))

	# instantiate fine-tuning model, load weights from pre-trained model
	model = FineTune(
		encoder = encoder,
		weights_path = pretrained_weights_path,
	).to(device)

	# Print number of parameters
	print('Number of parameters (fine-tune model):', sum(p.numel() for p in model.parameters()))

	# Define optimizer and loss function for mlp head
	optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
	# criterion = nn.CrossEntropyLoss()

	# Train head only
	n_epochs = 1000
	for epoch in range(n_epochs):
		epoch_start = time.time()
		j = 0 # Batch counter
		running_loss = 0.0 # for the epoch
		model.train() # Set model to training mode
		for i, data in enumerate(trainloader, 0):
			print(f'Batch {i + 1}/{len(trainloader)}  ', end='\r')
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)

			optimizer.zero_grad()
			loss, _, _ = model(inputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			j += 1

		print(f'Epoch {epoch + 1}/{n_epochs} | Loss: {running_loss / j:.5f} | Time: {time.time() - epoch_start:.2f}s')
		
		torch.save(model.state_dict(), f'finetuned_weights.pth')
		model.display_example(testset, show=False, save=True)

	# Save model
	torch.save(model.state_dict(), 'finetuned_weights.pth')

	# Evaluate model
	model.eval() # Set model to evaluation mode

	# Save example prediction
	model.display_example(testset, show=False, save=True)

	#TODO: accuracy or other metrics


