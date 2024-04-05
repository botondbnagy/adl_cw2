import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

import time

from vit import ViT
from simmim import SimMIM
from utils import OxfordIIITPetsAugmented, ToDevice, get_device

import matplotlib.pyplot as plt

# Set device
device = get_device()
print(f'Using device: {device}, {torch.cuda.get_device_name()}')
#use GPU to its full potential
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

# Set random seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True
torch.set_printoptions(precision=10
                          ,threshold=10000
                          ,edgeitems=3
                          ,linewidth=80
                          ,profile=None
                          ,sci_mode=False)



# Define some transformations for the Oxford IIIT Pet dataset
def tensor_trimap(t):
	x = t * 255
	x = x.to(torch.long)
	x = x - 1
	return x

def args_to_dict(**kwargs):
	return kwargs

transform_dict = args_to_dict(
	pre_transform=T.ToTensor(),
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

# Display a sample train image and its corresponding target
(train_image, train_target) = trainset[0]

# plt.figure(figsize=(3,3))
# plt.imshow(train_image.permute(1, 2, 0).to('cpu'))
# plt.savefig('train_image.png')

# plt.figure(figsize=(3,3))
# plt.imshow(train_target.squeeze(0).to('cpu'))
# plt.savefig('train_target.png')

model = ViT(
    image_size = 128,
    patch_size = 32,
    num_classes = 37,
    dim = 384,
    depth = 12,
    heads = 6,
    mlp_dim = 1536
).to(device)

# Print number of parameters
print('Number of parameters:', sum(p.numel() for p in model.parameters()))

mim = SimMIM(
    encoder = model,
    masking_ratio = 0.5  # they found 50% to yield the best results
).to(device)
optimizer = optim.AdamW(
		params=mim.parameters(),
		lr=8e-4,
		weight_decay=5e-2
)

n_epochs = 100
for i in range(n_epochs):
    epoch_start = time.time()
    j = 0
    running_loss = 0.0
    for images, _ in trainloader:
        print(f'Epoch {i} | Batch {j}  ', end='\r')
        j += 1

        images = images.to(device)
        loss, pred, masks = mim(images)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

    print(f'Epoch {i} - Loss: {running_loss / len(trainloader)} - Time: {time.time() - epoch_start}')


test_images, test_targets = next(iter(testloader))

plt.imshow(test_images[20].permute(1, 2, 0).to('cpu'))
plt.savefig('test_image.png')