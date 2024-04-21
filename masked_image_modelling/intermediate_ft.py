# Data (you have to dowmload the zip file at https://www.kaggle.com/datasets/puneet6060/intel-image-classification/code)
# Put the zip into the zip path below 
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import zipfile
import os
from torchvision import transforms as T
from typing import Tuple
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms.v2 import MixUp
import time
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils
from vit import ViT


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

# Path to the zip file and extraction path 
zip_path = '/content/drive/MyDrive/archive (1).zip'
extract_path = '/content/sample_data'  #Change to any folder you want inside content

# Unzip the dataset
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Need to normalise images the same way as the pre-trained model
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])

# Dataset transforms tailored to the Intel dataset
transform = T.Compose([
    T.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
    T.Resize((128, 128), interpolation=T.InterpolationMode.NEAREST),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(contrast=0.3),
    #RandAugment(num_ops=1, magnitude=6),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])

# Path to the extracted dataset directories
train_data_path = os.path.join(extract_path, 'seg_train/seg_train')
test_data_path = os.path.join(extract_path, 'seg_test/seg_test')

# Load the training and test datasets
train_set = datasets.ImageFolder(root=train_data_path, transform=transform)
test_set = datasets.ImageFolder(root=test_data_path, transform=transform)

train_set, test_set = list(train_set), list(test_set)

# Create data loaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

# Print dataset sizes
print(f"Training set size: {len(train_set)}")

# Pre trained wieght path
pretrained_weights_path = '/content/drive/MyDrive/pretrained_encoder_vit_4M_data_200K_00-02.pth'

# Set device
device = get_device()
print(f'Using device: {device}, {torch.cuda.get_device_name()}')


# Initialize the ViT with the appropriate parameters
encoder = ViT(
    image_size=128,
    patch_size=16,
    num_classes=2,
    dim=128,
    depth=12,
    heads=8,
    mlp_dim=512,
    dropout=0.1
).to(device)

encoder.load_state_dict(torch.load(pretrained_weights_path, map_location=device))

# Update the classification head for 6 classes, number of classes in intel image classification dataset
encoder.mlp_head = nn.Linear(128, 6).to(device)


# TRAINING LOOP 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(encoder.parameters(), lr=8e-4, weight_decay=0)
mixup = MixUp(alpha=0.1, num_classes=6)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[180, 190], gamma=0.1)

# Training loop
n_epochs = 200
for epoch in range(n_epochs):
    epoch_start = time.time()
    running_loss = 0.0
    encoder.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = mixup(inputs, labels)

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = encoder(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    print(f'Epoch {epoch + 1}/{n_epochs} | Avg Loss: {running_loss / len(train_loader):.5f} | Time: {time.time() - epoch_start:.2f}s')

# Save final model
torch.save(encoder.state_dict(), 'intel_int_finetuned_weights.pth')


# Test accuracy 

def load_model(path):
    encoder = ViT(
        image_size=128,
        patch_size=16,
        num_classes=6,
        dim=128,
        depth=12,
        heads=8,
        mlp_dim=512,
        dropout=0.1
    ).to(device)
    encoder.load_state_dict(torch.load(path, map_location=device))
    return encoder  # Corrected this line to return 'encoder' instead of 'model'

def evaluate(model, dataloader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

encoder_weights_paths = '/content/intel_int_finetuned_weights.pth'
model = load_model(encoder_weights_paths)
accuracy = evaluate(model, test_loader)
print(f'intel_int Test Accuracy: {accuracy:.2f}%')
