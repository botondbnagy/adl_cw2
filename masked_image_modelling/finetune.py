# Description: FineTune model class for training a Vision Transformer on a segmentation task.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision
import matplotlib.pyplot as plt

from vit import ViT
from simmim import SimMIM


class FineTune(nn.Module):
    """
    FineTune model class.
    -------------------
    Work in progress
    """
    def __init__(
        self,
        *,
        encoder,
        weights_path
    ):
        """
        Args:
            encoder (nn.Module): Vision Transformer to be trained.
            weights_path (str): Path to saved weights from pre-trained model.
        """
        super().__init__()

        # Instantiate encoder (ViT to be fine-tuned)
        self.encoder = encoder

        # Load weights from pre-trained model
        self.encoder.load_state_dict(torch.load(weights_path))

        # Freeze weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Get number of patches and encoder dimension
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])
        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        #infer patch size from above 
        self.patch_size = int((pixel_values_per_patch/3)**0.5)
        print('patch_size:', self.patch_size)

        # Linear head (decoder) to predict segmentation target
        self.mlp = nn.Linear(encoder_dim, self.patch_size ** 2 * 3)

    def forward(self, img, target):
        """
        Run a forward pass of the FineTune model.

        Args:
            img (torch.Tensor): Input image tensor.
            target (torch.Tensor): Target segmentation tensor.

        Returns:
            loss (torch.Tensor): CrossEntropy loss.
            pred_patches (torch.Tensor,
                shape=(batch_size, 3, num_patches, patch_size, patch_size)):
                Predicted pixel values (one hot encoded) organised in patches.
            target_patches (torch.Tensor,
                shape=(batch_size, 3, num_patches, patch_size, patch_size)):
                Target pixel values (one hot encoded) organised in patches.
        """
        # Get device ('cuda' or 'cpu')
        device = img.device

        # Get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # For indexing
        batch_range = torch.arange(batch, device = device)[:, None]

        # Get positions
        pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # Get embeddings
        tokens = self.patch_to_emb(patches)
        tokens = tokens + pos_emb

        # encoder output with weights frozen
        encoder_output = self.encoder.transformer(tokens)
        
        # pass each patch through the mlp
        mlp_output = self.mlp(encoder_output)
        pred_patches = mlp_output.reshape(batch, 3, num_patches, -1)
        mlp_output = mlp_output.reshape(batch, 3, -1)
        # mlp_output = nn.functional.softmax(mlp_output, dim=2)

        # target to patches and one hot encode
        target_patches = self.to_patch(target)
        target_patches = nn.functional.one_hot(target_patches, num_classes=3).float()
        target_patches = target_patches.permute(0, 3, 1, 2)
        target_flat = target_patches.reshape(batch, 3, -1)

        #calculate loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(mlp_output, target_flat)

        return loss, pred_patches, target_patches
    
    def display_example(self, testset, show=False, save=True):
        '''Display/save an example prediction from the segmentation model'''

        #pick image in test set
        idx = torch.randint(0, len(testset), (1,)).item()
        img, target = testset[idx]
        img = img.unsqueeze(0)
        target = target.unsqueeze(0)

        img_size = img.shape[-1]
        patch_size = self.patch_size

        with torch.no_grad():
            _, pred_patches, target_patches = model(img, target)

        pred_patches = pred_patches[0]
        target_patches = target_patches[0]

        #make empty tensor to store full image
        pred_full = torch.zeros(1, img_size, img_size)
        target_full = torch.zeros(1, img_size, img_size)

        patch_i = 0
        for row in range(4):
            for col in range(4):
                target_plot = target_patches[:,patch_i].cpu().numpy()
                target_plot = target_plot.reshape(3, patch_size, patch_size).transpose(1, 2, 0)
                pred_plot = pred_patches[:,patch_i].cpu().numpy()
                pred_plot = pred_plot.reshape(3, patch_size, patch_size).transpose(1, 2, 0)
                #take argmax to plot
                target_plot = target_plot.argmax(axis=2)
                pred_plot = pred_plot.argmax(axis=2)

                #add to full image
                target_full[0, row*32:(row+1)*32, col*32:(col+1)*32] = torch.tensor(target_plot)
                pred_full[0, row*32:(row+1)*32, col*32:(col+1)*32] = torch.tensor(pred_plot)
                patch_i += 1
                
        #plot targetand prediction
        fig, axs = plt.subplots(1, 3, figsize=(6, 2))
        axs[0].imshow(img[0].cpu().numpy().transpose(1, 2, 0))
        axs[0].set_title('Image')
        axs[1].imshow(target_full[0].cpu().numpy(), cmap='tab20')
        axs[1].set_title('Target')
        axs[2].imshow(pred_full[0].cpu().numpy(), cmap='tab20')
        axs[2].set_title('Prediction')

        for ax in axs:
            ax.axis('off')
        if save:
            plt.savefig('finetune_example.png')
        if show:
            plt.show()
        plt.close()