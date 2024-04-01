import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

class SimMIM(nn.Module):
    """
    SimMIM model class.
    -------------------
    Work in progress
    """
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        """
        Args:
            encoder (nn.Module): Vision Transformer to be trained.
            masking_ratio (float): Ratio of patches to be masked (default 0.5).
        """
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # Get hyperparameters and functions from encoder (ViT to be trained)
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # Linear head (decoder) to predict pixel values
        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_pixels = nn.Linear(encoder_dim, pixel_values_per_patch)

    def forward(self, img):
        """
        Run a forward pass of the SimMIM model.

        Args:
            img (torch.Tensor): Batch of images to be masked and reconstructed.

        Returns:
            float: Reconstruction loss for the batch.
            torch.Tensor: Predicted pixel values for masked patches.
            torch.Tensor: Ground truth masked patches.
        """
        
        device = img.device

        # Get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # For indexing
        batch_range = torch.arange(batch, device = device)[:, None]

        # Get positions
        pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # Patch to embeddings
        tokens = self.patch_to_emb(patches)
        tokens = tokens + pos_emb

        # Prepare mask tokens
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb

        # Calculate number of masked tokens and create a boolean mask
        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # Mask tokens
        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # Encode the masked tokens
        encoded = self.encoder.transformer(tokens)

        # Get the masked tokens
        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # Decode the masked tokens (predict pixel values)
        pred_pixel_values = self.to_pixels(encoded_mask_tokens)

        # Get the original masked patches
        masked_patches = patches[batch_range, masked_indices]

        # Calculate reconstruction loss (L1 loss)
        recon_loss = F.l1_loss(pred_pixel_values, masked_patches) / num_masked
        return recon_loss, pred_pixel_values, masked_patches