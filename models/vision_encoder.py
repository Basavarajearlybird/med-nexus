"""
Med-Nexus Vision Encoder
PyTorch visual backbone for medical chest X-ray feature extraction using pre-trained DenseNet-121.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import models

class MedicalVisionEncoder(nn.Module):
    """
    Robust visual encoder optimized for chest X-ray feature extraction using 
    DenseNet-121 (commonly used for CheXpert/NIH datasets).
    """
    
    def __init__(self, in_channels: int = 3, embedding_dim: int = 256, pretrained: bool | None = None):
        super().__init__()
        
        # Pretrained weights are recommended for real training, but local tests must also
        # work offline. MED_NEXUS_PRETRAINED=0 disables the download explicitly.
        if pretrained is None:
            pretrained = os.getenv("MED_NEXUS_PRETRAINED", "1") != "0"
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            densenet = models.densenet121(weights=weights)
        except Exception as exc:
            print(f"[Warning] DenseNet pretrained weights unavailable: {exc}. Using uninitialized weights for smoke/testing only.")
            densenet = models.densenet121(weights=None)
        
        # Extract features (exclude classifier head)
        self.features = densenet.features
        
        # Modify first conv layer if we have different input channels (e.g. grayscale)
        if in_channels != 3:
            original_conv = self.features.conv0
            self.features.conv0 = nn.Conv2d(
                in_channels, original_conv.out_channels, 
                kernel_size=original_conv.kernel_size, 
                stride=original_conv.stride, 
                padding=original_conv.padding, 
                bias=False
            )
        
        # Linear projection to our unified embedding space
        num_features = densenet.classifier.in_features # 1024 for DenseNet121
        self.fc_proj = nn.Linear(num_features, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Image tensor of shape [batch_size, channels, height, width]
        Returns:
            Normalized feature vector of shape [batch_size, embedding_dim]
        """
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        
        proj = self.fc_proj(out)
        return F.normalize(proj, p=2, dim=-1)
