"""
Deep Learning models for Image Forgery Detection
Includes: CNN, ResNet, EfficientNet, Xception, Vision Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from transformers import ViTModel, ViTConfig
from typing import Optional

from config import ModelConfig, TrainingConfig


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for image forgery detection
    """
    def __init__(
        self,
        num_classes: int = 2,
        filters: list = [32, 64, 128, 256],
        dropout: float = 0.5
    ):
        super(SimpleCNN, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        in_channels = 3
        
        for out_channels in filters:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2)
                )
            )
            in_channels = out_channels
        
        # Calculate flattened size (assuming 224x224 input)
        self.flattened_size = filters[-1] * (224 // (2 ** len(filters))) ** 2
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNetForForgeryDetection(nn.Module):
    """
    ResNet-based model for forgery detection with attention mechanism
    """
    def __init__(
        self,
        num_classes: int = 2,
        depth: int = 50,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super(ResNetForForgeryDetection, self).__init__()
        
        # Load pretrained ResNet
        if depth == 18:
            resnet = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif depth == 34:
            resnet = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif depth == 50:
            resnet = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif depth == 101:
            resnet = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        elif depth == 152:
            resnet = models.resnet152(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")
        
        # Extract backbone (all layers except final FC)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        attention_map = self.attention(features)
        attended_features = features * attention_map
        
        # Global pooling
        pooled = self.gap(attended_features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output, attention_map


class EfficientNetForForgeryDetection(nn.Module):
    """
    EfficientNet-based model for forgery detection
    """
    def __init__(
        self,
        num_classes: int = 2,
        variant: str = 'b0',
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super(EfficientNetForForgeryDetection, self).__init__()
        
        # Load pretrained EfficientNet
        self.backbone = timm.create_model(
            f'efficientnet_{variant}',
            pretrained=pretrained,
            features_only=False
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get feature dimension
        feature_dim = self.backbone.classifier.in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class XceptionForForgeryDetection(nn.Module):
    """
    Xception-based model for forgery detection
    """
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super(XceptionForForgeryDetection, self).__init__()
        
        # Load pretrained Xception
        self.backbone = timm.create_model(
            'xception',
            pretrained=pretrained,
            num_classes=0  # Remove classifier
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get feature dimension
        feature_dim = 2048  # Xception output dimension
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class VisionTransformerForForgeryDetection(nn.Module):
    """
    Vision Transformer (ViT) for forgery detection
    """
    def __init__(
        self,
        num_classes: int = 2,
        patch_size: int = 16,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super(VisionTransformerForForgeryDetection, self).__init__()
        
        # Load pretrained ViT
        model_name = 'google/vit-base-patch16-224' if patch_size == 16 else f'google/vit-base-patch{patch_size}-224'
        
        self.backbone = ViTModel.from_pretrained(
            model_name if pretrained else None,
            config=ViTConfig(
                image_size=224,
                patch_size=patch_size,
                hidden_size=dim,
                num_hidden_layers=depth,
                num_attention_heads=heads
            ) if not pretrained else None
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # ViT from transformers expects pixel_values
        # Input x is already normalized from data loader
        outputs = self.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]  # CLS token
        output = self.classifier(cls_token)
        return output


def create_model(
    model_name: str = 'resnet',
    num_classes: int = 2,
    config: TrainingConfig = None,
    model_config: ModelConfig = None
) -> nn.Module:
    """
    Factory function to create a model
    """
    if config is None:
        from config import TrainingConfig
        config = TrainingConfig()
    
    if model_config is None:
        from config import ModelConfig
        model_config = ModelConfig()
    
    if model_name.lower() == 'cnn':
        model = SimpleCNN(
            num_classes=num_classes,
            filters=model_config.CNN_FILTERS,
            dropout=model_config.CNN_DROPOUT
        )
    
    elif model_name.lower() == 'resnet':
        model = ResNetForForgeryDetection(
            num_classes=num_classes,
            depth=model_config.RESNET_DEPTH,
            pretrained=config.USE_PRETRAINED,
            freeze_backbone=config.FREEZE_BACKBONE
        )
    
    elif model_name.lower() == 'efficientnet':
        model = EfficientNetForForgeryDetection(
            num_classes=num_classes,
            variant=model_config.EFFICIENTNET_VARIANT,
            pretrained=config.USE_PRETRAINED,
            freeze_backbone=config.FREEZE_BACKBONE
        )
    
    elif model_name.lower() == 'xception':
        model = XceptionForForgeryDetection(
            num_classes=num_classes,
            pretrained=config.USE_PRETRAINED,
            freeze_backbone=config.FREEZE_BACKBONE
        )
    
    elif model_name.lower() == 'vit':
        model = VisionTransformerForForgeryDetection(
            num_classes=num_classes,
            patch_size=model_config.VIT_PATCH_SIZE,
            dim=model_config.VIT_DIM,
            depth=model_config.VIT_DEPTH,
            heads=model_config.VIT_HEADS,
            pretrained=config.USE_PRETRAINED,
            freeze_backbone=config.FREEZE_BACKBONE
        )
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model

