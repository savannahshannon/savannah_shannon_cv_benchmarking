"""
CNN Model Factory - Instantiate various pretrained architectures for transfer learning.
Supports: ResNet50, ResNet18, DenseNet121, EfficientNetB0, MobileNetV2, 
        ViT-B16, ConvNeXt-Tiny, AlexNet, VGG16, Inception-V3
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


def get_cnn_model(model_name: str, num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    get a pretrained CNN model with the final classification layer replaced.
    args:
        model_name: Name of the architecture (resnet50, resnet18, densenet121, etc.)
        num_classes: Number of output classes (default 10 for CIFAR-10/MNIST with padding)
        pretrained: Whether to use pretrained weights
    returns:
        model with replaced final layer for num_classes prediction
    """
    
    weights = "DEFAULT" if pretrained else None
    
    if model_name.lower() == "resnet50":
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name.lower() == "resnet18":
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name.lower() == "densenet121":
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        
    elif model_name.lower() == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    elif model_name.lower() == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    elif model_name.lower() == "vit_b_16":
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        
    elif model_name.lower() == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        
    elif model_name.lower() == "alexnet":
        model = models.alexnet(weights=weights)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        
    elif model_name.lower() == "vgg16":
        model = models.vgg16(weights=weights)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        
    elif model_name.lower() == "inception_v3":
        model = models.inception_v3(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        # Inception_V3 also has auxiliary classifiers
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights='DEFAULT')
        model.classifier[3] = nn.Linear(1024, num_classes)
    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights='DEFAULT')
        model.classifier[3] = nn.Linear(1280, num_classes)
    elif model_name == 'googlenet':
        model = models.googlenet(weights='DEFAULT', aux_logits=True)
        model.fc = nn.Linear(1024, num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Supported: resnet50, resnet18, densenet121, efficientnet_b0, "
                        f"mobilenet_v2, vit_b_16, convnext_tiny, alexnet, vgg16, inception_v3")
    
    return model


def count_parameters(model: nn.Module) -> tuple:
    """
    count total and trainable parameters in a model.
    returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
