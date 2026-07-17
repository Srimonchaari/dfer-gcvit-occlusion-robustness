# ============================================================
# FILE 4: models/gcvitt.py
# PURPOSE: Defines the GC-ViT model with a custom classification
#          head for 6 emotion classes.
# THIS IS THE MODEL WE USED for all training and evaluation.
# ============================================================

import timm
import torch.nn as nn
import torch
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from datetime import datetime
import pdb


class VanillaSwinT1(nn.Module):
    def __init__(self, n_classes: int, size: str = "small"):
        super(VanillaSwinT1, self).__init__()

        # ============================================================
        # STEP 1: Load pretrained GC-ViT Base from timm library
        # ← timm = PyTorch Image Models library (600+ pretrained models)
        # ← 'gcvit_base' = Global Context Vision Transformer, Base size
        # ← pretrained=True = download weights already trained on ImageNet
        #   (1.2 million images, 1000 categories)
        # ← WHY PRETRAINED? The model already knows edges, textures, shapes,
        #   and face structures. We don't train that from scratch.
        #   This is called TRANSFER LEARNING.
        # ← WHY GC-ViT? It combines local window attention (efficient) with
        #   global context tokens — captures both fine details (wrinkles around
        #   eyes) and whole-face relationships at the same time.
        # ============================================================
        self.model = timm.create_model('gcvit_base', pretrained=True)

        # ============================================================
        # STEP 2: Replace the classification head
        # ← The pretrained model's original head outputs 1000 scores
        #   (one per ImageNet category: cat, dog, airplane, etc.)
        #   That is USELESS for our task.
        # ← We replace it with our own head that outputs 6 scores
        #   (one per emotion: anger, disgust, fear, happy, sadness, surprise)
        # ← The body (feature extractor) stays UNCHANGED — we keep all
        #   the learned knowledge about understanding images.
        # ============================================================
        new_layers = nn.Sequential(
            nn.Linear(1024, 512),   # ← GC-ViT body outputs 1024 features → compress to 512
                                    #   Think of it as: 1024 clues → pick the 512 most useful ones
            nn.BatchNorm1d(512),    # ← Normalize the 512 numbers to be stable (mean=0, std=1)
                                    #   Prevents numbers from exploding or vanishing during training
            nn.ReLU(),              # ← Any negative number becomes 0
                                    #   Adds non-linearity so the model can learn complex patterns
            nn.Dropout(0.5),        # ← During training: randomly switch off 50% of neurons
                                    #   Forces the model to not rely on any single feature
                                    #   Prevents memorizing — model must learn robust features
            nn.Linear(512, n_classes),  # ← Final decision layer: 512 → 6 emotion scores
                                        #   Whichever of the 6 scores is highest = predicted emotion
        )
        self.model.head.fc = new_layers   # ← Swap the original 1000-class head with our 6-class head

    def forward(self, x):
        # ← forward() is called automatically when you do: output = model(image)
        # ← x is a batch of images (shape: batch_size × channels × 224 × 224)
        # ← GC-ViT body extracts features, then our custom head outputs 6 scores
        x = self.model(x)
        return x   # ← returns 6 scores per image — highest score = predicted emotion
