"""
This implements the data augmentation strategies from
"Bootstrap your own latent: A new approach to self-supervised Learning"
by Grill et al (2020).
https://arxiv.org/abs/2006.07733

They are:
 - Random crop
 - Horizontal flip
 - Color jittering
 - Brightness adjustment
 - Contrast adjustment
 - Saturation adjustment
 - Hue adjustment
 - Grayscale
 - Gaussian blurring
 - Solarization
"""

import torchvision


def byol():
    return torchvision.transforms.v2.RandomCrop()

