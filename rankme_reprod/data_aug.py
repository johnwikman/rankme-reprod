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

import torchvision.transforms as transforms


class BYOLTransform:
    def __init__(self, crop_size):
        v1, v2 = byol(crop_size)
        self.v1 = v1
        self.v2 = v2

    def __call__(self, x):
        return [self.v1(x), self.v2(x)]


def byol(crop_size):
    """
    The BYOL transform, with parameters from the 
    """
    common_transforms = [
        # random_crop_probability = 1.0
        transforms.RandomResizedCrop(size=crop_size),

        # horizontal_flip_probability = 0.5
        transforms.RandomHorizontalFlip(p=0.5),

        # color_jittering_probability = 0.8
        transforms.RandomApply([transforms.ColorJitter(
                # brightness_adjustment_max_intensity = 0.4
                brightness=0.4,
                # contrast_adjustment_max_intensity = 0.4
                contrast=0.4,
                # saturation_adjustment_max_intensity = 0.2
                saturation=0.2,
                # hue_adjustment_max_intensity = 0.1
                hue=0.1,
            )],
            p=0.8,
        ),

        # grayscale_probability = 0.2
        transforms.RandomGrayscale(p=0.2),
    ]

    # From BYOL paper, the gaussian kernel size is 23x23 and sigma is selected
    # randomly from the range (0.1, 2.0)

    view1_transforms = [
        # v1_gaussian_blurring_probability = 1.0
        transforms.RandomApply([transforms.GaussianBlur(
                kernel_size=23,
                sigma=(0.1, 2.0),
            )],
            p=1.0,
        ),

        # v1_solarization_probability = 0.0
        # pass, no solarization will be done here

        transforms.ToTensor(),
    ]

    view2_transforms = [
        # v2_gaussian_blurring_probability = 0.1
        transforms.RandomApply([transforms.GaussianBlur(
                kernel_size=23,
                sigma=(0.1, 2.0),
            )],
            p=0.1,
        ),

        # v2_solarization_probability = 0.2
        transforms.RandomSolarize(
            # In BYOL, they describe this transform as
            # x -> x * 1{x<0.5} + (1 − x) * 1{x≥0.5}
            threshold=0.5,
            p=0.2
        ),

        transforms.ToTensor(),
    ]

    return (transforms.Compose(view1_transforms), transforms.Compose(view2_transforms))
