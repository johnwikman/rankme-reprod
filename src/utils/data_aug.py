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
import PIL.Image


class SplitTransform:
    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def __call__(self, x):
        return [self.t1(x), self.t2(x)]



class BYOLTransform:
    def __init__(self, crop_size):
        """
        Recieve two views of the same image, and apply the BYOL transform to them.
        """
        v1, v2 = self._byol(crop_size)
        self.v1 = v1
        self.v2 = v2

    def __call__(self, x: PIL.Image) -> [PIL.Image, PIL.Image]:
        """
        Parameters
        ----------
        x : PIL.Image
            The image to transform.

        """
        return [self.v1(x), self.v2(x)]

    def _byol(self, crop_size) -> (transforms.Compose, transforms.Compose):
        """
        The BYOL transform, with parameters from the paper.
        Parameters
        ----------
        crop_size : int
            The size of the random crop to take from the image.

        Returns
        -------
        (transforms.Compose, transforms.Compose)
            The transforms for the two views.
        """
        common_in = [
            transforms.RandomResizedCrop(size=crop_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.1,
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
        ]

        # From BYOL paper:
        # "the gaussian kernel size is 23x23 and sigma is selected
        # randomly from the range (0.1, 2.0)"

        view1_transforms = [
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=23,
                        sigma=(0.1, 2.0),
                    )
                ],
                p=1.0,
            ),
            # v1_solarization_probability = 0.0
            # pass, no solarization will be done here
        ]

        view2_transforms = [
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=23,
                        sigma=(0.1, 2.0),
                    )
                ],
                p=0.1,
            ),
            transforms.RandomSolarize(
                # In BYOL, they describe this transform as
                # x -> x * 1{x<0.5} + (1 − x) * 1{x≥0.5}
                threshold=0.5,
                p=0.2,
            ),
        ]

        common_out = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        return (
            transforms.Compose(common_in + view1_transforms + common_out),
            transforms.Compose(common_in + view2_transforms + common_out),
        )
