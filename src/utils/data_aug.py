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
        common_transforms = [
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
            transforms.ToTensor(),
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
            transforms.ToTensor(),
        ]

        return (
            transforms.Compose(common_transforms + view1_transforms),
            transforms.Compose(common_transforms + view2_transforms),
        )
