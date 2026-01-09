# sim_core/learning/datasets/image_segmentation.py
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ragdnet.learning.datasets.base import AlignedImageDatasetBase
from ragdnet.utils.color_mapper import map_color_to_id_image
import torch 


class DrawingDataset(AlignedImageDatasetBase, Dataset):
    def __init__(
        self,
        images_dir: str,
        train: bool = True,
        num_classes: int = 4,
    ):
        # Build aligned image_paths + train flag
        AlignedImageDatasetBase.__init__(self, root=images_dir, train=train)

        self.num_classes = num_classes

        # Train-time transform
        self.train_transform = A.Compose([
            A.PadIfNeeded(
                512,
                512,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=-1,
            ),
            A.RandomCrop(width=512, height=512, p=1.0),
            A.HorizontalFlip(p=0.5),
            ToTensorV2(),
        ])

        # Eval-time transform
        self.eval_transform = A.Compose([
            A.PadIfNeeded(
                512,
                512,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=-1,
            ),
            ToTensorV2(),
        ])

    def __getitem__(self, idx):
        img_path = self.get_image_path(idx)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = (image < 255).astype(np.uint8)

        mask = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        mask = map_color_to_id_image(mask)
        
        transform = self.train_transform if self.train else self.eval_transform
        transformed = transform(image=image, mask=mask)

        image = (transformed["image"]).float()
        mask = transformed["mask"].long()

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        assert len(torch.unique(image)) <= 2, "Image is not binary"
        assert torch.min(image) == 0, "Image is not binary"
        assert torch.max(image) <= 1, "Image is not binary"

        assert torch.max(mask) < self.num_classes, "Number class error"

        return image, mask
