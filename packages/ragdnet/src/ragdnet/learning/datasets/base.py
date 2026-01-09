from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

IMG_EXTS = (".png", ".jpg", ".jpeg")


def natural_key(path: str) -> list:
    """Key function for natural sort (e.g. img2.png before img10.png)."""
    name = os.path.basename(path)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


def list_images(
    root: str | Path,
) -> List[str]:
    """
    List image files under `root` (non-recursive) with deterministic order.
    """
    root = Path(root)

    images: List[str] = []
    for p in root.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        images.append(str(p))

    # deterministic, natural ordering
    return sorted(images, key=natural_key)


class TrainEvalDatasetBase:
    """
    Base class for datasets that have a train/eval mode flag.
    """

    def __init__(self, train: bool = True) -> None:
        self.train: bool = train

    def set_train(self, train: bool = True) -> None:
        """Switch dataset between train and eval mode."""
        self.train = train


class AlignedImageDatasetBase(TrainEvalDatasetBase):
    """
    Base class for image datasets that:
    - keep a deterministic, aligned ordering of files
    - expose `image_paths` and `get_image_path(idx)`
    - support train/eval mode via TrainEvalDatasetBase
    """

    def __init__(self, root: str | Path, *, train: bool = True) -> None:
        super().__init__(train=train)
        self.root = Path(root)
        self.image_paths: List[str] = list_images(self.root)

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_image_path(self, idx: int) -> str:
        return self.image_paths[idx]
