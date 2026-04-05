from pathlib import Path
from typing import Callable, List, Optional, Tuple

import albumentations as A
import h5py
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

NUMBER_OF_PIXELS_SUFFIX = "_number_of_pixels"
MASK_KEY = "mask"
DATA_KEY = "data"


class H5Dataset(Dataset):  # noqa: WPS230
    def __init__(self, files: List[Path], image_size: int, transform: Optional[Callable] = None):
        self.files = [Path(p) for p in files]
        self.transform = transform
        self.image_size = image_size
        with h5py.File(self.files[0], "r") as f:  # noqa: WPS226
            h, w, c = f[DATA_KEY].shape
        self.tiles_y = h // self.image_size
        self.tiles_x = w // self.image_size
        self.images_in_file = self.tiles_y * self.tiles_x
        self.len = len(self.files) * self.images_in_file

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: WPS210
        file_idx = idx // self.images_in_file
        image_idx = idx % self.images_in_file

        row_idx = image_idx // self.tiles_y
        col_idx = image_idx % self.tiles_x

        y0 = row_idx * self.image_size
        y1 = y0 + self.image_size

        x0 = col_idx * self.image_size
        x1 = x0 + self.image_size

        with h5py.File(self.files[file_idx], "r") as f:
            c = f[DATA_KEY].shape[2]
            z0 = np.random.randint(0, (c - 4) // 3 + 1)
            z1 = z0 + 3
            img = f[DATA_KEY][y0:y1, x0:x1, z0:z1].astype(np.float32)
            mask = f[DATA_KEY][y0:y1, x0:x1, -1].astype(np.uint8)
            i = 0
            for k, v in f.attrs.items():
                if k not in ["x", "y", "zoom"] and NUMBER_OF_PIXELS_SUFFIX not in k:
                    mask[mask == v] = i
                    i += 1

        mask = mask.astype(np.int64)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            key = "image"
            img = augmented[key]
            mask = augmented[MASK_KEY]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
            img = img / float(255)  # noqa: WPS432
            mask = torch.from_numpy(mask).long()

        return img, mask


class Loader(LightningDataModule):  # noqa: WPS230
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_files = None
        self.val_files = None
        self.test_files = None
        self.weights = None
        self.classes = None
        self.classes_pixels_values = None

    def get_files(self):
        dataset_dir = Path(self.cfg.dataset_dir)
        if not dataset_dir.exists():
            return []
        return list(dataset_dir.glob("*.h5"))

    def split(self, files: List[Path]):
        files = np.array(files)
        indices = np.random.permutation(len(files))
        files = files[indices]
        n = len(files)
        n_train = int(round(self.cfg.train_ratio * n))
        n_val = int(round(self.cfg.val_ratio * n))
        self.train_files = files[:n_train]
        self.val_files = files[n_train : n_train + n_val]
        self.test_files = files[n_train + n_val :]

    def calculate_weights(self):  # noqa: WPS210
        classes = []
        classes_pixels_values = []
        classes_number_of_pixels = {}
        with h5py.File(self.train_files[0], "r") as f:
            for attr in f.attrs:
                if attr.endswith(NUMBER_OF_PIXELS_SUFFIX):
                    class_name = attr.replace(NUMBER_OF_PIXELS_SUFFIX, "")
                    classes.append(class_name)
                    classes_pixels_values.append(f.attrs.get(class_name, 0))
                    classes_number_of_pixels[class_name] = 0

        total = 0
        for dataset_file in self.train_files:
            with h5py.File(dataset_file, "r") as f:
                for class_name in classes:
                    npx = f.attrs.get(f"{class_name}_number_of_pixels", 0)
                    classes_number_of_pixels[class_name] += npx
                    total += npx
        weights = []
        for class_name in classes:
            w = (
                (total / (len(classes) * classes_number_of_pixels[class_name]))
                if classes_number_of_pixels[class_name] > 0
                else float(0)
            )
            weights.append(w)

        self.weights = weights
        self.classes = classes
        self.classes_pixels_values = classes_pixels_values

    def prepare_data(self):
        files = self.get_files()
        self.split(files)
        self.calculate_weights()

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_tf = A.Compose(
                [
                    A.RandomCrop(self.cfg.image_size, self.cfg.image_size),
                    A.RandomRotate90(p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                additional_targets={MASK_KEY: MASK_KEY},
            )

            test_tf = A.Compose(
                [
                    A.RandomCrop(self.cfg.image_size, self.cfg.image_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                additional_targets={MASK_KEY: MASK_KEY},
            )

            self.prepare_data()
            self.train_dataset = H5Dataset(self.train_files, image_size=self.cfg.image_size, transform=train_tf)
            self.val_dataset = H5Dataset(self.val_files, image_size=self.cfg.image_size, transform=train_tf)
            self.test_dataset = H5Dataset(self.test_files, image_size=self.cfg.image_size, transform=test_tf)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, shuffle=False
        )
