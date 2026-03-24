from pathlib import Path

import h5py
import numpy as np
import rootutils
import torch
import torch.backends.cudnn
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
from typing import Callable, List, Optional, Tuple

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from src.utils import make_dir, set_seed

train_tf = A.Compose(
    [A.RandomCrop(224, 224), A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.5), A.Normalize(), ToTensorV2()],
    additional_targets={"mask": "mask"},
)


class H5Dataset(Dataset):
    def __init__(self, files: List[Path], image_size: int, transform: Optional[Callable] = None):
        self.files = [Path(p) for p in files]
        self.transform = transform
        self.image_size = image_size

        with h5py.File(self.files[0], "r") as f:
            self.files_shape = f["data"].shape
        h, w, _ = self.files_shape
        self.tiles_y = h // self.image_size
        self.tiles_x = w // self.image_size
        self.images_in_file = self.tiles_y * self.tiles_x
        self.len = len(self.files) * self.images_in_file

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx = idx // self.images_in_file
        image_idx = idx % self.images_in_file

        row_idx = image_idx // self.tiles_y
        col_idx = image_idx % self.tiles_x

        y0 = row_idx * self.image_size
        y1 = y0 + self.image_size
        x0 = col_idx * self.image_size
        x1 = x0 + self.image_size

        with h5py.File(self.files[file_idx], "r") as f:
            channels = f["data"].shape[2]
            max_rgb_start = max(0, (channels - 1) - 3)
            z0 = np.random.randint(0, max_rgb_start + 1) if max_rgb_start > 0 else 0
            z1 = z0 + 3
            img = f["data"][y0:y1, x0:x1, z0:z1].astype(np.float32)
            mask = f["data"][y0:y1, x0:x1, -1].astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img_t = augmented["image"]
            mask_t = augmented["mask"]
            mask_t = torch.as_tensor(mask_t, dtype=torch.long)
        else:
            img_t = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
            mask_t = torch.from_numpy(mask).long()
        return img_t, mask_t


class Loader(LightningDataModule):
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
        return (files[:n_train], files[n_train : n_train + n_val], files[n_train + n_val :])

    def calculate_weights(self):
        classes = []
        classes_pixels_values = []
        classes_number_of_pixels = {}
        with h5py.File(self.train_files[0], "r") as f:
            for k, v in f.attrs.items():
                if k in ["x", "y", "zoom"] or "_number_of_pixels" not in k and k.endswith("_number_of_pixels") is False:
                    pass
            for attr in f.attrs:
                if attr.endswith("_number_of_pixels"):
                    k = attr.replace("_number_of_pixels", "")
                    classes.append(k)
                    classes_pixels_values.append(f.attrs.get(k, 0))
                    classes_number_of_pixels[k] = 0

        total = 0
        for file in self.train_files:
            with h5py.File(file, "r") as f:
                for k in classes:
                    npx = f.attrs.get(k + "_number_of_pixels", 0)
                    classes_number_of_pixels[k] += npx
                    total += npx

        weights = [
            (total / (len(classes) * classes_number_of_pixels[k])) if classes_number_of_pixels[k] > 0 else 0.0
            for k in classes
        ]
        return weights, classes, classes_pixels_values

    def prepare_data(self):
        files = self.get_files()
        self.train_files, self.val_files, self.test_files = self.split(files)
        self.weights, self.classes, self.classes_pixels_values = self.calculate_weights()

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.prepare_data()
            self.train_dataset = H5Dataset(self.train_files, image_size=self.cfg.image_size, transform=train_tf)
            self.val_dataset = H5Dataset(self.val_files, image_size=self.cfg.image_size, transform=train_tf)
            self.test_dataset = H5Dataset(self.test_files, image_size=self.cfg.image_size, transform=train_tf)

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


def main(cfg: DictConfig):
    set_seed(cfg.random_seed)

    device = torch.device("cuda" if cfg.cuda else "cpu")
    checkpoint_dir = make_dir(cfg.checkpoint_dir, delete_if_exist=True)

    print(device, checkpoint_dir)
    loader = Loader(cfg.loader)
    loader.prepare_data()
    """
    num_classes = len(cfg.classes)
    net = UNet(num_classes)
    net = DataParallel(net)
    net = net.to(device)
    """


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg.training
    main(cfg)
