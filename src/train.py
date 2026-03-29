from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import h5py
import numpy as np
import rootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanMetric

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.losses import CrossEntropyLoss2d, FocalLoss2d, LovaszLoss2d, mIoULoss2d
from src.metrics import Metrics
from src.unet import UNet
from src.utils import make_dir

MASK_KEY = "mask"
DATA_KEY = "data"
NUMBER_OF_PIXELS_SUFFIX = "_number_of_pixels"


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


class Model(LightningModule):  # noqa: WSP 214
    def __init__(self, cfg: DictConfig, weights: Optional[List[float]] = None) -> None:
        super().__init__()

        self.cfg = cfg
        self.weight = weights
        if self.cfg.cuda:
            torch.backends.cudnn.benchmark = True

        self._init_criterion()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_metric = Metrics(range(self.cfg.num_classes))
        self.test_metric = Metrics(range(self.cfg.num_classes))

        self.net = UNet(self.cfg.num_classes)

    def _init_criterion(self):
        if self.cfg.loss == "CrossEntropy":
            self.criterion = CrossEntropyLoss2d(weight=torch.Tensor(self.weight))
        elif self.cfg.loss == "mIoU":
            self.criterion = mIoULoss2d(weight=torch.Tensor(self.weight))
        elif self.cfg.loss == "Focal":
            self.criterion = FocalLoss2d(weight=torch.Tensor(self.weight))
        elif self.cfg.loss == "Lovasz":
            self.criterion = LovaszLoss2d()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)

    def configure_optimizers(self) -> dict[str, Any]:
        """создаем и возвращаем оптимизатор и scheduler"""

        optimizer = AdamW(
            params=self.net.parameters(),
            lr=self.cfg.opt.lr,
            betas=self.cfg.opt.betas,
            eps=self.cfg.opt.eps,
            weight_decay=self.cfg.opt.weight_decay,
        )

        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.sch.num_epoch, eta_min=self.cfg.sch.eta_min
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self) -> None:
        """эта функция вызывается в начале каждой эпохи обучения"""
        pass

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """вызывается для каждого батча обучения"""
        images, targets = batch
        targets = targets.long()
        logits = self(images)
        loss = self.criterion(logits, targets)
        self.train_loss(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        """вызывается в конце каждой эпохи обучения"""
        loss = self.train_loss.compute()
        self.train_loss.reset()
        self.log("train/loss", loss)

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> None:
        """вызывается для каждого батча на валидации"""

        images, targets = batch
        targets = targets.long()

        logits = self(images)
        loss = self.criterion(logits, targets)

        self.val_loss(loss)
        self.val_metric.add(logits, targets)

    def on_validation_epoch_end(self) -> None:
        """вызывается в конце каждой эпохи валидации"""

        loss = self.val_loss.compute()
        self.val_loss.reset()

        metric = self.val_metric.compute()
        self.val_metric.reset()

        self.log("val/loss", loss)
        self.log("val/metric", metric, prog_bar=True)

    def test_step(self, batch: list[torch.Tensor], batch_idx: int) -> None:
        """вызывается для каждого батча при тестировании"""

        images, targets = batch
        logits = self(images)
        self.test_metric.add(logits, targets)

    def on_test_epoch_end(self) -> None:
        """вызывается в конце тестирования"""
        metric = self.test_metric.compute()
        self.test_metric.reset()

        self.log("test/metric", metric)


def main(cfg: DictConfig):  # noqa: WPS210
    # set_seed(cfg.random_seed)

    make_dir(cfg.checkpoint_dir, delete_if_exist=True)

    loader = Loader(cfg.loader)
    loader.prepare_data()

    model = Model(cfg.model, weights=loader.weights)

    callbacks = [
        # логирование learning rate
        LearningRateMonitor(logging_interval="epoch"),
        # сохраняет две лучшие модели по метрике на валидации
        ModelCheckpoint(save_top_k=2, monitor="val/metric", mode="max", every_n_epochs=1),
    ]

    # будем логировать метрики с помощью tensorboard
    logger = TensorBoardLogger(
        save_dir="data/logs",  # путь к логам
        name="tensorboard",
    )

    # Trainer сам выберет лучший ускоритель, если он явно не задан
    trainer = Trainer(
        max_epochs=cfg.num_epoch,
        default_root_dir="data/logs",  # путь к checkpoints
        callbacks=callbacks,
        logger=logger,
    )

    # train / val loop
    trainer.fit(model, loader)

    # тестирование
    trainer.test(model, loader)


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg.training
    main(cfg)
