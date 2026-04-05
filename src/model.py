from typing import Any, List, Optional

import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torch.optim import AdamW, lr_scheduler
from torchmetrics import MeanMetric

from src.losses import CrossEntropyLoss2d, FocalLoss2d, LovaszLoss2d, mIoULoss2d
from src.metrics import Metrics
from src.unet import UNet


class Model(LightningModule):  # noqa: WPS230 WPS214
    def __init__(self, cfg: DictConfig, weights: Optional[List[float]] = None) -> None:
        super().__init__()

        self.cfg = cfg
        self.weight = weights
        if self.cfg.cuda:
            torch.backends.cudnn.benchmark = True

        self._init_criterion()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_metric = Metrics(range(self.cfg.num_classes), self.cfg.metric)
        self.test_metric = Metrics(range(self.cfg.num_classes), self.cfg.metric)

        self.net = UNet(self.cfg.num_classes)

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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)

    def on_train_epoch_end(self) -> None:
        """вызывается в конце каждой эпохи обучения"""
        loss = self.train_loss.compute()
        self.train_loss.reset()
        self.log("train/loss", loss)

    def on_train_epoch_start(self) -> None:
        """эта функция вызывается в начале каждой эпохи обучения"""
        pass

    def on_validation_epoch_end(self) -> None:
        """вызывается в конце каждой эпохи валидации"""
        loss = self.val_loss.compute()
        self.val_loss.reset()

        metric = self.val_metric.compute()
        self.val_metric.reset()

        self.log("val/loss", loss)
        self.log("val/metric", metric, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """вызывается в конце тестирования"""
        metric = self.test_metric.compute()
        self.test_metric.reset()

        self.log("test/metric", metric)

    def test_step(self, batch: list[torch.Tensor], batch_idx: int) -> None:
        """вызывается для каждого батча при тестировании"""
        images, targets = batch
        logits = self(images)
        self.test_metric.add(logits, targets)

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """вызывается для каждого батча обучения"""
        images, targets = batch
        targets = targets.long()
        logits = self(images)
        loss = self.criterion(logits, targets)
        self.train_loss(loss)
        return loss

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> None:
        """вызывается для каждого батча на валидации"""
        images, targets = batch
        targets = targets.long()

        logits = self(images)
        loss = self.criterion(logits, targets)

        self.val_loss(loss)
        self.val_metric.add(logits, targets)

    def _init_criterion(self):
        if self.cfg.loss == "CrossEntropy":
            self.criterion = CrossEntropyLoss2d(weight=torch.Tensor(self.weight))
        elif self.cfg.loss == "mIoU":
            self.criterion = mIoULoss2d(weight=torch.Tensor(self.weight))
        elif self.cfg.loss == "Focal":
            self.criterion = FocalLoss2d(weight=torch.Tensor(self.weight))
        elif self.cfg.loss == "Lovasz":
            self.criterion = LovaszLoss2d()
