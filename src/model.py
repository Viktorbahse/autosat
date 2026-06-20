# src/model.py
from typing import Any, List, Optional, Tuple, Union

import segmentation_models_pytorch as smp
import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torch.optim import AdamW, lr_scheduler
from torchmetrics import MeanMetric

from src.losses import CrossEntropyLoss2d, FocalLoss2d, LovaszLoss2d, mIoULoss2d
from src.metrics import Metrics
from src.pspnet import PSPNet
from src.unet import UNet

RESNET50_NUMBER_OF_LAYERS = 50


class Model(LightningModule):  # noqa: 214
    def __init__(self, cfg: DictConfig, weights: Optional[List[float]] = None) -> None:
        super().__init__()

        self.save_hyperparameters(
            {"cfg_type": str(cfg.type), "num_classes": int(cfg.num_classes), "loss": str(cfg.loss)}
        )
        self.cfg = cfg
        self.weight = weights

        if self.cfg.cuda:
            torch.backends.cudnn.benchmark = True

        self._init_model()
        self._init_criterion()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_metric = Metrics(range(self.cfg.num_classes), self.cfg.metric)
        self.test_metric = Metrics(range(self.cfg.num_classes), self.cfg.metric)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)

    def configure_optimizers(self) -> dict[str, Any]:
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

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:  # noqa: WPS210
        images, targets = batch
        targets = targets.long()

        if self.net.training and hasattr(self.net, "aux_loss") and self.net.aux_loss:
            logits, aux_logits, _ = self.net(images, targets)

            main_loss = self.criterion(logits, targets)

            aux_loss = self.criterion(aux_logits, targets)

            loss = main_loss + self.net.aux_weight * aux_loss

            self.log("train/main_loss", main_loss, prog_bar=False)
            self.log("train/aux_loss", aux_loss, prog_bar=False)
        else:
            logits = self.net(images)
            loss = self.criterion(logits, targets)

        self.train_loss(loss)

        return loss

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> None:
        images, targets = batch
        targets = targets.long()

        logits = self.net(images)
        loss = self.criterion(logits, targets)

        self.val_loss(loss)
        self.val_metric.add(logits, targets)

    def test_step(self, batch: list[torch.Tensor], batch_idx: int) -> None:
        images, targets = batch
        logits = self(images)
        self.test_metric.add(logits, targets)

    def on_train_epoch_end(self) -> None:
        loss = self.train_loss.compute()
        self.train_loss.reset()
        self.log("train/loss", loss, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute()
        self.val_loss.reset()

        metric = self.val_metric.compute()
        self.val_metric.reset()

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/metric", metric, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        metric = self.test_metric.compute()
        self.test_metric.reset()
        self.log("test/metric", metric, prog_bar=True)

    def _init_model(self) -> None:
        if self.cfg.type == "unet50":
            self.net = UNet(self.cfg.num_classes)
        elif self.cfg.type == "pspnet50":
            self.net = PSPNet(
                num_classes=self.cfg.num_classes,
                layers=RESNET50_NUMBER_OF_LAYERS,
                bins=(1, 2, 3, 6),
            )
        elif self.cfg.type == "deeplabv3plus":
            self.net = smp.DeepLabV3Plus(
                encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=int(self.cfg.num_classes)
            )
        else:
            raise ValueError("Unknown model type!")

    def _init_criterion(self) -> None:  # noqa: WPS231
        if self.weight is None:
            self.weight = [1.0 for _ in range(int(self.cfg.num_classes))]

        weight_tensor = torch.Tensor(self.weight)

        if self.cfg.loss == "CrossEntropy":
            self.criterion = CrossEntropyLoss2d(weight=weight_tensor)
        elif self.cfg.loss == "mIoU":
            self.criterion = mIoULoss2d(weight=weight_tensor)
        elif self.cfg.loss == "Focal":
            self.criterion = FocalLoss2d(weight=weight_tensor)
        elif self.cfg.loss == "Lovasz":
            self.criterion = LovaszLoss2d()
        else:
            raise ValueError("Unknown loss type!")
