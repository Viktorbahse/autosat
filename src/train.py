from pathlib import Path

import numpy as np
import rootutils
from dvclive.lightning import DVCLiveLogger
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
import h5py
import torch
from PIL import Image

from src.loader import Loader
from src.model import Model
from src.utils import image_cutting, make_dir, make_mask

MASK_KEY = "mask"

NUMBER_OF_PIXELS_SUFFIX = "_number_of_pixels"


def main(cfg: DictConfig):  # noqa: WPS210 WPS213 WPS231
    # set_seed(cfg.random_seed)

    make_dir(cfg.logs_dir, delete_if_exist=False)

    loader = Loader(cfg.loader)
    loader.prepare_data()

    model = Model(cfg.model, weights=loader.weights)

    callbacks = [
        # логирование learning rate
        LearningRateMonitor(logging_interval="epoch"),
        # сохраняет две лучшие модели по метрике на валидации
        ModelCheckpoint(save_top_k=2, monitor="val/metric", mode="max", every_n_epochs=1),
    ]

    logger = DVCLiveLogger(run_name="model_training", log_model=True, dir=f"{cfg.logs_dir}/dvclive")

    # Trainer сам выберет лучший ускоритель, если он явно не задан
    trainer = Trainer(
        max_epochs=cfg.num_epoch,
        default_root_dir=f"{cfg.logs_dir}",  # путь к checkpoints
        callbacks=callbacks,
        logger=logger,
    )

    # train / val loop
    trainer.fit(model, loader)

    # тестирование
    trainer.test(model, loader)

    if cfg.predict_on_test_data:
        files = loader.get_test_files()

        version_dirs = Path(cfg.logs_dir) / "DvcLiveLogger/model_training/checkpoints/"
        latest_version = [f for f in version_dirs.iterdir() if f.is_file()]
        if latest_version:
            latest_checkpoint = str(latest_version[0])  # или выберите нужный
            model = Model.load_from_checkpoint(checkpoint_path=latest_checkpoint, cfg=cfg.model)
            make_dir(Path("data/predict/"), True)
            model.eval()
            model.to("cuda" if cfg.model.cuda else "cpu")
            for i, file_h5 in enumerate(files):
                path = Path(f"data/predict/{i}/")
                make_dir(path, True)
                with h5py.File(file_h5, "r") as f:
                    img = f["data"][..., 0:3].astype(np.uint8)
                    mask = f["data"][..., -1].astype(np.uint8)
                    Image.fromarray(img).save(path / "true.png")
                    Image.fromarray(mask).save(path / "mask.png")
                    h, w, _ = img.shape
                    new_h = h + cfg.loader.image_size
                    new_w = w + cfg.loader.image_size
                    input = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                    overlay = cfg.loader.image_size // 4
                    input[overlay : overlay + h, overlay : overlay + w, 0:3] = img
                    tiles = image_cutting(input, cfg.loader.image_size)
                    tiles = np.transpose(tiles, (0, 3, 1, 2))
                    tiles = torch.from_numpy(tiles).to(model.device)

                    mean = torch.tensor([0.485, 0.456, 0.406], device=model.device).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=model.device).view(1, 3, 1, 1)
                    tiles = (tiles - mean) / std

                    with torch.no_grad():
                        logits = []  # noqa: WPS220
                        for tile in tiles:  # noqa: WPS220
                            log = model(tile.unsqueeze(0))  # noqa: WPS220
                            logits.append(log.cpu().numpy())  # noqa: WPS220

                    out = make_mask(input.shape, logits, cfg.loader.image_size, cfg.model.num_classes)

                    idx = 1
                    for obj_class in cfg.classes:
                        out[out == idx] = obj_class[1]  # noqa: WPS220
                        idx += 1
                    overlay = cfg.loader.image_size // 4
                    out = out[overlay : overlay + h, overlay : overlay + w]
                    mask_image = Image.fromarray(out)
                    mask_image.save(path / "pred.png")


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg.training
    main(cfg)
