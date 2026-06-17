import rootutils
from dvclive.lightning import DVCLiveLogger
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.loader import Loader
from src.model import Model
from src.utils import make_dir

MASK_KEY = "mask"

NUMBER_OF_PIXELS_SUFFIX = "_number_of_pixels"


def main(cfg: DictConfig):  # noqa: WPS210 WPS213 WPS231
    make_dir(cfg.logs_dir, delete_if_exist=False)

    loader = Loader(cfg.loader)
    loader.prepare_data()

    model = Model(cfg.model, weights=loader.weights)

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(save_top_k=1, monitor="val/metric", mode="max", every_n_epochs=1),
    ]

    logger = DVCLiveLogger(run_name="model_training", log_model=True, dir=f"{cfg.logs_dir}/dvclive")

    trainer = Trainer(
        max_epochs=cfg.num_epoch,
        default_root_dir=f"{cfg.logs_dir}",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
    )

    trainer.fit(model, loader)

    trainer.test(model, loader)


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg.training
    main(cfg)
