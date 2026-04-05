import rootutils
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.loader import Loader
from src.model import Model
from src.utils import make_dir

MASK_KEY = "mask"

NUMBER_OF_PIXELS_SUFFIX = "_number_of_pixels"


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
