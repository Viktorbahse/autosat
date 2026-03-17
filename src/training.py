import logging
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.unet import UNet
from src.utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_files(data_dir: Path):
    if not data_dir.exists():
        return []

    files = list(data_dir.glob("*.h5"))

    return files


def _process_file(file_h5: str, obj_class: str) -> float:
    with h5py.File(Path(file_h5), "r") as f:
        intensity = f.attrs[obj_class]
        last = f["/data"][...][..., -1]
        total = last.size
        return np.sum(last == intensity) / total if total else 0


def compyting_class_balanse(obj_class: str, files: list[str], max_workers: int = 8) -> float:
    fractions = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_file, file_h5, obj_class): file_h5 for file_h5 in files}
        for fut in as_completed(futures):
            fractions.append(fut.result())
    return sum(fractions) / len(fractions) if fractions else 0


def split(files, train_ratio, val_ratio, test_ratio, seed):
    rnd = random.Random(seed)
    rnd.shuffle(files)
    n = len(files)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    n_test = n - n_train - n_val

    if any(x < 0 for x in (n_train, n_val, n_test)):
        raise ValueError("Пропорции train, val и test должны быть в диапазоне [0, 1], их сумма должна быть 1.")

    return files[:n_train], files[n_train : n_train + n_val], files[n_train + n_val :]


def main(cfg: DictConfig):  # noqa: WPS210

    device = torch.device("cuda" if cfg.model.common.cuda else "cpu")
    if cfg.model.common.cuda and not torch.cuda.is_available():
        sys.exit("Error: CUDA requested but not available")
    # checkpoints_dir = make_dir(cfg.model.common.checkpoints_dir, delete_if_exist=True)

    num_classes = 2

    net = UNet(num_classes)
    net = net.to(device)

    if cfg.model.common.cuda:
        torch.backends.cudnn.benchmark = True

    set_seed(cfg.random_seed)
    dataset_dir = Path(cfg.dataset.dataset_dir)
    train_ratio, val_ratio, test_ratio = (
        cfg.dataset.dataset_split.train,
        cfg.dataset.dataset_split.val,
        cfg.dataset.dataset_split.test,
    )

    files = get_files(dataset_dir)

    train_files, val_files, test_files = split(files, train_ratio, val_ratio, test_ratio, cfg.random_seed)

    logger.info(f"Доли — train: {train_ratio}, val: {val_ratio}, test: {test_ratio}.")
    logger.info("Количество файлов — train: %d, val: %d, test: %d.", len(train_files), len(val_files), len(test_files))

    if cfg.dataset.compute_class_balance:
        class_fraction = compyting_class_balanse(
            cfg.dataset.obj_class, train_files + val_files, max_workers=cfg.max_workers
        )
    else:
        class_fraction = 0.5
    logger.info(f"Доля {cfg.dataset.obj_class} в данных для обучения модели: {class_fraction}.")


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg.training
    main(cfg)
