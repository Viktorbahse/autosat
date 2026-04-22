import shutil
from pathlib import Path

import numpy as np
import torch


def make_dir(path: str | Path, delete_if_exist: bool) -> Path:
    path = Path(path)

    if delete_if_exist and path.is_dir():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)

    return path


import hashlib
import os
import random

MAX_UINT32 = 4294967295
HEX_BASE = 16


def make_mask(shape, logits, image_size, num_classes):
    mask = np.zeros((num_classes, shape[0], shape[1]), dtype=np.float32)
    i = 0
    j = 0
    t = 0
    while i + image_size <= shape[0]:
        while j + image_size <= shape[1]:
            mask[..., i : i + image_size, j : j + image_size] += logits[t][0]  # noqa: WPS221
            t += 1
            j += image_size // 4 * 3
        j = 0
        i += image_size // 4 * 3

    return mask.argmax(axis=0).astype(np.uint8)


def image_cutting(image, tile_size):
    tiles = []
    i = 0
    j = 0
    while i + tile_size <= image.shape[0]:
        while j + tile_size <= image.shape[1]:
            tiles.append(image[i : i + tile_size, j : j + tile_size, 0:3])
            j += tile_size // 4 * 3
        j = 0
        i += tile_size // 4 * 3
    return np.array(tiles, dtype=np.float32) / 255.0  # noqa: WPS 432


def _seed_from_int(seed: int) -> int:
    return int(hashlib.sha256(str(seed).encode()).hexdigest(), HEX_BASE) % MAX_UINT32


def set_seed(seed: int):
    seed32 = _seed_from_int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed32)
    random.seed(seed)
    np.random.seed(seed32)

    torch.manual_seed(seed)
    torch.manual_seed(seed32)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed32)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
