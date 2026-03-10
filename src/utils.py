import shutil
from pathlib import Path

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

import numpy as np

MAX_UINT32 = 4294967295
HEX_BASE = 16


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
