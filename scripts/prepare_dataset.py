import json
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple

import h5py
import numpy as np
import rootutils
from omegaconf import DictConfig, OmegaConf
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

root = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.utils import make_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

h5_write_lock = Lock()


def get_files(root: Path) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for fname in filenames:
            files.append(Path(dirpath) / fname)
    return files


def safe_open_image(path: Path):
    try:
        if path.stat().st_size == 0:
            return None
        with Image.open(path) as img:
            return np.array(img)
    except (UnidentifiedImageError, OSError):
        return None


def process_layer(item: Tuple[Tuple[int, int], List[Path]], cfg: DictConfig, dataset_path: Path, task_index: int, max_len: int):
    (x, y), files = item
    if not files:
        return (x, y, False)

    num_image_files = len([f for f in files if f.stem != "mask"])
    channels = num_image_files * 3 + (1 if any(f.stem == "mask" for f in files) else 0)
    height = cfg.num_tiles * cfg.tile_size
    width = cfg.num_tiles * cfg.tile_size

    data = np.zeros((height, width, channels), dtype=np.uint8)

    index = 0
    for f in files:
        arr = safe_open_image(f)
        if arr is None:
            if f.stem == "mask":
                data[..., -1] = 0
            else:
                data[..., index:index + 3] = 0
                index += 3
            continue

        if f.stem == "mask":
            if arr.ndim == 3:
                arr = np.array(Image.fromarray(arr).convert("L"), dtype=np.uint8)
            else:
                arr = arr.astype(np.uint8)
            if arr.shape != (height, width):
                temp = np.zeros((height, width), dtype=np.uint8)
                h0 = min(arr.shape[0], height)
                w0 = min(arr.shape[1], width)
                temp[0:h0, 0:w0] = arr[0:h0, 0:w0]
                arr = temp
            data[..., -1] = arr
        else:
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[2] == 4:
                arr = arr[..., :3]
            elif arr.shape[2] != 3:
                arr = arr[..., :3] if arr.shape[2] > 3 else np.pad(arr, ((0, 0), (0, 0), (0, 3 - arr.shape[2])), mode="constant")
            if arr.shape != (height, width, 3):
                h0 = min(arr.shape[0], height)
                w0 = min(arr.shape[1], width)
                data[..., index:index + 3] = 0
                data[0:h0, 0:w0, index:index + 3] = arr[0:h0, 0:w0, 0:3]
            else:
                data[..., index:index + 3] = arr
            index += 3

    filename = f"{str(task_index).zfill(len(str(max_len - 1)))}.h5"
    out_path = dataset_path / filename
    try:
        classes_json = json.dumps(OmegaConf.to_container(cfg.classes, resolve=True))
    except Exception:
        classes_json = json.dumps(list(cfg.classes) if hasattr(cfg.classes, "__iter__") else str(cfg.classes))

    with h5_write_lock:
        with h5py.File(out_path, "w") as h5f:
            dset = h5f.create_dataset('item', shape=data.shape, dtype='uint8', compression=None)
            dset[...] = data
            dset.attrs['zoom'] = int(cfg.zoom)
            dset.attrs['x'] = int(x)
            dset.attrs['y'] = int(y)
            dset.attrs['classes'] = classes_json
            time.sleep(0.01)

    return (x, y, True)


def main(cfg: DictConfig):
    sat_dir = Path(cfg.data_dir) / f"{cfg.zoom}"
    logger.info("Collecting files from %s", sat_dir)
    files = get_files(sat_dir)
    dataset_path = make_dir(Path(cfg.dataset_dir) / f"{cfg.zoom}", delete_if_exist=True)

    tiles_dict: Dict[Tuple[int, int], List[Path]] = defaultdict(list)

    for f in files:
        parent = f.parent
        grand = parent.parent
        x = int(grand.name)
        y = int(parent.name)
        tiles_dict[(x, y)].append(f)

    items = list(tiles_dict.items())
    total = len(items)
    logger.info("Found %d tiles to process", total)
    if total == 0:
        return

    max_workers = int(cfg.max_workers) if getattr(cfg, "max_workers", None) else min(8, os.cpu_count() or 1)
    logger.info("Using max_workers=%d", max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_layer, item, cfg, dataset_path, idx, len(items)): item[0] for idx, item in enumerate(items)}
        pbar = tqdm(total=total, desc="Processing files", unit="file")
        pending = set(futures.keys())
        try:
            while pending:
                done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                if not done:
                    logger.info("%d tasks still pending...", len(pending))
                for fut in list(done):
                    try:
                        x, y, ok = fut.result()
                        if not ok:
                            logger.info("Tile (%s,%s) processed with errors", x, y)
                    except Exception as e:
                        logger.error("Task raised exception: %s", e)
                    pbar.update(1)
        finally:
            pbar.close()


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg["prepare_satellite_data"]
    main(cfg)
