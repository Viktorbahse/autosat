import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, cast

import h5py
import numpy as np
import rootutils
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

root = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.utils import make_dir


def get_files(root: Path) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    if root.is_file():
        return [root]

    files: List[Path] = []
    for p in root.rglob('*'):
        if p.is_file():
            files.append(p)
    return files


def process_layer(item: Tuple[Tuple[int, int], List[Path]], cfg: DictConfig, dataset_path: Path):
    (x, y), files = item
    if not files:
        return (x, y, False)
    data = np.empty((cfg.num_tiles * cfg.tile_size, cfg.num_tiles * cfg.tile_size, len(files) * 3 - 2), dtype=np.uint8)
    index = 0
    for f in files:
        if f.stem == 'mask':
            with Image.open(f) as img:
                img = img.convert("L")
                arr = np.array(img, dtype=np.uint8)
            data[..., -1] = arr
        else:
            with Image.open(f) as img:
                img = img.convert("RGB")
                arr = np.array(img, dtype=np.uint8)
            if arr.shape != (cfg.num_tiles * cfg.tile_size, cfg.num_tiles * cfg.tile_size, 3):
                data[..., index:index + 3] = np.zeros((cfg.num_tiles * cfg.tile_size, cfg.num_tiles * cfg.tile_size, 3), dtype=np.uint8)
                data[0:arr.shape[0], 0:arr.shape[1], index:index + 3] = arr
            else:
                data[..., index:index + 3] = arr
            index += 3

    out_dir = make_dir(dataset_path / f"{str(x).zfill(5)}" / f"{str(y).zfill(5)}", delete_if_exist=False)
    with h5py.File(out_dir / f"item.hdf", "w") as h5f:
        dset = h5f.create_dataset('item',
                                  shape=data.shape,
                                  dtype='uint8',
                                  compression=None)
        dset[...] = data
        dset.attrs['zoom'] = int(cfg.zoom)
        dset.attrs['x'] = int(x)
        dset.attrs['y'] = int(y)
        dset.attrs['classes'] = json.dumps([['road', 204], ['building', 255], ['parking', 153], ['water', 102], ['field', 51]])
    return (x, y, True)


def main(cfg: DictConfig):
    sat_dir = Path(cfg.data_dir) / f"{cfg.zoom}"
    files = get_files(sat_dir)
    dataset_path = make_dir(Path(cfg.dataset_dir) / f"{cfg.zoom}", delete_if_exist=True)

    tiles_dict: Dict[Tuple[int, int], List[Path]] = defaultdict(list)

    for f in files:
        x = int(f.parent.parent.name)
        y = int(f.parent.name)
        tiles_dict[(x, y)].append(f)

    items = list(tiles_dict.items())
    total = len(items)
    if total == 0:
        return

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        futures = {ex.submit(process_layer, item, cfg, dataset_path): item[0] for item in items}
        with tqdm(total=total, desc="Processing files", unit="file") as pbar:
            for fut in as_completed(futures):
                try:
                    x, y, ok = fut.result()
                except Exception as e:
                    ok = False
                pbar.update(1)


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg["prepare_satellite_data"]

    main(cfg)
