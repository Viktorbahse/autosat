from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, cast

import h5py
import numpy as np
import rootutils
from omegaconf import DictConfig, OmegaConf
from PIL import Image

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


def merge_layers(layer_dict: Dict[str, List[Path]], cfg, bounds: Tuple[int,int,int,int]) -> np.ndarray:
    x1, y1, x2, y2 = bounds
    n_tiles_x = x2 - x1 + cfg.num_tiles
    n_tiles_y = y2 - y1 + cfg.num_tiles

    rgb_layers = [k for k in layer_dict.keys() if k != 'mask']
    channels = 3 * len(rgb_layers) + (1 if 'mask' in layer_dict else 0)
    H = n_tiles_y * cfg.tile_size
    W = n_tiles_x * cfg.tile_size
    data = np.empty((H, W, channels), dtype=np.uint8)

    def process_rgb_layer(index: int, layer_name: str):
        base_channel = 3 * index
        for f in layer_dict[layer_name]:
            p = Path(f)
            tx = int(p.parent.parent.name) - x1
            ty = int(p.parent.name) - y1
            y0 = ty * cfg.tile_size
            y1p = y0 + cfg.tile_size * cfg.num_tiles
            x0 = tx * cfg.tile_size
            x1p = x0 + cfg.tile_size * cfg.num_tiles
            img = Image.open(p).convert("RGB")
            arr = np.array(img, dtype=np.uint8)
            data[y0:y1p, x0:x1p, base_channel:base_channel+3] = arr

    max_workers = min(cfg.max_workers, max(1, len(rgb_layers)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for i, l in enumerate(rgb_layers):
            futures.append(ex.submit(process_rgb_layer, i, l))
        for fut in as_completed(futures):
            fut.result()

    if 'mask' in layer_dict:
        for f in layer_dict['mask']:
            p = Path(f)
            tx = int(p.parent.parent.name) - x1
            ty = int(p.parent.name) - y1
            y0 = ty * cfg.tile_size
            y1p = y0 + cfg.tile_size * cfg.num_tiles
            x0 = tx * cfg.tile_size
            x1p = x0 + cfg.tile_size * cfg.num_tiles
            img = Image.open(p).convert("L")
            arr = np.array(img, dtype=np.uint8)
            data[y0:y1p, x0:x1p, 3 * len(rgb_layers)] = arr

    return data

def main(cfg: DictConfig):
    sat_dir = Path(cfg.data_dir) / f"{cfg.zoom}"
    files = get_files(sat_dir)

    layer_dict: Dict[str, List[Path]] = defaultdict(list)

    for f in files:
        key = f.stem
        layer_dict[key].append(f)

    x_min = min(int(f.parent.parent.name) for f in files)
    x_max = max(int(f.parent.parent.name) for f in files)

    y_min = min(int(f.parent.name) for f in files)
    y_max = max(int(f.parent.name) for f in files)
    hdf_arr = merge_layers(layer_dict, cfg, (x_min, y_min, x_max, y_max))

    hdf_path = make_dir(Path(cfg.dataset_dir) / f"{cfg.zoom}", delete_if_exist=True)
    with h5py.File(hdf_path / f"{cfg.hdf}", "w") as h5f:
        dset = h5f.create_dataset('layers',
                                shape=hdf_arr.shape,
                                dtype='uint8',
                                chunks=(cfg.chunk_h, cfg.chunk_w, cfg.chunk_c),
                                compression=None)
        dset[...] = hdf_arr
        dset.attrs['zoom'] = int(cfg.zoom)
        dset.attrs['x'] = int(x_min)
        dset.attrs['y'] = int(y_min)


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg["prepare_satellite_data"]

    main(cfg)
