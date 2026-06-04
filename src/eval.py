from itertools import product

import numpy as np
import rootutils
from omegaconf import DictConfig, OmegaConf
from PIL import Image

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
import glob
import json
from pathlib import Path

import h5py
import torch

from src.model import Model
from src.utils import make_dir

DATA_KEY = "data"


def get_tile_coords(h, w, image_size, overlap=0.5):
    step = int(image_size * (1 - overlap))
    coords = []
    y = 0
    while y + image_size <= h:
        x = 0
        while x + image_size <= w:
            coords.append((x, x + image_size, y, y + image_size))
            x += step
        y += step

    return np.array(coords)


def main(cfg: DictConfig):  # noqa: WPS210
    make_dir(cfg.data_out, delete_if_exist=False)

    ckpt_files = glob.glob(f"{cfg.checkpoint_path}/*.ckpt")
    model = Model.load_from_checkpoint(checkpoint_path=ckpt_files[0], cfg=cfg.model, weights=None)

    model.eval()
    model.to("cuda" if cfg.model.cuda else "cpu")

    with open(cfg.data_in, "r") as f:
        loaded_split_info = json.load(f)

    test_files_paths = [Path(p) for p in loaded_split_info["test_files"]]

    for number, p in enumerate(test_files_paths):
        output_path = Path(cfg.data_out)
        with h5py.File(p, "r") as f:
            make_dir(output_path / f"{number}", delete_if_exist=False)

            img = Image.fromarray(f[DATA_KEY][..., 0:3], "RGB")
            img.save(output_path / f"{number}" / "satellite.jpg")

            img = Image.fromarray(f[DATA_KEY][..., -1], "L")
            img.save(output_path / f"{number}" / "mask.jpg")

            h, w, c = f[DATA_KEY].shape
            images_array_with_padding = np.zeros((h + cfg.image_size, w + cfg.image_size, c - 1))

            step = cfg.image_size // 2

            x1, x2 = step, w + step
            y1, y2 = step, h + step

            images_array = f[DATA_KEY][..., 0:-1]
            images_array_with_padding[y1:y2, x1:x2, slice(None)] = images_array
            segmentation_probability_array = np.zeros((h + cfg.image_size, w + cfg.image_size, 3))
            coords = get_tile_coords(h + cfg.image_size, w + cfg.image_size, cfg.image_size)

            for i, coord in product(range(0, c - 1, 3), coords):
                x1, x2, y1, y2 = coord
                img = images_array_with_padding[x1:x2, y1:y2, i : i + 3]

                img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(model.device)
                mean = torch.tensor([0.485, 0.456, 0.406], device=model.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=model.device).view(1, 3, 1, 1)
                img_tensor = (img_tensor - mean) / std

                with torch.no_grad():
                    logits = model(img_tensor)
                    probabilities = torch.softmax(logits, dim=1)
                    predictions = probabilities.cpu().numpy()
                    predictions_np = predictions.squeeze(0)
                    predictions_np = np.transpose(predictions_np, (1, 2, 0))
                    segmentation_probability_array[x1:x2, y1:y2, slice(None)] += predictions_np

            out = segmentation_probability_array.argmax(axis=2).astype(np.uint8)
            idx = 1
            for obj_class in cfg.classes:
                out[out == idx] = obj_class[1]
                idx += 1
            x1, x2 = step, w + step
            y1, y2 = step, h + step
            mask_image = Image.fromarray(out[y1:y2, x1:x2])
            mask_image.save(output_path / f"{number}" / "predictions.jpg")


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg.eval
    main(cfg)
