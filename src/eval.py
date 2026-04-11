import numpy as np
import rootutils
from omegaconf import DictConfig, OmegaConf
from PIL import Image

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
from pathlib import Path

import torch

from src.model import Model
from src.utils import make_dir


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


def make_mask(shape, logits, image_size, num_classes):
    mask = np.zeros((num_classes, shape[0], shape[1]), dtype=np.float32)
    i = 0
    j = 0
    t = 0
    while i + image_size <= shape[0]:
        while j + image_size <= shape[1]:
            mask[0:-1, i : i + image_size, j : j + image_size] += logits[t][0]  # noqa: WPS221
            t += 1
            j += image_size // 4 * 3
        j = 0
        i += image_size // 4 * 3

    return mask.argmax(axis=0).astype(np.uint8)


def main(cfg: DictConfig):  # noqa: WPS210
    make_dir(cfg.data_out, delete_if_exist=False)

    model = Model.load_from_checkpoint(checkpoint_path=cfg.checkpoint_path, cfg=cfg.model, weights=None)

    model.eval()
    model.to("cuda" if cfg.model.cuda else "cpu")

    input_path = Path(cfg.data_in)

    output_dir = Path(cfg.data_out)
    output_path = output_dir / "predict.png"
    img = Image.open(input_path).convert("RGB")
    img.save(output_dir / "true.png")
    img = np.array(img)
    h, w, _ = img.shape
    new_h = h + cfg.image_size
    new_w = w + cfg.image_size
    input = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    overlay = cfg.image_size // 4
    input[overlay : overlay + h, overlay : overlay + w, 0:3] = img
    tiles = image_cutting(input, cfg.image_size)
    tiles = np.transpose(tiles, (0, 3, 1, 2))
    tiles = torch.from_numpy(tiles).to(model.device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=model.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=model.device).view(1, 3, 1, 1)
    tiles = (tiles - mean) / std

    with torch.no_grad():
        logits = []
        for tile in tiles:
            log = model(tile.unsqueeze(0))  # добавляем batch dimension
            logits.append(log.cpu().numpy())

    out = make_mask(input.shape, logits, cfg.image_size, cfg.model.num_classes)

    idx = 1
    for obj_class in cfg.classes:
        out[out == idx] = obj_class[1]
    overlay = cfg.image_size // 4
    out = out[overlay : overlay + h, overlay : overlay + w]
    mask_image = Image.fromarray(out)
    mask_image.save(output_path)


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg.eval
    main(cfg)
