import numpy as np
import rootutils
from omegaconf import DictConfig, OmegaConf
from PIL import Image

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
from pathlib import Path

import torch

from src.model import Model
from src.utils import image_cutting, make_dir, make_mask


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
        idx += 1
    overlay = cfg.image_size // 4
    out = out[overlay : overlay + h, overlay : overlay + w]
    mask_image = Image.fromarray(out)
    mask_image.save(output_path)


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg.eval
    main(cfg)
