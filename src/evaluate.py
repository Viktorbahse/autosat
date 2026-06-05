import numpy as np
import rootutils
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
import json
from pathlib import Path

import h5py
import torch

from src.model import Model
from src.utils import make_dir

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


def restoring_class_brightness(image, classes):
    img = image.astype(np.uint8)
    for i, object_class in zip(range(1, len(classes) + 1), classes):
        img[img == i] = object_class[1]
    return img


def get_coords(h, w, image_size, overlap=0.5):
    step = image_size - int(overlap * image_size)
    coords = []
    y = 0
    while y + image_size <= h:
        x = 0
        while x + image_size <= w:
            coords.append((x, x + image_size, y, y + image_size))
            x += step
        y += step
    return coords


def get_probs(images, model, device, batch_size):  # noqa: WPS210
    model.eval()
    model.to(device)

    logits_list = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            batch_tensor = torch.stack([transform(img) for img in batch]).float().to(device)
            logits = model(batch_tensor)
            logits = logits.permute(0, 2, 3, 1).cpu().numpy()
            for j, _ in enumerate(batch):
                logits_list.append(logits[j])

    return logits_list


def predict(image, model, device, batch_size, image_size, shape):  # noqa: WPS210 WPS211
    h, w, c = shape
    prediction = np.zeros((h, w, c))
    coords = get_coords(h, w, image_size)

    tiles = []
    for coord in coords:
        x1, x2, y1, y2 = coord
        tiles.append(image[y1:y2, x1:x2, slice(None)])

    logits_list = get_probs(tiles, model, device, batch_size)

    for coord, logit in zip(coords, logits_list):
        x1, x2, y1, y2 = coord
        prediction[y1:y2, x1:x2, slice(None)] += logit

    return prediction


def predict_by_images(images, model, device, batch_size, image_size, number_of_classes):  # noqa: WPS211
    h, w, c = images[0].shape[0], images[0].shape[1], number_of_classes
    prediction = sum(predict(img, model, device, batch_size, image_size, (h, w, c)) for img in images)
    return prediction


def main(cfg: DictConfig):  # noqa: WPS210 WPS231
    make_dir(cfg.data_out, delete_if_exist=True)

    checkpoint_path = Path(cfg.checkpoint_path)

    if checkpoint_path.is_file():
        checkpoint_path = Path(checkpoint_path)
    elif checkpoint_path.is_dir():
        ckpt_files = list(checkpoint_path.glob("*.ckpt"))
        if ckpt_files:
            checkpoint_path = Path(ckpt_files[0])
        else:
            raise FileNotFoundError(f"Веса модели не найдены в директории: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Путь не существует или не является файлом/директорией: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model.load_from_checkpoint(checkpoint_path=ckpt_files[0], cfg=cfg.model, weights=None)
    model.to(device)

    with open(cfg.data_in, "r") as f:
        loaded_split_info = json.load(f)

    test_files_paths = [Path(p) for p in loaded_split_info["test_files"]]

    for h5_file in tqdm(test_files_paths, desc="Обработка файлов"):
        output_path = Path(cfg.data_out)
        with h5py.File(h5_file, "r") as f:
            output_path = output_path / str(f.attrs["x"]) / str(f.attrs["y"])
            make_dir(output_path, delete_if_exist=True)

            img = Image.fromarray(f["data"][..., -1], "L")
            img.save(output_path / "mask.jpg")

            channels_count = f["data"].shape[2]
            step = 3
            indices = range(0, channels_count - 1, step)
            images = [f["data"][..., i : i + step] for i in indices]
            pred = predict_by_images(images, model, device, cfg.batch_size, cfg.image_size, len(cfg.classes) + 1)
            pred = np.argmax(pred, axis=-1)
            mask = restoring_class_brightness(pred, cfg.classes)
            img = Image.fromarray(mask, "L")
            img.save(output_path / "prediction.jpg")


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg.evaluate
    main(cfg)
