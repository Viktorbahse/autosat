import collections
import io
import json
import math
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import httpx
import mercantile
import numpy as np
import rootutils
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import transform
from supermercado import burntiles

Image.MAX_IMAGE_PIXELS = None
SESSION = httpx.Client()


root = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from scripts.handlers import BuildingHandler, RoadHandler

from src.utils import make_dir


def resolution_to_zoom_level(resolution):
    """
    Convert map resolution in meters to zoom level for Web Mercator (EPSG:3857) tiles.
    """
    # Web Mercator tile size in meters at zoom level 0
    initial_resolution = 156543.03392804097

    # Calculate the zoom level
    zoom_level = math.log2(initial_resolution / resolution)

    return int(zoom_level)

def tile_pixel_to_lonlat(x: int, y: int, z: int, px: int, py: int, tile_size: int = 256):
    """
    Convert tile x,y at zoom z and pixel coordinates px,py (0..tile_size-1)
    to geographic coordinates (lon, lat) in degrees.

    Returns (lon, lat).
    """
    n = 2 ** z
    gx = (x * tile_size) + px
    gy = (y * tile_size) + py
    world = tile_size * n
    lon = (gx / world) * 360.0 - 180.0
    fy = gy / world
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * fy)))
    lat = math.degrees(lat_rad)

    return lon, lat

def yandex_tile_pixel_to_lon_lat(x: int, y: int, z: int, px: int, py: int) -> tuple:
    """
    Преобразует координаты тайла Яндекс.Карт и пикселя внутри него в географические координаты (долгота, широта).

    Parameters:
    -----------
    x, y : int
        Номера тайла по горизонтали и вертикали
    z : int
        Уровень масштабирования (0-17 для Яндекс.Карт)
    px, py : int
        Координаты пикселя внутри тайла (0-255)

    Returns:
    --------
    tuple : (longitude, latitude) в градусах
    """
    # Константы
    pi = math.pi
    tile_size = 256

    # Яндекс.Карты используют проекцию WGS84 Mercator с эксцентриситетом 0.0818191908426
    e = 0.0818191908426

    # 1. Вычисляем глобальные пиксельные координаты
    x_p = x * tile_size + px
    y_p = y * tile_size + py

    # 2. Вычисляем rho (половина размера карты в пикселях)
    rho = math.pow(2, z + 8) / 2

    # 3. Вычисляем долготу (обратная операция к x = rho * (1 + long/180))
    longitude = (x_p / rho - 1) * 180

    # 4. Вычисляем theta из y координаты
    # y_p = rho * (1 - ln(theta)/pi)
    # ln(theta)/pi = 1 - y_p/rho
    # ln(theta) = pi * (1 - y_p/rho)
    theta = math.exp(pi * (1 - y_p / rho))

    # 5. Решаем уравнение для широты
    # theta = tan(pi/4 + beta/2) * ((1 - e*sin(beta))/(1 + e*sin(beta)))^(e/2)
    # где beta - изометрическая широта в радианах

    # Начальное приближение (сферическая Земля)
    beta = 2 * math.atan(theta) - pi / 2

    # Итеративное уточнение для эллипсоида (метод Ньютона)
    # Обычно достаточно 10 итераций
    for _ in range(10):
        # Вычисляем текущее значение theta
        sin_beta = math.sin(beta)
        phi = (1 - e * sin_beta) / (1 + e * sin_beta)
        theta_calc = math.tan(pi / 4 + beta / 2) * math.pow(phi, e / 2)

        # Вычисляем производную
        f_prime = theta_calc * (
            1 / (2 * math.cos(pi / 4 + beta / 2)**2) / math.tan(pi / 4 + beta / 2) +
            e * math.cos(beta) / (1 - (e * sin_beta)**2)
        )

        # Обновляем beta методом Ньютона
        delta = (theta_calc - theta) / f_prime
        beta -= delta

        # Проверка на сходимость
        if abs(delta) < 1e-12:
            break

    # Преобразуем радианы в градусы
    latitude = beta * 180 / pi

    return longitude, latitude

def lat_lon_to_yandex_tile(lat, lon, z):
    e = 0.0818191908426
    r = 2**(int(z) + 8) / 2
    b = lat * math.pi / 180
    p = (1 - e * math.sin(b)) / (1 + e * math.sin(b))
    t = math.tan(math.pi/4 + b/2) * p**(e/2)
    x = r * (1 + lon / 180)
    y = r * (1 - math.log(t) / math.pi)
    return [int(x/256), int(y/256)]

def deg2num(lon, lat, zoom):
    lat_r = math.radians(lat)
    n = 2**zoom
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n
    return x, y


def get_xy_corners(map_box: tuple[float, ...], zoom: int) -> tuple[int, ...]:
    x1, y1 = deg2num(map_box[0], map_box[1], zoom)
    x2, y2 = deg2num(map_box[2], map_box[3], zoom)

    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    x1, y1 = map(math.floor, (x1, y1))
    x2, y2 = map(math.ceil, (x2, y2))

    return x1, y1, x2, y2


def get_tile(url):
    retry = 10

    while 1:
        try:
            r = SESSION.get(url, timeout=60)
            break
        except Exception:
            retry -= 1
            if not retry:
                raise

    if r.status_code == 404:
        return None
    elif not r.content:
        return None

    r.raise_for_status()

    return r.content


def is_empty(im):
    extrema = im.getextrema()

    if len(extrema) >= 3:
        if len(extrema) > 3 and extrema[-1] == (0, 0):
            return True

        for ext in extrema[:3]:
            if ext != (0, 0):
                return False

        return True
    else:
        return extrema[0] == (0, 0)


def feature_to_mercator(feature):
    """Normalize feature and converts coords to 3857.

    Args:
      feature: geojson feature to convert to mercator geometry.
    """
    # Ref: https://gist.github.com/dnomadb/5cbc116aacc352c7126e779c29ab7abe

    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(3857)

    geometry = feature["geometry"]
    if geometry["type"] == "Polygon":
        xys = (zip(*part) for part in geometry["coordinates"])
        xys = (list(zip(*transform(src_crs, dst_crs, *xy))) for xy in xys)

        yield {"coordinates": list(xys), "type": "Polygon"}

    elif geometry["type"] == "MultiPolygon":
        for component in geometry["coordinates"]:
            xys = (zip(*part) for part in component)
            xys = (list(zip(*transform(src_crs, dst_crs, *xy))) for xy in xys)

            yield {"coordinates": list(xys), "type": "Polygon"}


def burn(tile, features, size):
    """Burn tile with features.

    Args:
      tile: the mercantile tile to burn.
      features: the geojson features to burn.
      size: the size of burned image.

    Returns:
      image: rasterized file of size with features burned.
    """

    # the value you want in the output raster where a shape exists
    burnval = 1
    shapes = ((geometry, burnval) for feature in features for geometry in feature_to_mercator(feature))

    bounds = mercantile.xy_bounds(tile)
    transform = from_bounds(*bounds, size, size)

    return rasterize(shapes, out_shape=(size, size), transform=transform)

def paste_tile_yandex(bigim, tile, corner_xy, bbox, z):
    if tile is None:
        return bigim

    im = Image.open(io.BytesIO(tile))
    mode = "RGB" if im.mode == "RGB" else "RGBA"
    size = im.size


    if bigim is None:
        newim = Image.new(mode, (size[0] * (bbox[2] - bbox[0]), size[1] * (bbox[3] - bbox[1])))
    else:
        newim = bigim


    big_lon_min, big_lat_max = tile_pixel_to_lonlat(bbox[0], bbox[1], z, 0, 0, size[0])
    big_lon_max, big_lat_min = tile_pixel_to_lonlat(bbox[2]-1, bbox[3]-1, z, size[0], size[0], size[0])


    tile_lon_min, tile_lat_max = yandex_tile_pixel_to_lon_lat(corner_xy[0], corner_xy[1], z, 0, 0)
    tile_lon_max, tile_lat_min = yandex_tile_pixel_to_lon_lat(corner_xy[0], corner_xy[1], z, size[0], size[0])

    lon_per_pixel = (big_lon_max - big_lon_min) / newim.size[0]
    lat_per_pixel = (big_lat_min - big_lat_max) / newim.size[1]

    paste_x = int((tile_lon_min - big_lon_min) / lon_per_pixel)
    paste_y = int((tile_lat_max - big_lat_max) / lat_per_pixel)

    if (paste_x + size[0] < 0 or paste_x >= newim.size[0] or
        paste_y + size[1] < 0 or paste_y >= newim.size[1]):
        im.close()
        return newim

    if paste_x < 0 or paste_y < 0 or paste_x + size[0] > newim.size[0] or paste_y + size[1] > newim.size[1]:
        crop_x = max(0, -paste_x)
        crop_y = max(0, -paste_y)
        crop_w = min(size[0], newim.size[0] - paste_x) - crop_x
        crop_h = min(size[1], newim.size[1] - paste_y) - crop_y

        if crop_w <= 0 or crop_h <= 0:
            im.close()
            return newim

        im_cropped = im.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))

        paste_x = max(0, paste_x)
        paste_y = max(0, paste_y)

        if mode == "RGB":
            newim.paste(im_cropped, (paste_x, paste_y))
        else:
            if im_cropped.mode != mode:
                im_cropped = im_cropped.convert(mode)
            if not is_empty(im_cropped):
                newim.paste(im_cropped, (paste_x, paste_y))
    else:
        if mode == "RGB":
            newim.paste(im, (paste_x, paste_y))
        else:
            if im.mode != mode:
                im = im.convert(mode)
            if not is_empty(im):
                newim.paste(im, (paste_x, paste_y))

    im.close()
    return newim

def paste_tile(bigim, tile, corner_xy, bbox):
    if tile is None:
        return bigim

    im = Image.open(io.BytesIO(tile))
    mode = "RGB" if im.mode == "RGB" else "RGBA"
    size = im.size

    if bigim is None:
        newim = Image.new(mode, (size[0] * (bbox[2] - bbox[0]), size[1] * (bbox[3] - bbox[1])))
    else:
        newim = bigim

    dx = abs(corner_xy[0] - bbox[0])
    dy = abs(corner_xy[1] - bbox[1])

    xy0 = (size[0] * dx, size[1] * dy)

    if mode == "RGB":
        newim.paste(im, xy0)
    else:
        if im.mode != mode:
            im = im.convert(mode)

        if not is_empty(im):
            newim.paste(im, xy0)

    im.close()

    return newim


def paste_mask(bigmsk, tile_size, feature_map, xy, bbox, zoom):
    tile = mercantile.Tile(int(xy[0]), int(xy[1]), int(zoom))

    if tile in feature_map:
        msk = burn(tile, feature_map[tile], tile_size)
        msk = (255 * msk).astype(np.uint8)
    else:
        msk = np.zeros((tile_size, tile_size), dtype=np.uint8)

    if bigmsk is None:
        newmsk = np.zeros((tile_size * (bbox[2] - bbox[0]), tile_size * (bbox[3] - bbox[1])), dtype=np.uint8)
    else:
        newmsk = bigmsk

    dx = abs(xy[0] - bbox[0])
    dy = abs(xy[1] - bbox[1])

    x1 = tile_size * dx
    y1 = tile_size * dy

    x2 = x1 + tile_size
    y2 = y1 + tile_size

    newmsk[y1:y2, x1:x2] = msk

    return newmsk


def xyz_to_quadkey(x, y, z):
    x, y, z = int(x), int(y), int(z)
    quadkey = ""
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (x & mask) != 0:
            digit += 1
        if (y & mask) != 0:
            digit += 2
        quadkey += str(digit)
    return quadkey


def main(cfg: DictConfig):
    X1, Y1, X2, Y2 = get_xy_corners(cfg.map_box, cfg.zoom)

    data_dir = make_dir(Path(cfg.data_dir) / f"{cfg.zoom}", delete_if_exist=True)

    mask_cache = set()

    for service, source in cfg.sources.items():
        print(f"TMS service: {service}")

        for X, Y in product(range(X1, X2, cfg.num_tiles)[:-1], range(Y1, Y2, cfg.num_tiles)[:-1]):
            # генерация изображения

            x1, x2 = X, X + cfg.num_tiles
            y1, y2 = Y, Y + cfg.num_tiles

            if service == "yandex":
                lon_min, lat_min = tile_pixel_to_lonlat(x1,y2-1,cfg.zoom, 0, cfg.tile_size-1)
                lon_max, lat_max = tile_pixel_to_lonlat(x2-1,y1,cfg.zoom, cfg.tile_size-1, 0)
                x_min, y_max = lat_lon_to_yandex_tile(lat_min, lon_min, cfg.zoom)
                x_max, y_min = lat_lon_to_yandex_tile(lat_max, lon_max, cfg.zoom)
                corners = tuple(product(range(x_min, x_max+1), range(y_max, y_min-1, -1)))
            else:
                corners = tuple(product(range(x1, x2), range(y1, y2)))


            futures = []

            with ThreadPoolExecutor(cfg.max_workers) as executor:
                for x, y in corners:
                    if service in ["google", "arcgis", "yandex"]:
                        url = source.format(x=x, y=y, z=cfg.zoom)
                    elif service == "bing":
                        quadkey = xyz_to_quadkey(x, y, cfg.zoom)
                        url = source.format(quadkey)
                    else:
                        raise NotImplementedError

                    futures.append(executor.submit(get_tile, url))

            image: Image.Image | None = None
            for feat, xy in zip(futures, corners):
                if feat.result() is not None:
                    if service=="yandex":
                        image = paste_tile_yandex(image, feat.result(), xy, (x1, y1, x2, y2), z=cfg.zoom)
                    else:
                        image = paste_tile(image, feat.result(), xy, (x1, y1, x2, y2))

            # сохранение изображения
            out_dir = make_dir(data_dir / f"{str(X).zfill(5)}" / f"{str(Y).zfill(5)}", delete_if_exist=False)
            file = out_dir / f"{service}.jpg"

            if image is not None:
                image.save(file)
                print(file)

            # генерация маски
            if (X, Y) not in mask_cache:
                with tempfile.TemporaryDirectory(dir=root / "data") as tmpdir:
                    file = Path(tmpdir) / "features.geojson"
                    handler = RoadHandler(file, batch=100_000)
                    handler.apply_file(cfg.osm, locations=True)
                    handler.flush()
                    files = list(Path(tmpdir).glob("*.geojson"))
                    if len(files) != 1:
                        all_features = []

                        for file in files:
                            with open(file, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                all_features.extend(data["features"])

                        features = {
                            "type": "FeatureCollection",
                            "features": all_features
                        }
                    else:
                        with open(files[0], "r", encoding="utf-8") as f:
                            features = json.load(f)

                feature_map = collections.defaultdict(list)

                for i, feature in enumerate(features["features"]):
                    if feature["geometry"]["type"] != "Polygon":
                        continue

                    try:
                        for tile in burntiles.burn([feature], zoom=cfg.zoom):
                            feature_map[mercantile.Tile(int(tile[0]), int(tile[1]), int(tile[2]))].append(feature)
                    except ValueError as _:
                        print("Warning: invalid feature {}, skipping".format(i), file=sys.stderr)
                        continue

                mask = None
                corners = tuple(product(range(x1, x2), range(y1, y2)))
                with ThreadPoolExecutor(cfg.max_workers) as executor:
                    for xy in corners:
                        mask = paste_mask(mask, cfg.tile_size, feature_map, xy, (x1, y1, x2, y2), cfg.zoom)

                mask_cache.add((X, Y))

                # сохранение маски
                file = out_dir / "mask.png"
                mask = Image.fromarray(mask, "L")
                mask.save(file, optimize=True)
                print(file)


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg["prepare_satellite_data"]

    main(cfg)
