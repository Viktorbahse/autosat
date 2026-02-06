import argparse
import concurrent.futures as futures
import os
import sys
import time

import mercantile
import requests
from PIL import Image
from tqdm import tqdm

from tile_utils import (fetch_image, latlon_to_yandex_tile, pixel_to_location,
                        tiles_from_csv, xyz_to_quadkey)

sources = ["yandex", "google", "bing", "arcgis", "all"]

def main(args):
    tiles = list(tiles_from_csv(args.input))

    with requests.Session() as session:
        num_workers = args.rate

        # tqdm has problems with concurrent.futures.ThreadPoolExecutor; explicitly call `.update`
        # https://github.com/tqdm/tqdm/issues/97
        progress = tqdm(total=len(tiles), ascii=True, unit="image")

        with futures.ThreadPoolExecutor(num_workers) as executor:

            def worker(tile):
                tick = time.monotonic()
                def download_from_google(x, y, z, path):
                    url = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
                    url = url.format(x=x, y=y, z=z)
                    res = fetch_image(session, url)
                    if not res:
                        return tile, False
                    try:
                        image = Image.open(res)
                        image.save(path, optimize=True)
                    except OSError:
                        return tile, False
                    tock = time.monotonic()
                    time_for_req = tock - tick
                    time_per_worker = num_workers / args.rate
                    if time_for_req < time_per_worker:
                        time.sleep(time_per_worker - time_for_req)
                    progress.update()
                    return (x,y,z), True
                
                def download_from_arcgis(x, y, z, path):
                    url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
                    url = url.format(x=x, y=y, z=z)
                    res = fetch_image(session, url)
                    if not res:
                        return tile, False
                    try:
                        image = Image.open(res)
                        image.save(path, optimize=True)
                    except OSError:
                        return tile, False
                    tock = time.monotonic()
                    time_for_req = tock - tick
                    time_per_worker = num_workers / args.rate
                    if time_for_req < time_per_worker:
                        time.sleep(time_per_worker - time_for_req)
                    progress.update()
                    return (x,y,z), True
                
                def download_from_bing(x, y, z, path):
                    quadkey = xyz_to_quadkey(x, y, z)
                    server_num = (int(x) + int(y)) % 4 + 1
                    url = f"http://a{server_num}.ortho.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=90"
                    res = fetch_image(session, url)
                    if not res:
                        return (x,y,z), False
                    try:
                        image = Image.open(res)
                        image.save(path, optimize=True)
                    except OSError:
                        return tile, False
                    tock = time.monotonic()
                    time_for_req = tock - tick
                    time_per_worker = num_workers / args.rate
                    if time_for_req < time_per_worker:
                        time.sleep(time_per_worker - time_for_req)
                    progress.update()
                    return tile, True

                def download_from_yandex(x, y, z, path):
                    url = f"https://sat04.maps.yandex.net/tiles?l=sat&x={x}&y={y}&z={z}"
                    res = fetch_image(session, url)
                    if not res:
                        return tile, False
                    try:
                        image = Image.open(res)
                        image.save(path, optimize=True)
                    except OSError:
                        return tile, False
                    tock = time.monotonic()
                    time_for_req = tock - tick
                    time_per_worker = num_workers / args.rate
                    if time_for_req < time_per_worker:
                        time.sleep(time_per_worker - time_for_req)
                    progress.update()
                    return (x,y,z), True

                x, y, z = map(str, [tile.x, tile.y, tile.z])
                if args.source == 'all':
                    flag_a=flag_b=flag_g=flag_y=True
                    
                    os.makedirs(os.path.join(args.out, 'google', z, x), exist_ok=True)
                    path = os.path.join(args.out, 'google', z, x, "{}.{}".format(y, args.ext))
                    if not os.path.isfile(path):
                        tile, flag_y = download_from_google(x,y,z,path)

                    os.makedirs(os.path.join(args.out, 'bing', z, x), exist_ok=True)
                    path = os.path.join(args.out, 'bing', z, x, "{}.{}".format(y, args.ext))
                    if not os.path.isfile(path):
                        tile, flag_b = download_from_bing(x,y,z,path)
                    
                    os.makedirs(os.path.join(args.out, 'arcgis', z, x), exist_ok=True)
                    path = os.path.join(args.out, 'arcgis', z, x, "{}.{}".format(y, args.ext))
                    if not os.path.isfile(path):
                        tile, flag_a = download_from_arcgis(x,y,z,path)

                    os.makedirs(os.path.join(args.out, 'yandex', z, x), exist_ok=True)
                    path = os.path.join(args.out, 'yandex', z, x, "{}.{}".format(y, args.ext))

                    tile = mercantile.Tile(x=int(x), y=int(y), z=int(z))
                    lon, lat = pixel_to_location(tile, dx=0.5, dy=0.5)
                    x, y = latlon_to_yandex_tile(lat, lon, int(z))
                    if not os.path.isfile(path):
                        tile, flag_g = download_from_yandex(x,y,int(z),path)
                    return tile, flag_a*flag_b*flag_g*flag_y
                else:    
                    os.makedirs(os.path.join(args.out, z, x), exist_ok=True)
                    path = os.path.join(args.out, z, x, "{}.{}".format(y, args.ext))

                    if os.path.isfile(path):
                        return tile, True

                    if args.url is not None:
                        url = args.url.format(x=tile.x, y=tile.y, z=tile.z)
                        res = fetch_image(session, url)
                        if not res:
                            return tile, False
                        try:
                            image = Image.open(res)
                            image.save(path, optimize=True)
                        except OSError:
                            return tile, False
                        tock = time.monotonic()
                        time_for_req = tock - tick
                        time_per_worker = num_workers / args.rate
                        if time_for_req < time_per_worker:
                            time.sleep(time_per_worker - time_for_req)
                        progress.update()
                        return tile, True                    
                    elif args.source == 'google':
                        return download_from_google(x,y,z,path)
                    elif args.source == 'arcgis':
                        return download_from_arcgis(x,y,z,path)
                    elif args.source == 'bing':
                        return download_from_bing(x,y,z,path)
                    else:
                        lat, lot = pixel_to_location(mercantile.Tile(x=int(x), y=int(y), z=int(z)), 0.5, 0.5)
                        x, y = latlon_to_yandex_tile(lat, lot, z)
                        return download_from_yandex(x,y,z,path)
            for tile, ok in executor.map(worker, tiles):
                if not ok:
                    print("Warning: {} failed, skipping".format(tile), file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скачивает тайлы из разных источников.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", choices=sources, help="Возможные источники: yandex, google, bing, arcgis, all.")
    group.add_argument("--url", type=str, help="Endpoint with {z}/{x}/{y} variables to fetch image tiles from")
    parser.add_argument("--input", type=str, required=True, help="Путь к .csv файлу (x/y/z).")
    parser.add_argument("--out", type=str, help="Путь к папке куда будут сохранены тайлы в формате path/z/x/y.jpg")
    parser.add_argument("--rate", type=int, default=20, help="rate limit in max. requests per second")
    parser.add_argument("--ext", type=str, default="jpg", help="file format to save images in")

    args = parser.parse_args()

    if args.input:
        if not args.out:
            if args.source != 'all' and args.source in sources:
                args.out = f'data/tiles/{args.source}'
            else:
                args.out = 'data/tiles'        

    main(args)
