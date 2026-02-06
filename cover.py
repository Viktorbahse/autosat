import argparse
import csv
import json
from pathlib import Path

from supermercado import burntiles
from tqdm import tqdm


def get_tiles(path, zoom):
    with open(path, "r", encoding="utf-8") as f:
        features = json.load(f)

    tiles = set()
    for feature in tqdm(features.get("features", []), ascii=True, unit="feature"):
        try:
            tiles.update(map(tuple, burntiles.burn([feature], zoom).tolist()))
        except Exception:
            continue
    return tiles

def write_tiles(out_path, tiles):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not tiles:
        return
    with open(out_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerows(sorted(tiles))

def main(args):
    if args.type == "all":
        for t in types:
            dir_path = Path(f"data/{t}/")
            tiles = set()
            for path in dir_path.rglob("*.geojson"):
                if path.is_file():
                    tiles.update(get_tiles(path, args.zoom))
            out_path = Path(args.out) if args.out is not None else Path(f"data/{t}_tiles_{args.zoom}.csv")
            write_tiles(out_path, tiles)

    elif args.type is not None:
        dir_path = Path(f"data/{args.type}/")
        tiles = set()
        for path in dir_path.rglob("*.geojson"):
            if path.is_file():
                tiles.update(get_tiles(path, args.zoom))
        out_path = Path(args.out) if args.out is not None else Path(f"data/{args.type}_tiles_{args.zoom}.csv")
        write_tiles(out_path, tiles)

    else:
        in_path = Path(args.input)
        tiles = get_tiles(in_path, args.zoom)
        out_path = Path(args.out)
        write_tiles(out_path, tiles)

if __name__ == "__main__":
    types = ["parking", "building", "road", "water", "field"]

    parser = argparse.ArgumentParser(description="Создает набор tiles по geojson.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--type", choices=types + ["all"], help="Класс объектов (взаимоисключающе с --input).")
    group.add_argument("--input", type=str, help="Путь к .geojson файлу (взаимоисключающе с --type).")

    parser.add_argument("--zoom", type=int, default=20, help="zoom level of tiles")
    parser.add_argument("--out", type=str, help="path to csv file to store tiles in") 

    args = parser.parse_args()

    if args.input:
        in_path = Path(args.input)
        if not in_path.is_file():
            parser.error(f"Input file not found: {args.input}")
        if not args.out:
            args.out = str(in_path.with_name(in_path.stem + f"_tiles_{args.zoom}.csv"))

    main(args)
