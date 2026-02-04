import argparse
import os

from handlers import (BuildingHandler, FieldHandler, ParkingHandler,
                      RoadHandler, WaterBodyHandler)

handlers = {
    "parking": ParkingHandler,
    "building": BuildingHandler,
    "road": RoadHandler,
    "water": WaterBodyHandler,
    "field": FieldHandler,
}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="extract",
        description="Extracts GeoJSON features from OpenStreetMap",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--type", type=str, required=True, choices=handlers.keys(), help="type of feature to extract")
    parser.add_argument("--batch", type=int, default=100000, help="number of features to save per file")
    parser.add_argument("map", type=str, help="path to .osm.pbf base map")
    parser.add_argument("out", type=str, nargs="?", default=None, help="path to GeoJSON file or output directory")

    return parser


def run_from_args(args):
    handler_cls = handlers[args.type]
    out_path = args.out if args.out is not None else f"data/{args.type}/{args.type}s.geojson"
    out_dir = out_path if out_path.endswith(os.sep) else os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    handler = handler_cls(out_path, args.batch)
    handler.apply_file(filename=args.map, locations=True)
    handler.flush()



def main():
    parser = build_parser()
    args = parser.parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()
