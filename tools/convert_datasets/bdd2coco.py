import argparse
import json
import os.path as osp
from scalabel.label.io import load
from scalabel.label.to_coco import (load_coco_config, scalabel2coco_box_track,
                                    scalabel2coco_detection)

DEFAULT_COCO_CONFIG = osp.join(
    osp.dirname(osp.abspath(__file__)), "configs.toml"
)
SHAPE = (720, 1280)


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="bdd100k to coco format")
    parser.add_argument(
        "-l",
        "--label",
        help=(
            "root directory of bdd100k label Json files or path to a label "
            "json file"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help="path to save coco formatted label file",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="det",
        choices=["det", "box_track"],
        help="conversion mode: det or box_track.",
    )
    parser.add_argument(
        "-ri",
        "--remove-ignore",
        action="store_true",
        help="Remove the ignored annotations from the label file.",
    )
    parser.add_argument(
        "-ic",
        "--ignore-as-class",
        action="store_true",
        help="Put the ignored annotations to the `ignored` category.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=4,
        help="number of processes for mot evaluation",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_COCO_CONFIG,
        help="Configuration for COCO categories",
    )
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()
    categories, name_mapping, ignore_mapping = load_coco_config(
        mode=args.mode,
        filepath=args.config,
        ignore_as_class=args.ignore_as_class,
    )

    print("Loading annotations...")
    frames = load(args.label, nprocs=args.nproc)

    print("Converting format...")
    convert_func = dict(
        det=scalabel2coco_detection,
        box_track=scalabel2coco_box_track,
    )[args.mode]
    coco = convert_func(
        shape=SHAPE,
        frames=frames,
        categories=categories,
        name_mapping=name_mapping,
        ignore_mapping=ignore_mapping,
        ignore_as_class=args.ignore_as_class,
        remove_ignore=args.remove_ignore,
    )

    print("Saving converted annotations to disk...")
    with open(args.output, "w") as f:
        json.dump(coco, f)
    print("Finished!")


if __name__ == "__main__":
    main()
