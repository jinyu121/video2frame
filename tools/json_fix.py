import json
from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser()

    # Names and folders
    parser.add_argument("annotation", type=str, help="The annotation file, in json format")
    parser.add_argument("fix", type=str, help="The fix list")
    parser.add_argument("--output", type=str, default="", help="The output file")

    args = parser.parse_args()

    if not args.output:
        args.output = Path(args.annotation).name.split(".")[0] + "-fix.json"

    return args


if "__main__" == __name__:
    args = parse_args()

    error_list = [x.strip() for x in Path(args.fix).open()]
    annotation = json.load(Path(args.annotation))

    annotation = {k: v for k, v in annotation.items() if v['path'] not in error_list}

    json.dump(annotation, Path(args.output))
    print("Done")
