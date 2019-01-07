import json
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

video_ext = ['.mp4', '.avi', '.flv', '.mkv', '.webm', '.mov']


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("annotation_file", type=str, help="The annotation file, in json format")
    parser.add_argument("video_folder", type=str, help="The video folder")
    return parser.parse_args()


if "__main__" == __name__:
    args = parse_args()
    video_folder = Path(args.video_folder)

    # Get class names
    classes = sorted([c.name for c in video_folder.iterdir() if c.is_dir()])

    # Prepare the annotations
    annotations = []

    for clazz in tqdm(video_folder.iterdir()):
        if not clazz.is_dir():
            continue

        clazz_name = clazz.name
        tqdm.write(clazz_name)
        clazz_num = classes.index(clazz_name)

        for f in clazz.iterdir():
            if f.suffix in video_ext:
                annotations.append({
                    "path": str(f),
                    "class_name": clazz_name,
                    "class": clazz_num
                })

    json.dump(annotations, Path(args.annotation_file).open("w"))

    print("{} classes, {} videos".format(len(classes), len(annotations)))
    print("Done")
