import json
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

video_ext = ['mp4', 'avi', 'flv', 'mkv', 'webm', 'mov']


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("annotation_file", type=str, help="the annotation file, in json format")
    parser.add_argument("video_folder", type=str, help="the video folder")
    return parser.parse_args()


if "__main__" == __name__:
    args = parse_args()
    annotation_file = Path(args.annotation_file)
    video_folder = Path(args.video_folder)

    classes = sorted([c.name for c in video_folder.iterdir() if c.is_dir()])

    annotations = []

    for clazz in tqdm(video_folder.iterdir()):
        if not clazz.is_dir():
            continue

        clazz_name = clazz.name
        tqdm.write(clazz_name)
        clazz_num = classes.index(clazz_name)

        for f in clazz.iterdir():
            _, f_ext = f.name.split(".")
            if f_ext in video_ext:
                annotations.append({
                    "path": str(f),
                    "class_name": clazz_name,
                    "class": clazz_num
                })
    json.dump(annotations, annotation_file.open("w"))

    print("Done")
