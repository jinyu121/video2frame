import hashlib
import json
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

video_ext = ['.mp4', '.avi', '.flv', '.mkv', '.webm', '.mov']


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("video_folder", type=str, help="The video folder")
    parser.add_argument("output", type=str, help="The annotation json file")
    return parser.parse_args()


if "__main__" == __name__:
    args = parse_args()
    video_folder = Path(args.video_folder)

    # Get class names
    classes = sorted([c.name for c in video_folder.iterdir() if c.is_dir()])

    # Prepare the annotations
    data = {}

    for clazz in tqdm(video_folder.iterdir()):
        if not clazz.is_dir():
            continue

        clazz_name = clazz.name
        clazz_num = classes.index(clazz_name)

        tqdm.write(clazz_name)

        for f in clazz.iterdir():
            if f.suffix.lower() in video_ext:
                video_path = str(f)
                key = hashlib.md5(video_path.encode()).hexdigest()[:8]

                data[key] = {
                    "path": video_path,
                    "class_name": clazz_name,
                    "class": clazz_num
                }

    json.dump(data, Path(args.output).open("w"), indent=4)

    print("{} classes, {} videos".format(len(classes), len(data)))
    print("Done")
