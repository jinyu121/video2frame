import hashlib
import json
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

from tqdm import tqdm


def parse_args():
    description = """
    This is a json generator that converts the `Something-Something` dataset.

    You should provide the following path(s):
    1. The class-definition file (e.g. /Dataset/Something/annotation/something-something-v2-labels.json).
    2. The annotation file (e.g. /Dataset/Something/annotation/something-something-v2-train.json).
    3. The video root folder (e.g. /Dataset/Something/video).
    """
    parser = ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("classes", type=str, help="The class definition json file")
    parser.add_argument("annotation", type=str, help="The annotation json file")
    parser.add_argument("video_folder", type=str, help="The video folder")
    parser.add_argument("output", type=str, help="The output json file")
    return parser.parse_args()


if "__main__" == __name__:
    args = parse_args()
    video_folder = Path(args.video_folder)

    # Get class names
    classes = json.load(Path(args.classes).open())

    # Get video file list
    video_list = json.load(Path(args.annotation).open())
    video_list.sort(key=lambda x: int(x['id']))

    # Prepare the annotations
    data = {}

    for item in tqdm(video_list):
        # clazz_name = template.sub("something", item["template"])
        clazz_name = item["template"].replace("[", "").replace("]", "")
        clazz_num = int(classes[clazz_name])

        video_path = str(video_folder / "{}.webm".format(item['id']))
        key = hashlib.md5(video_path.encode()).hexdigest()[:8]

        data[key] = {
            "path": video_path,
            "class_name": clazz_name,
            "class": clazz_num
        }

    data_all = {
        "meta": {
            "class_num": len(classes),
            "class_name": classes
        },
        "annotation": data
    }
    json.dump(data_all, Path(args.output).open("w"), indent=4)

    print("{} classes, {} videos".format(len(classes), len(data)))
    print("Done")
