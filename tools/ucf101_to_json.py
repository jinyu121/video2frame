import hashlib
import json
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

from tqdm import tqdm


def parse_args():
    description = """
    This is a json generator that converts the `UCF101` dataset.
    
    You should provide the following path(s):
    1. The class-definition file (e.g. /Dataset/UCF101/annotation/classInd.txt).
    2. The annotation file (e.g. /Dataset/UCF101/annotation/trainlist01.txt).
    3. The video root folder (e.g. /Dataset/UCF101/video).
    """
    parser = ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("classes", type=str, help="The class definition file")
    parser.add_argument("annotation", type=str, help="The annotation txt file")
    parser.add_argument("video_folder", type=str, help="The video folder")
    parser.add_argument("output", type=str, help="The output json file")
    return parser.parse_args()


if "__main__" == __name__:
    args = parse_args()
    video_folder = Path(args.video_folder)

    classes = [x.split()[1] for x in open(args.classes)]
    annotations = [x.split()[0] for x in open(args.annotation)]

    # Prepare the annotations
    data = {}

    for video_path in tqdm(annotations):
        clazz_name, *_ = video_path.split('/')
        clazz_num = classes.index(clazz_name)

        key = hashlib.md5(video_path.encode()).hexdigest()[:8]

        data[key] = {
            "path": str(video_folder / video_path),
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
