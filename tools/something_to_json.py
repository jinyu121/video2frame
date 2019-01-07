import json
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("class_file", type=str, help="The class definition json")
    parser.add_argument("split", type=str, help="The split json")
    parser.add_argument("video_folder", type=str, help="The video folder")
    parser.add_argument("annotation_file", type=str, help="The output annotation_file json")
    return parser.parse_args()


if "__main__" == __name__:
    args = parse_args()
    class_file = Path(args.class_file)
    split_file = Path(args.split)
    video_folder = Path(args.video_folder)

    # Get class names
    classes = json.load(class_file.open())

    # Get video file list
    video_list = json.load(split_file.open())
    video_list.sort(key=lambda x: int(x['id']))

    # Prepare the annotations
    annotations = []

    for item in tqdm(video_list):
        # clazz_name = template.sub("something", item["template"])
        clazz_name = item["template"].replace("[", "").replace("]", "")
        clazz_num = classes[clazz_name]

        annotations.append({
            "path": str(video_folder / "{}.webm".format(item['id'])),
            "class_name": clazz_name,
            "class": clazz_num
        })

    json.dump(annotations, Path(args.annotation_file).open("w"))

    print("{} classes, {} videos".format(len(classes), len(annotations)))
    print("Done")
