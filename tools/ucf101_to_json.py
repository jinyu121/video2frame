import json
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

video_ext = ['.mp4', '.avi', '.flv', '.mkv', '.webm', '.mov']


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("class_file", type=str, help="The class definition file")
    parser.add_argument("annotation_file", type=str, help="The input txt file")
    parser.add_argument("video_folder", type=str, help="The video folder")
    parser.add_argument("output_file", type=str, help="The output json file")
    return parser.parse_args()


if "__main__" == __name__:
    args = parse_args()
    video_folder = Path(args.video_folder)

    classes = [x.split()[1] for x in open(args.class_file)]
    annotations = [x.split()[0] for x in open(args.annotation_file)]

    # Prepare the annotations
    output_annotations = []

    for record in tqdm(annotations):
        clazz_name, *_ = record.split('/')
        clazz_num = classes.index(clazz_name)

        output_annotations.append({
            "path": str(video_folder / record),
            "class_name": clazz_name,
            "class": clazz_num
        })

    json.dump(output_annotations, Path(args.output_file).open("w"))

    print("{} classes, {} videos".format(len(classes), len(output_annotations)))
    print("Done")
