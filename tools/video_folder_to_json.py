import hashlib
import json
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

from tqdm import tqdm

video_ext = ['.mp4', '.avi', '.flv', '.mkv', '.webm', '.mov']


def parse_args():
    description = """
    This is a json generator, where the videos are arranged in this way:

    ```text
    root/swimming/xxx.mp4
    root/swimming/xxy.avi
    root/swimming/xxz.flv
    
    root/dancing/123.mkv
    root/dancing/nsdf3.webm
    root/dancing/asd932_.mov
    ``` 
    
    You should provide the path of the root folder.
    """
    parser = ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)
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
