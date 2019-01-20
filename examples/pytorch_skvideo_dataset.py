import json
from random import random

from skvideo.io import ffprobe, vread
from torch.utils.data import Dataset


class SKVideoDataset(Dataset):
    def __init__(self, annotation, frames, duration=-1, resize="", transform=None):
        data = json.load(open(annotation, "r"))
        self.n_classes = data['meta']["class_num"]
        self.annotation = data['annotation']
        self.num_frames = frames
        self.clip_duration = duration
        self.transform = transform
        self.base_parameter = {"-vframes": "{}".format(self.num_frames)}
        if resize:
            w, h, *_ = (int(x) for x in resize.split("x")[:2])
            self.base_parameter.update({"-vf": "scale='{}:{}'".format(w, h)})

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        annotation = self.annotation[index]
        video_path = annotation['path']
        clazz = annotation['class']

        metadata = ffprobe(video_path)
        duration = float(metadata["video"]["@duration"])

        output_parameter = self.base_parameter

        if self.clip_duration > 0:
            sta = int(random() * max((duration - self.clip_duration), 0.))
            output_parameter.update({
                "-ss": "{}".format(sta),
                "-t": "{}".format(min(self.clip_duration, duration - sta))
            })
        video_data = vread(video_path, outputdict=output_parameter)

        if self.transform:
            video_data = self.transform(video_data)

        return video_data, clazz


if "__main__" == __name__:
    import argparse
    from tqdm import tqdm, trange

    parser = argparse.ArgumentParser()
    parser.add_argument("annotation", type=str, help="The annotation file, in json format")
    parser.add_argument("--resize", type=str, default="320x240", help="Resize the video to WxH")
    parser.add_argument("--duration", type=int, default=5, help="Seconds per clip")
    parser.add_argument("--frames", type=int, default=16, help="Num of frames per clip")
    args = parser.parse_args()

    dataset = SKVideoDataset(
        annotation=args.annotation, frames=args.frames, duration=args.duration, resize=args.resize)
    error_index = []

    for i in trange(len(dataset)):
        try:
            frame, label = dataset[i]
            tqdm.write("Index {}, Class Label {}, Shape {}".format(i, label, frame.shape))
        except Exception as e:
            tqdm.write("=====> Video {} check failed".format(i))
            error_index.append(i)

    print("There are {} videos.".format(len(dataset)))
    if not error_index:
        print("All is well! Congratulations!")
    else:
        print("Ooops! There are {} bad videos:".format(len(error_index)))
        print(error_index)
