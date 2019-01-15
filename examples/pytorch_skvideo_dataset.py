import json
from random import random, randint

import numpy as np
from skvideo.io import vread, ffprobe
from torch.utils.data import Dataset


class SKVideoDataset(Dataset):
    def __init__(self, annotation, num_frames, clip_duration=-1, resize="", crop=0, flip=0):
        self.annotation = json.load(open(annotation))
        self.num_frames = num_frames
        self.clip_duration = clip_duration
        self.crop = crop
        self.flip = flip
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
        frames = vread(video_path, outputdict=output_parameter)
        if self.crop:
            _, h, w, *_ = frames.shape
            y1 = randint(0, h - self.crop - 1)
            x1 = randint(0, w - self.crop - 1)
            y2, x2 = y1 + self.crop, x1 + self.crop
            frames = frames[:, y1:y2, x1:x2, :]
        if random() < self.flip:
            frames = frames[:, :, ::-1, :]

        video_data = np.array(frames).transpose([3, 0, 1, 2]).astype(np.float32) / 255.

        return video_data, clazz


if "__main__" == __name__:
    import argparse
    from tqdm import tqdm, trange

    parser = argparse.ArgumentParser()
    parser.add_argument("annotation", type=str, help="The annotation file, in json format")
    parser.add_argument("--resize", type=str, default="320x240", help="Resize the video to WxH")
    parser.add_argument("--duration", type=int, default=5, help="Seconds per clip")
    parser.add_argument("--frames", type=int, default=16, help="Num of frames per clip")
    parser.add_argument("--crop", type=int, default=160, help="Crop size")
    args = parser.parse_args()

    dataset = SKVideoDataset(args.annotation, args.frames, args.duration, args.resize, args.crop)
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
