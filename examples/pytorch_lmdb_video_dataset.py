import json
from io import BytesIO
from random import randint

import lmdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm, trange


class LMDBVideoDataset(Dataset):
    def __init__(self, annotation_file, database_file, num_frames_per_clip, crop_size=0):
        super().__init__()

        self.num_frames_per_clip = num_frames_per_clip
        assert self.num_frames_per_clip >= 0
        self.crop_size = crop_size
        assert self.crop_size >= 0

        self.annotation = json.load(open(annotation_file, "r"))
        self.database = lmdb.open(database_file, readonly=True).begin().cursor()

    def __getitem__(self, index):
        annotation = self.annotation[index]

        # Decode the frames
        frames = [
            Image.open(
                BytesIO(
                    self.database.get(
                        "{:08d}/{:08d}".format(index, i).encode()
                    )
                )
            ) for i in range(self.num_frames_per_clip)
        ]

        # Crop the videos
        if self.crop_size:
            w, h = frames[0].size
            y1 = randint(0, h - self.crop_size - 1)
            x1 = randint(0, w - self.crop_size - 1)
            y2, x2 = y1 + self.crop_size, x1 + self.crop_size
            frames = [im.crop((x1, y1, x2, y2)) for im in frames]

        # To video blob
        video_data = np.array([np.asarray(x) for x in frames]).transpose([3, 0, 1, 2])

        return video_data, annotation['class']

    def __len__(self):
        return len(self.annotation)

    def __repr__(self):
        return "{} {} videos, {}, {}".format(
            type(self), len(self),
            "{} frames per video".format(self.num_frames_per_clip),
            "Crop to {}".format(self.crop_size) if self.crop_size else "Not cropped")


if "__main__" == __name__:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("annotation", type=str, help="The annotation file, in json format")
    parser.add_argument("data", type=str, help="The hdf5 file")
    parser.add_argument("--frames", type=int, default=16, help="Num of frames per clip")
    parser.add_argument("--crop", type=int, default=160, help="Crop size")
    args = parser.parse_args()

    dataset = LMDBVideoDataset(args.annotation, args.data, args.frames, args.crop)
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
