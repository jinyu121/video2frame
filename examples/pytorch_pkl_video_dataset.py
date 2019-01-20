import json
import pickle
from io import BytesIO
from pathlib import Path
from random import randint

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm, trange


class PKLVideoDataset(Dataset):
    def __init__(self, annotation, database, clips=1, frames=16, transform=None):
        super().__init__()

        self.num_clips = clips
        assert self.num_clips > 0
        self.num_frames_per_clip = frames
        assert self.num_frames_per_clip >= 0
        self.transform = transform

        data = json.load(open(annotation, "r"))
        self.n_classes = data['meta']["class_num"]
        self.annotation = data['annotation']
        self.base_dir = Path(database)
        self.videos = sorted([x for x in self.annotation.keys()])

    def __getitem__(self, index):
        video_id = self.videos[index]
        video_clip_choice = "{}/{:03d}".format(video_id, randint(0, self.num_clips - 1))

        annotation = self.annotation[video_id]
        frames_binary = pickle.load(video_clip_choice + ".pkl")

        # Sample the frames
        len_of_frames = len(frames_binary)
        if len_of_frames != self.num_frames_per_clip > 0:
            if self.num_frames_per_clip == 1:
                frame_index = [len_of_frames / 2]
            else:
                skips = (len_of_frames - 1) * 1. / (self.num_frames_per_clip - 1)
                frame_index = [round(fi * skips) for fi in range(self.num_frames_per_clip)]
        else:
            frame_index = list(range(len_of_frames))

        # Decode the frames
        frames = [
            Image.open(
                BytesIO(frames_binary[ith_frame])
            ) for ith_frame in frame_index
        ]

        # To video blob
        video_data = np.array([np.asarray(x) for x in frames])

        if self.transform:
            video_data = self.transform(video_data)

        return video_data, annotation['class']

    def __len__(self):
        return len(self.annotation)

    def __repr__(self):
        return "{} {} videos, {} clips per video, {}".format(
            type(self), len(self), self.num_clips,
            "Sample to {} frames".format(self.num_frames_per_clip) if self.num_frames_per_clip else "Not sampled"
        )


if "__main__" == __name__:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("annotation", type=str, help="The annotation file, in json format")
    parser.add_argument("database", type=str, help="The hdf5 file")
    parser.add_argument("--clips", type=int, default=1, help="Num of video clips")
    parser.add_argument("--frames", type=int, default=16, help="Num of frames per clip")
    args = parser.parse_args()

    dataset = PKLVideoDataset(
        annotation=args.annotation, database=args.database, clips=args.clips, frames=args.frames)
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
