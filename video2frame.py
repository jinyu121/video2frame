import argparse
import json
import os
import pickle
import shutil
import subprocess
from concurrent import futures
from pathlib import Path
from random import shuffle, random

import h5py
import lmdb
import numpy as np
from easydict import EasyDict
from tqdm import tqdm


class Storage:
    def __init__(self):
        self.database = None

    def put(self, video_key, ith_clip, clip_tmp_dir, frame_files):
        raise NotImplementedError()

    def close(self):
        self.database.close()


class LMDBStorage(Storage):
    def __init__(self, path):
        super().__init__()
        self.database = lmdb.open(path, map_size=1 << 40)

    def put(self, video_key, ith_clip, clip_tmp_dir, frame_files):
        for ith_frame, (frame_id, frame_path) in enumerate(frame_files):
            data = (clip_tmp_dir / frame_path).open("rb").read()
            key = "{}/{:03d}/{:08d}".format(video_key, ith_clip, ith_frame)
            with self.database.begin(write=True, buffers=True) as txn:
                txn.put(key.encode(), data)


class HDF5Storage(Storage):
    def __init__(self, path):
        super().__init__()
        self.database = h5py.File(path, 'w')

    def put(self, video_key, ith_clip, clip_tmp_dir, frame_files):
        for ith_frame, (frame_id, frame_path) in enumerate(frame_files):
            data = (clip_tmp_dir / frame_path).open("rb").read()
            key = "{}/{:03d}/{:08d}".format(video_key, ith_clip, ith_frame)
            self.database[key] = np.void(data)


class PKLStorage(Storage):
    def __init__(self, path):
        super().__init__()
        self.base_path = Path(path)

    def put(self, video_key, ith_clip, clip_tmp_dir, frame_files):
        save_dir = self.base_path / video_key
        save_dir.mkdir(exist_ok=True, parents=True)
        frame_data = []
        for ith_frame, (frame_id, frame_path) in enumerate(frame_files):
            data = (clip_tmp_dir / frame_path).open("rb").read()
            frame_data.append(data)
        pickle.dump(frame_data, (save_dir / "{:03d}.pkl".format(ith_clip)).open("wb"))


class FileStorage(Storage):
    def __init__(self, path):
        super().__init__()
        self.base_path = Path(path)

    def put(self, video_key, ith_clip, clip_tmp_dir, frame_files):
        save_dir = self.base_path / video_key / "{:03d}".format(ith_clip)
        save_dir.mkdir(exist_ok=True, parents=True)
        for ith_frame, (frame_id, frame_path) in enumerate(frame_files):
            data = (clip_tmp_dir / frame_path).open("rb").read()
            (save_dir / "{:08d}.jpg".format(ith_frame)).open("wb").write(data)


STORAGE_TYPES = {
    "HDF5": HDF5Storage,
    "LMDB": LMDBStorage,
    "FILE": FileStorage,
    "PKL": PKLStorage
}


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Names and folders
    parser.add_argument("annotation_file", type=str, help="The annotation file, in json format")
    parser.add_argument("--db_name", type=str, help="The database to store extracted frames")
    parser.add_argument("--db_type", type=str, choices=["LMDB", "HDF5", "FILE", "PKL"], default="HDF5",
                        help="Type of the database, LMDB or HDF5")
    parser.add_argument("--tmp_dir", type=str, default="/tmp", help="Tmp dir")

    # Clips
    parser.add_argument("--duration", type=float, default=-1, help="Length of the clip")
    parser.add_argument("--clips", type=int, default=1, help="Num of clips per video")

    # Resize mode
    parser.add_argument("--resize_mode", type=int, default=0, choices=[0, 1, 2],
                        help="Resize mode\n"
                             "  0: Do not resize\n"
                             "  1: 800x600: Resize to W*H\n"
                             "  2: L600 or S600: keep the aspect ration and scale the longer/shorter side to s"
                        )
    parser.add_argument("--resize", type=str, help="Parameter of resize mode")

    # Frame sampling options
    parser.add_argument("--fps", type=float, default=-1, help="Sample the video at X fps")
    parser.add_argument("--sample_mode", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Frame sampling options\n"
                             "  0: Keep all frames\n"
                             "  1: Uniformly sample n frames\n"
                             "  2: Randomly sample n frames\n"
                             "  3: Mod mode"
                        )
    parser.add_argument("--sample", type=str, help="Parameter of sample mode")

    # performance
    parser.add_argument("--threads", type=int, default=0, help="Number of threads")
    parser.add_argument("--not_remove", action="store_true", help="Do not delete tmp files at last")

    args = parser.parse_args()
    args = EasyDict(args.__dict__)
    args = modify_args(args)

    return args


def modify_args(args):
    # check the options
    if not args.db_name:
        if args.annotation_file.lower().endswith(".json"):
            args.db_name = args.annotation_file[:-5]
        else:
            args.db_name = args.annotation_file

    if args.db_name.lower().endswith(".hdf5"):
        args.db_type = 'HDF5'
    elif args.db_name.lower().endswith(".lmdb"):
        args.db_type = 'LMDB'
    else:
        if args.db_type == 'HDF5':
            args.db_name += ".hdf5"
        elif args.db_type == 'LMDB':
            args.db_name += ".lmdb"

    # Range check
    args.clips = max(args.clips, 1)
    args.duration = max(args.duration, 0)

    # Parse the resize mode
    args.vf_setting = []
    if args.resize_mode == 0:
        pass
    elif args.resize_mode == 1:
        W, H, *_ = args.resize.split("x")
        W, H = int(W), int(H)
        assert W > 0 and H > 0
        args.vf_setting.extend([
            "-vf", "scale={}:{}".format(W, H)
        ])
    elif args.resize_mode == 2:
        side = args.resize[0].lower()
        assert side in ['l', 's'], "The (L)onger side, or the (S)horter side?"
        scale = int(args.resize[1:])
        assert scale > 0
        args.vf_setting.extend([
            "-vf",
            "scale='iw*1.0/{0}(iw,ih)*{1}':'ih*1.0/{0}(iw,ih)*{1}'".format("max" if side == 'l' else 'min', scale)
        ])
    else:
        raise Exception('Unspecified frame scale option')

    # Parse the fps setting
    if args.fps > 0:
        args.vf_setting.extend([
            "-r", "{}".format(args.fps)
        ])

    if args.threads:
        if args.threads < 0:
            args.threads = max(os.cpu_count() / 2, 1)

    return args


def get_video_meta(video_file):
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_streams",
            "-print_format", "json",
            video_file
        ]
        output = subprocess.check_output(cmd)
        output = json.loads(output)

        streamsbytype = {}
        for stream in output["stream"]:
            streamsbytype[stream["codec_type"].lower()] = stream

        return streamsbytype
    except:
        return {}


def video_to_frames(args, video_file, video_meta, tmp_dir):
    # Random clip the video
    clip_setting = []
    if args.duration > 0:
        video_duration = float(video_meta["video"]["duration"])
        sta = max(0., random() * (video_duration - args.duration))
        dur = min(args.duration, video_duration - sta)
        clip_setting.extend([
            "-ss", "{}".format(sta),
            "-t", "{}".format(dur)
        ])

    cmd = [
        "ffmpeg",
        "-loglevel", "panic",
        "-vsync", "vfr",
        "-i", str(video_file),
        *args.vf_setting,
        *clip_setting,
        "-qscale:v", "2",
        str(tmp_dir / "%8d.jpg")
    ]
    subprocess.call(cmd)

    frames = [(int(f.name.split('.')[0]), f) for f in tmp_dir.iterdir()]
    frames.sort(key=lambda x: x[0])

    return frames


def sample_frames(args, frames):
    if args.sample_mode:
        n = int(args.sample)
        assert n > 0, "N must >0, but get {}".format(n)

        tot = len(frames)
        if args.sample_mode == 1:  # Uniformly sample n frames
            if n == 1:
                index = [tot >> 1]
            else:
                step = (tot - 1.) / (n - 1)
                index = [round(x * step) for x in range(n)]
            frames = [frames[x] for x in index]
        elif args.sample_mode == 2:  # Randomly sample n frames
            shuffle(frames)
            frames = frames[:min(n, tot)]
            frames.sort(key=lambda x: x[0])
        elif args.sample_mode == 3:  # Mod mode.
            frames = frames[::n]
        else:
            raise AttributeError("Sample mode is not supported")
    return frames


def process(args, video_key, video_info, frame_db):
    video_file = Path(video_info['path'])
    video_tmp_dir = Path(args.tmp_dir) / "{}".format(video_key)
    video_tmp_dir.mkdir(exist_ok=True)

    video_meta = get_video_meta(video_file)
    if not video_meta:
        raise RuntimeError("Can not get video info")

    for ith_clip in range(args.clips):
        clip_tmp_dir = video_tmp_dir / "{:03d}".format(ith_clip)

        frames = video_to_frames(args, video_file, video_meta, clip_tmp_dir)
        if not frames:
            raise RuntimeError("Extract frame failed")

        frames = sample_frames(args, frames)
        if not frames:
            raise RuntimeError("No frames in video")

        # Save to database
        frame_db.put(video_key, ith_clip, clip_tmp_dir, frames)

        if not args.not_remove:
            shutil.rmtree(clip_tmp_dir, ignore_errors=True)

    if not args.not_remove:
        shutil.rmtree(video_tmp_dir, ignore_errors=True)

    return "OK"


if "__main__" == __name__:
    args = parse_args()
    Path(args.tmp_dir).mkdir(exist_ok=True)

    frame_db = STORAGE_TYPES[args.db_type](args.db_name)

    annotations = json.load(Path(args.annotation_file).open())

    if args.threads > 0:
        with futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
            jobs = {
                executor.submit(process, args, video_key, video_info, frame_db): video_info['path']
                for video_key, video_info in annotations.items()
            }
            for future in tqdm(futures.as_completed(jobs), total=len(annotations)):
                try:
                    video_status = future.result()
                except Exception as e:
                    tqdm.write("{} : {}".format(jobs[future], e))
                else:
                    tqdm.write("{} : {}".format(jobs[future], video_status))
    else:
        for ith, video_info in enumerate(tqdm(annotations)):
            try:
                video_status = process(args, ith, video_info, frame_db)
            except Exception as e:
                tqdm.write("{} : {}".format(video_info['path'], e))
            else:
                tqdm.write("{} : {}".format(video_info['path'], video_status))

    frame_db.close()
    print("Done")
