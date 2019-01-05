import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from random import shuffle

import h5py
import lmdb
import numpy as np
from easydict import EasyDict
from tqdm import tqdm


class Storage:
    def put(self, k, v):
        raise NotImplementedError()


class LMDBStorage(Storage):
    def __init__(self, path):
        super().__init__()
        self.database = lmdb.open(path, map_size=1 << 40)

    def put(self, k, v):
        with self.database.begin(write=True, buffers=True) as txn:
            txn.put(k, v)


class HDF5Storage(Storage):
    def __init__(self, path):
        super().__init__()
        self.database = h5py.File(path, 'a')

    def put(self, k, v):
        self.database[k] = v


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Names and folders
    parser.add_argument("annotation_file", type=str, help="The annotation file, in json format")
    parser.add_argument("--db_name", type=str, help="The database to store extracted frames")
    parser.add_argument("--db_type", type=str, choices=["LMDB", "HDF5"], default="HDF5",
                        help="Type of the database, LMDB or HDF5")
    parser.add_argument("--tmp_dir", type=str, default="/tmp", help="Tmp dir")

    # Resize mode
    parser.add_argument("--resize_mode", type=int, default=0, choices=[0, 1, 2],
                        help="Resize mode\n"
                             "  0: Do not resize\n"
                             "  1: 800-600: Resize to W*H\n"
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
    parser.add_argument("-t", "--threads", type=int, default=0, help="Number of threads")
    parser.add_argument("-nrm", "--not_remove", action="store_true", help="Do not delete tmp files at last")

    args = parser.parse_args()
    args = EasyDict(args.__dict__)
    args = modify_args(args)

    return args


def modify_args(args):
    # check the options
    if not args.db_name:
        args.db_name = args.annotation_file.strip(".json")
    if not (args.db_name.lower().endswith(".hdf5") or args.db_name.lower().endswith(".lmdb")):
        if args.db_type == 'HDF5':
            args.db_name += ".hdf5"
        elif args.db_type == 'LMDB':
            args.db_name += ".lmdb"
        else:
            raise Exception('Unknown db_type')
    elif args.db_name.lower().endswith(".hdf5"):
        args.db_type = 'HDF5'
    elif args.db_name.lower().endswith(".lmdb"):
        args.db_type = 'LMDB'

    # Parse the resize mode
    if args.resize_mode == 0:
        args.vf_scale = []
    elif args.resize_mode == 1:
        W, H, *_ = args.resize.split("-")
        W, H = int(W), int(H)
        assert W > 0 and H > 0
        args.vf_scale = [
            "-vf", "scale={}:{}".format(W, H)
        ]
    elif args.resize_mode == 2:
        side = args.resize[0].lower()
        assert side in ['l', 's'], "The (L)onger side, or the (S)horter side?"
        scale = int(args.resize[1:])
        assert scale > 0
        args.vf_scale = [
            "-vf",
            "scale='iw*1.0/{0}(iw,ih)*{1}':'ih*1.0/{0}(iw,ih)*{1}'".format("max" if side == 'l' else 'min', scale)
        ]
    else:
        raise Exception('Unspecified frame scale option')

    args.vf_sample = []
    if args.fps > 0:
        args.vf_sample = [
            "-vsync", "vfr",
            "-vf", "fps={}".format(args.fps)
        ]

    return args


def video_to_frames(args, video_file, tmp_dir):
    cmd = [
        "ffmpeg",
        "-loglevel", "panic",
        "-i", str(video_file),
        *args.vf_scale,
        *args.vf_sample,
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
                step = tot * 1. / (n - 1)
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


def process(args, ith, video_info, frame_db, pbar=None):
    video_file = Path(video_info['path'])
    tqdm.write(str(video_file))
    tmp_dir = Path(args.tmp_dir) / video_file.name
    tmp_dir.mkdir(exist_ok=True)

    frames = video_to_frames(args, video_file, tmp_dir)
    files = sample_frames(args, frames)

    try:
        for fid, f_name in files:
            key = "{:08d}/{:08d}".format(ith, fid)
            s = (tmp_dir / f_name).open("rb").read()
            frame_db.put(key, s if args.db_type == 'LMDB' else np.void(s))

    except RuntimeError as e:
        tqdm.write("When writing [{}] to file, error occurs:{}".format(video_file, e))

    if not args.not_remove:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if pbar:
        pbar.update()


if "__main__" == __name__:
    args = parse_args()

    frame_db = LMDBStorage(args.db_name) if args.db_type == 'LMDB' else HDF5Storage(args.db_name)

    annotations = json.load(Path(args.annotation_file).open())

    executor = ThreadPoolExecutor(max_workers=args.threads) if args.threads > 0 else None
    with tqdm(total=len(annotations)) as pbar:
        for ith, video_info in enumerate(annotations):
            if args.threads > 0:
                executor.submit(process, args, ith, video_info, frame_db, pbar)
            else:
                process(args, ith, video_info, frame_db, pbar)

    print("Done")

