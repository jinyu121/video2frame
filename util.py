import argparse
import os
from functools import wraps

from easydict import EasyDict


def retry(tries=5):
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries = tries
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except:
                    mtries -= 1
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return deco_retry


class RawTextArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                                           argparse.RawTextHelpFormatter):
    # RawTextHelpFormatter implements _split_lines
    # ArgumentDefaultsHelpFormatter implements _get_help_string
    # so we can guess that they will work together just fine.
    pass


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=RawTextArgumentDefaultsHelpFormatter)

    # Names and folders
    parser.add_argument("annotation_file", type=str, help="The annotation file, in json format")
    parser.add_argument("--db_name", type=str, help="The database to store extracted frames")
    parser.add_argument("--db_type", type=str, choices=["LMDB", "HDF5", "FILE", "PKL"], default="HDF5",
                        help="Type of the database")
    parser.add_argument("--tmp_dir", type=str, default="/tmp", help="Temporary folder")

    # Clips
    parser.add_argument("--clips", type=int, default=1, help="Num of clips per video")
    parser.add_argument("--duration", type=float, default=-1, help="Length of each clip")

    # Resize mode
    parser.add_argument("--resize_mode", type=int, default=0, choices=[0, 1, 2],
                        help="Resize mode\n"
                             "  0: Do not resize\n"
                             "  1: 800x600: Resize to WxH\n"
                             "  2: L600 or S600: keep the aspect ration and scale the longer/shorter side to s"
                        )
    parser.add_argument("--resize", type=str, help="Parameter of resize mode")

    # Frame sampling options
    parser.add_argument("--fps", type=float, default=-1, help="Sample the video at X fps")
    parser.add_argument("--sample_mode", type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help="Frame sampling options\n"
                             "  0: Keep all frames\n"
                             "  1: Uniformly sample n frames\n"
                             "  2: Randomly sample n continuous frames\n"
                             "  3: Randomly sample n frames\n"
                             "  4: Sample 1 frame every n frames"
                        )
    parser.add_argument("--sample", type=int, help="How many frames")

    # performance
    parser.add_argument("--threads", type=int, default=0, help="Number of threads")
    parser.add_argument("--keep", action="store_true", help="Do not delete temporary files at last")

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
