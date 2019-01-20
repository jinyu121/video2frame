import json
import re
import shutil
import subprocess
import warnings
from concurrent import futures
from pathlib import Path
from random import randint, random, shuffle

from tqdm import tqdm

from storage import STORAGE_TYPES
from util import parse_args, retry

ffmpeg_duration_template = re.compile(r"time=\s*(\d+):(\d+):(\d+)\.(\d+)")


def get_video_duration(video_file):
    cmd = [
        "ffmpeg",
        "-i", str(video_file),
        "-f", "null", "-"
    ]

    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    result_all = ffmpeg_duration_template.findall(output.decode())
    if result_all:
        result = result_all[-1]
        duration = float(result[0]) * 60 * 60 \
                   + float(result[1]) * 60 \
                   + float(result[2]) \
                   + float(result[3]) * (10 ** -len(result[3]))
    else:
        duration = -1
    return duration


def get_video_meta(video_file):
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_streams",
            "-print_format", "json",
            str(video_file)
        ]
        output = subprocess.check_output(cmd)
        output = json.loads(output)

        streamsbytype = {}
        for stream in output["streams"]:
            streamsbytype[stream["codec_type"].lower()] = stream

        return streamsbytype
    except:
        return {}


@retry()
def video_to_frames(args, video_file, video_meta, tmp_dir, error_when_empty=True):
    # Random clip the video
    clip_setting = []
    if args.duration > 0:
        try:
            video_duration = float(video_meta["video"]["duration"])
        except:
            video_duration = get_video_duration(video_file)
        if video_duration > 0:
            sta = max(0., random() * (video_duration - args.duration))
            dur = min(args.duration, video_duration - sta)
            clip_setting.extend([
                "-ss", "{}".format(sta),
                "-t", "{}".format(dur)
            ])
        else:
            warnings.warn("Ignore `duration` parameter for video {}.".format(video_file))

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

    if error_when_empty and not frames:
        raise RuntimeError("Extract frame failed")

    return frames


@retry()
def sample_frames(args, frames, error_when_empty=True):
    if args.sample_mode:
        assert args.sample > 0, "Sample must >0, but get {}".format(args.sample)

        tot = len(frames)
        if args.sample_mode == 1:  # Uniformly sample n frames
            if args.sample == 1:
                index = [tot >> 1]
            else:
                step = (tot - 1.) / (args.sample - 1)
                index = [round(x * step) for x in range(args.sample)]
            frames = [frames[x] for x in index]
        elif args.sample_mode == 2:  # Randomly sample n continuous frames
            sta = randint(0, max(0, len(frames) - args.sample - 1))
            frames = frames[sta:sta + args.sample]
        elif args.sample_mode == 3:  # Randomly sample n frames
            shuffle(frames)
            frames = frames[:min(args.sample, tot)]
            frames.sort(key=lambda x: x[0])
        elif args.sample_mode == 4:  # Sample 1 frame every n frames
            frames = frames[::args.sample]
        else:
            raise AttributeError("Sample mode is not supported")

    if error_when_empty and not frames:
        raise RuntimeError("No frame selected")

    return frames


def process(args, video_key, video_info, frame_db):
    video_file = Path(video_info['path'])
    video_tmp_dir = Path(args.tmp_dir) / "{}".format(video_key)
    video_tmp_dir.mkdir(exist_ok=True)

    if not video_file.exists():
        raise RuntimeError("Video not exists")

    video_meta = get_video_meta(video_file)
    if not video_meta:
        raise RuntimeError("Can not get video info")

    for ith_clip in range(args.clips):
        clip_tmp_dir = video_tmp_dir / "{:03d}".format(ith_clip)
        clip_tmp_dir.mkdir(exist_ok=True, parents=True)

        # Get all frames
        frames = video_to_frames(args, video_file, video_meta, clip_tmp_dir)

        # Sample frames
        frames = sample_frames(args, frames)

        # Save to database
        frame_db.put(video_key, ith_clip, clip_tmp_dir, frames)

        if not args.keep:
            shutil.rmtree(clip_tmp_dir, ignore_errors=True)

    if not args.keep:
        shutil.rmtree(video_tmp_dir, ignore_errors=True)

    return "OK"


if "__main__" == __name__:
    args = parse_args()
    Path(args.tmp_dir).mkdir(exist_ok=True)

    frame_db = STORAGE_TYPES[args.db_type](args.db_name)

    annotation_all = json.load(Path(args.annotation_file).open())
    annotation = annotation_all["annotation"]
    total = len(annotation)
    fails = []

    if args.threads > 0:
        with futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
            jobs = {
                executor.submit(process, args, video_key, video_info, frame_db): video_info['path']
                for video_key, video_info in annotation.items()
            }
            for future in tqdm(futures.as_completed(jobs), total=total):
                try:
                    video_status = future.result()
                except Exception as e:
                    tqdm.write("{} : {}".format(jobs[future], e))
                    fails.append(jobs[future])
                else:
                    tqdm.write("{} : {}".format(jobs[future], video_status))
    else:
        for video_key, video_info in tqdm(annotation.items()):
            try:
                video_status = process(args, video_key, video_info, frame_db)
            except Exception as e:
                tqdm.write("{} : {}".format(video_info['path'], e))
                fails.append(video_info['path'])
            else:
                tqdm.write("{} : {}".format(video_info['path'], video_status))

    frame_db.close()

    print("Processed {} videos".format(total))
    if not fails:
        print("All success! Congratulations!")
    else:
        print("{} Success, {} Error".format(total - len(fails), len(fails)))

        annotation = {k: v for k, v in annotation.items() if v['path'] not in fails}
        annotation_all["annotation"] = annotation
        if args.annotation_file.lower().endswith(".json"):
            save_path = args.annotation_file[:-5] + "-fix.json"
        else:
            save_path = args.annotation_file + "-fix.json"
        json.dump(annotation_all, Path(save_path).open("w"), indent=4)

    print("All Done!")
