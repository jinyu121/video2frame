import pickle
import shutil
from pathlib import Path

import h5py
import lmdb
import numpy as np


class Storage:
    def __init__(self):
        self.database = None

    def put(self, video_key, ith_clip, clip_tmp_dir, frame_files):
        raise NotImplementedError()

    def close(self):
        pass


class LMDBStorage(Storage):
    def __init__(self, path):
        super().__init__()
        self.database = lmdb.open(path, map_size=1 << 40)

    def put(self, video_key, ith_clip, clip_tmp_dir, frame_files):
        with self.database.begin(write=True, buffers=True) as txn:
            for ith_frame, (frame_id, frame_path) in enumerate(frame_files):
                data = (clip_tmp_dir / frame_path).open("rb").read()
                key = "{}/{:03d}/{:08d}".format(video_key, ith_clip, ith_frame)
                txn.put(key.encode(), data)

    def close(self):
        self.database.close()


class HDF5Storage(Storage):
    def __init__(self, path):
        super().__init__()
        self.database = h5py.File(path, 'w')

    def put(self, video_key, ith_clip, clip_tmp_dir, frame_files):
        for ith_frame, (frame_id, frame_path) in enumerate(frame_files):
            data = (clip_tmp_dir / frame_path).open("rb").read()
            key = "{}/{:03d}/{:08d}".format(video_key, ith_clip, ith_frame)
            self.database[key] = np.void(data)

    def close(self):
        self.database.close()


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
            shutil.copy(str((clip_tmp_dir / frame_path)), str(save_dir / "{:08d}.jpg".format(ith_frame)))


STORAGE_TYPES = {
    "HDF5": HDF5Storage,
    "LMDB": LMDBStorage,
    "FILE": FileStorage,
    "PKL": PKLStorage
}
