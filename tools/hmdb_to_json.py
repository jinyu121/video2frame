import os
import hashlib
import json

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

video_ext = ['.mp4', '.avi', '.flv', '.mkv', '.webm', '.mov']


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("annotation", type=str, help="The annotation json file")
    parser.add_argument("split_file_folder", type=str, help="The split files folder")
    parser.add_argument("output_prefix", type=str, help="The prefix of output json file's name")
    parser.add_argument("--which_split", type=str, default="1", help="which split to use")
    return parser.parse_args()


if "__main__" == __name__:
	args = parse_args()

	data = json.load(open(args.annotation, "r"))
	annotation = data['annotation']

	split_files = os.listdir(args.split_file_folder)
	split_files = [sf for sf in split_files if sf[-5]==args.which_split]
	train_split = []
	test_split = []
	for sf in split_files:
		sf_path = os.path.join(args.split_file_folder, sf)
		for x in open(sf_path):
			file_name, split_number = x.split()
			if split_number == '1':
				train_split.append(file_name)
			elif split_number == '2':
				test_split.append(file_name)

	# for k, v in annotation.items():
	# 	print(v['path'].split('/')[-1])
	# print(train_split)
	train_annotation = {k: v for k, v in annotation.items() if v['path'].split('/')[-1] in train_split}
	test_annotation = {k: v for k, v in annotation.items() if v['path'].split('/')[-1] in test_split}

	assert (len(train_annotation) + len(test_annotation)) < len(annotation)
	print(len(train_annotation), len(test_annotation), len(annotation))
	# assert len(set(list(train_annotation.keys())+list(test_annotation.keys()))) == len(annotation)

	data['annotation'] = train_annotation
	json.dump(data, Path(args.output_prefix+'_train.json').open("w"), indent=4)

	data['annotation'] = test_annotation
	json.dump(data, Path(args.output_prefix+'_test.json').open("w"), indent=4)
