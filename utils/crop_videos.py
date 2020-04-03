import json
import os
import subprocess
from glob import glob
from shutil import copyfile

from joblib import delayed, Parallel
from tqdm import tqdm

with open("PATH/TO/borders.json") as f:
    fails_borders = json.load(f)

path = '/local3/vondrick3/datasets/fails/scenes'

newpath = '/proj/vondrick/datasets/fails/scene_clips_split_cropped'

vids = glob(os.path.join(path, '*', '*', '*.mp4'))


def crop_video(v):
    l, r = fails_borders[os.path.splitext(os.path.basename(v))[0]]
    newv = os.path.join(newpath, os.path.sep.join(v.rsplit(os.path.sep, 3)[1:]))
    if l > 0 and r > 0:
        cmd = [
            'ffmpeg',
            '-i',
            v,
            '-filter:v',
            f'crop=in_w*{r - l}:in_h:in_w*{l}:0',
            '-loglevel',
            'panic',
            newv
        ]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print('Command failed')
            print(' '.join(cmd))
    else:
        copyfile(v, newv)


# # for v in tqdm(vids):
# if __name__ == "__main__":
#     Parallel(n_jobs=2)(delayed(crop_video)(v) for v in tqdm(vids))

for v in tqdm(vids):
    crop_video(v)