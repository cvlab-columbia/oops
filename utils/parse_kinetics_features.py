import json
import os
from collections import Counter, defaultdict
from glob import glob

from tqdm import tqdm


def process():
    fns = glob(
        '/local/vondrick/dave/slidingwindow/fails_kinetics_features/fails_kinetics_preds_*.json')

    data = []
    for fn in fns:
        with open(fn) as f:
            data.extend(json.load(f))

    vid_actions = defaultdict(lambda: defaultdict(int))

    for p_vec, fn in tqdm(data):
        dirname = os.path.dirname(fn[-1])
        argmax = max(range(len(p_vec)), key=lambda i: p_vec[i])
        vid_actions[dirname][argmax] += 1

    for vid, act_freqs in tqdm(vid_actions.items()):
        vid_actions[vid] = Counter(act_freqs).most_common(1)[0][0]

    return vid_actions


if __name__ == "__main__":
    vid_actions = process()
    x = 0
