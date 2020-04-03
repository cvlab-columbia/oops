import json
from collections import Counter, defaultdict
from glob import glob

from joblib import Parallel, delayed
from tqdm import tqdm


def process():
    fns = glob(
        "PATH/TO/fails_kinetics_*.json")

    data = []
    for fn in fns:
        with open(fn) as f:
            data.extend(json.load(f))

    vid_actions = defaultdict(lambda: defaultdict(int))

    def loop(datum):
        p_vec, fn, _ = datum
        dirname = fn
        argmax = max(range(len(p_vec)), key=lambda i: p_vec[i])
        vid_actions[dirname][argmax] += 1

    parallel_result = Parallel(n_jobs=40, require='sharedmem')(delayed(loop)(d) for d in tqdm(data))

    most_common = []

    for vid, act_freqs in tqdm(vid_actions.items()):
        # vid_actions[vid] = Counter(act_freqs).most_common(1)[0][0]
        most_common.append(Counter(act_freqs).most_common(1)[0][0])

    return Counter(most_common)


if __name__ == "__main__":
    with open("PATH/TO/places_dist.json", 'w') as f:
        json.dump(process(), f)
