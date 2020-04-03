import ipdb
import numpy as np
import os
from multiprocessing import Process, Queue, Lock
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import skvideo.measure as skv
from glob import glob
import csv
from tqdm import tqdm
from cvbase.optflow.visualize import flow2rgb
from cvbase.optflow.io import read_flow
import torch


def make_uint8(im):
    return (255 * im).astype('uint8')


def normalize_flows(vid, newmin=-1, newmax=1):
    minval = vid.min()
    maxval = vid.max()
    return (newmax-newmin)/(maxval-minval)*(vid-minval)+newmin


def job(item):
    fn = item
    outpath = os.path.join(fn, 'flow.mp4')
    if not os.path.exists(outpath):
        flows = torch.stack([torch.from_numpy(read_flow(_))
                             for _ in glob(os.path.join(fn, '*.flo'))])
        flows = list(normalize_flows(flows))
        flows = list(flows)
        rgb_flows = [make_uint8(flow2rgb(_.numpy())) for _ in flows]
        vid = ImageSequenceClip(
            rgb_flows, fps=8)
        vid.write_videofile(outpath, fps=8, verbose=False, logger=None)
        vid.close()


def worker(inq, outq, lock):
    for item in iter(inq.get, None):
        job(item)
        outq.put(0)


if __name__ == "__main__":
    inq = Queue()
    outq = Queue()
    lock = Lock()
    nproc = 3
    basepath = "YOUR PATH HERE"
    data = glob(os.path.join(basepath, '*'))
    for item in data:
        inq.put((item))
    for i in range(nproc):
        inq.put(None)
    for i in range(nproc):
        Process(target=worker, args=(inq, outq, lock)).start()
    c = 0
    for item in tqdm(data):
        outq.get()
