import ipdb
import numpy as np
import os
from multiprocessing import Process, Queue, Lock
from moviepy.video.io.VideoFileClip import VideoFileClip as Video
import skvideo.measure as skv
from glob import glob
import csv
from tqdm import tqdm


def job(item):
    fn, indir, outdir = item
    outdir = os.path.splitext(fn.replace(indir, outdir))[0]
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
        vid = Video(fn)
        vid.write_images_sequence(os.path.join(outdir, '%06d.bmp'), fps=8, verbose=False, logger=None)
        vid.close()

def worker(inq, outq, lock):
    for item in iter(inq.get, None):
        job(item)
        outq.put(0)

if __name__ == "__main__":
    inq = Queue()
    outq = Queue()
    lock = Lock()
    nproc = 40
    #basepath = "/local/vondrick/datasets/fails/scenes"
    basepath = "YOUR PATH HERE"
    outdir = "YOUR PATH HERE"
    data=glob(os.path.join(basepath, '**/*.mp4'), recursive=True)
    for item in data:
        inq.put((item, basepath, outdir))
    for i in range(nproc):
        inq.put(None)
    for i in range(nproc):
        Process(target=worker, args=(inq, outq, lock)).start()
    for item in tqdm(data):
        outq.get()
