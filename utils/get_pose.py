import ipdb
import numpy as np
import os
import subprocess
from multiprocessing import Process, Queue, Lock
from moviepy.video.io.VideoFileClip import VideoFileClip as Video
import skvideo.measure as skv
from glob import glob
import csv
from tqdm import tqdm


def job(fn, gpuid):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpuid)
    result = subprocess.run(['python', "PATH/TO/video_demo.py', '--video', fn, '--outdir",
        "PATH/TO/scenes_pose', '--save_video', '--sp', '--nThreads", str(1)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result

def worker(inq, outq, lock, wid, ngpus):
    for item in iter(inq.get, None):
        outq.put(job(item, wid%ngpus))
        #outq.put(0)
        #with lock:
            #print('Task done')


if __name__ == "__main__":
    fns = glob("PATH/TO/*.mp4")
    inq = Queue()
    outq = Queue()
    lock = Lock()
    nproc = 12
    ngpus = 6
    output = []
    for fn in fns:
        inq.put(fn)
    for i in range(nproc):
        inq.put(None)
    for i in range(nproc):
        Process(target=worker, args=(inq, outq, lock, i, ngpus)).start()
    for fn in tqdm(fns):
        output.append(outq.get())
    ipdb.set_trace()
