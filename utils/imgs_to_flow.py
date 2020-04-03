import ipdb
import numpy as np
import os
import subprocess
from multiprocessing import Process, Queue, Lock
from moviepy.video.io.VideoFileClip import VideoFileClip as Video
from glob import glob
import csv
from tqdm import tqdm


def job(path, gpuid):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpuid)
    result = subprocess.run(['python', '/local/vondrick/shared/flownet2-pytorch/main.py', '--inference', '--model', 'FlowNet2', '--save_flow', '--inference_dataset', 'ImagesFromFolder', '--inference_dataset_root', path, '--resume', '/local/vondrick/shared/flownet2-pytorch/checkpoint/FlowNet2_checkpoint.pth.tar', '--save', path, '--name', 'flow'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result

def worker(inq, outq, lock, wid, ngpus):
    for item in iter(inq.get, None):
        outq.put(job(item, wid%ngpus))
        #outq.put(0)
        #with lock:
            #print('Task done')


if __name__ == "__main__":
    fns = sorted(glob('/proj/vondrick/datasets/fails/scenes/*/*/*'))
    inq = Queue()
    outq = Queue()
    lock = Lock()
    nproc = 24
    ngpus = 8
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
