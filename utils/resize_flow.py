import json
import ipdb
import numpy as np
import os
import subprocess
from multiprocessing import Process, Queue, Lock
from moviepy.video.io.VideoFileClip import VideoFileClip as Video
from glob import glob
import csv
from tqdm import tqdm
import cvbase.optflow.io as flowlib
import torch.nn.functional as F
import torch

def job(fn, gpuid):
    flow = flowlib.read_flow(fn)
    newflow = F.interpolate(torch.tensor(flow).to('cuda:{0}'.format(gpuid)).permute(2,0,1).unsqueeze(0), scale_factor=224/min(flow.shape[:-1])).squeeze()
    newfn = fn.replace('scenes_flow', 'scenes_flow_small')
    os.makedirs(os.path.dirname(newfn), exist_ok=True)
    torch.save(newflow.cpu(),newfn)
    return 0

def worker(inq, outq, lock, wid, ngpus):
    for item in iter(inq.get, None):
        outq.put(job(item, wid%ngpus))
        #outq.put(0)
        #with lock:
            #print('Task done')


if __name__ == "__main__":
    #fns = sorted(glob("PATH/TO/*.flo"))
    with open('flo_fns.json') as f:
        fns=json.load(f)
    fns=fns
    inq = Queue()
    outq = Queue()
    lock = Lock()
    nproc = 40
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
    #ipdb.set_trace()
