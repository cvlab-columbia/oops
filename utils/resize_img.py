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
from torchvision.transforms import Resize
from PIL import Image

def job(fn, gpuid):
    #flow = flowlib.read_flow(fn)
    #newflow = F.interpolate(torch.tensor(flow).to('cuda:{0}'.format(gpuid)).permute(2,0,1).unsqueeze(0), scale_factor=224/min(flow.shape[:-1])).squeeze()
    newfn = fn.replace('scenes', 'scenes_small')
    #if os.path.exists(newfn): return 0
    img = Image.open(fn).convert('RGB')
    os.makedirs(os.path.dirname(newfn), exist_ok=True)
    img = Resize(224)(img)
    img.save(newfn)
    return 0

def worker(inq, outq, lock, wid, ngpus):
    for item in iter(inq.get, None):
        outq.put(job(item, wid%ngpus))
        #outq.put(0)
        #with lock:
            #print('Task done')


if __name__ == "__main__":
    #fns = sorted(glob("PATH/TO/*.flo"))
    with open('bmp_fns.json') as f:
        fns=json.load(f)
    #fns=fns[:1]
    dirs=['28 Best Skateboard Fail Nominees - FailArmy Hall of Fame (August 2017)1',
            '28 Best Skateboard Fail Nominees - FailArmy Hall of Fame (August 2017)10',
            '28 Best Skateboard Fail Nominees - FailArmy Hall of Fame (August 2017)12',
            '28 Best Skateboard Fail Nominees - FailArmy Hall of Fame (August 2017)13',
            '28 Best Skateboard Fail Nominees - FailArmy Hall of Fame (August 2017)14',
            "Best Fails of the Week 1 March 2016 _ 'WTF Was that!' by FailArmy57",
            'Best Fails of the Week 2 February 2016 _ FailArmy16']
    ipdb.set_trace()
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
