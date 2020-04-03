import ipdb
import numpy as np
from multiprocessing import Process, Queue, Lock
from moviepy.video.io.VideoFileClip import VideoFileClip as Video
import skvideo.measure as skv
from glob import glob
import csv
from tqdm import tqdm


def proc_out(out, out_fn, min_scene_len=1):
    data = [('ytid', 'start', 'end', 'split')]
    split = 0.8
    for idx, (fn, scenes) in enumerate(out.items()):
        scenes = scenes.astype(np.float64)
        fps = Video(fn).fps
        scenes /= fps
        dirty = True
        while dirty:
            dirty = False
            for i in range(len(scenes)-1):
                d=scenes[i+1]-scenes[i]
                if d < min_scene_len:
                    scenes=np.delete(scenes,i+1)
                    dirty = True
                    break
        for i in range(len(scenes)-1):
            data.append(
                (fn, scenes[i], scenes[i+1], 'train' if idx < (len(out)*split) else 'test'))
    with open(out_fn, 'w') as f:
        csv.writer(f).writerows(data)


def lookahead(iterable):
    # Get an iterator and pull the first value.
    it = iter(iterable)
    last = next(it)
    # Run the iterator to exhaustion (starting from the second value).
    for val in it:
        # Report the *previous* value (more to come).
        yield last, False
        last = val
    # Report the last value.
    yield last, True


def job(fn, wid):
    vid = Video(fn)
    vidit = vid.iter_frames()
    data = None
    ret = [np.array([0])]
    i = 0
    buf = 100
    cnt = 0
    for fr, last in tqdm(lookahead(vidit), total=int(vid.duration*vid.fps), position=wid, desc='Thread {0}'.format(wid), leave=False, disable=True):
        if data is None:
            data = np.zeros((buf, *fr.shape), dtype=np.uint8)
        data[i] = fr
        i += 1
        if i == buf or last:
            data = data[:i]  # when last=True i may be <buf
            res = skv.scenedet(data, method='edges', parameter1=0.7)
            ret.append(res[1:]+(buf-1)*cnt)
            cnt += 1
            if not last:
                data_ = np.zeros((buf, *fr.shape), dtype=np.uint8)
                data_[0] = data[-1]
                i = 1
                data = data_
    vid.close()
    return np.concatenate(ret)


def worker(inq, outq, lock, wid):
    for fn in iter(inq.get, None):
        outq.put((fn, job(fn, wid)))
        with lock:
            print('Task done')


if __name__ == "__main__":
    import ipdb
    f = glob('*.webm')+glob('*.mp4')
    #f = ['Best Fails of Week 3 June 2016 _ FailArmy.mp4']
    inq = Queue()
    outq = Queue()
    lock = Lock()
    nproc = 96
    #nproc = 1
    try:
        data=np.load('scene_frames.npy', allow_pickle=True).item()
        f=list(set(f) - set(data.keys()))
        data=list(data.items())
        #ipdb.set_trace()
    except:
        data = []
    print(len(f),'tasks remaining')
    for fn in f:
        inq.put(fn)
    for i in range(nproc):
        inq.put(None)
    for i in range(nproc):
        Process(target=worker, args=(inq, outq, lock, i)).start()
    for fn in f:
        data.append(outq.get())
        with lock:
            print('Harvested', len(data))
            np.save('scene_frames', dict(data))
    data = dict(data)
    np.save('scene_frames_final', data)
    import ipdb
    ipdb.set_trace()
