import json
import os
import random
import statistics
from argparse import Namespace
from glob import glob

import av
import ipdb
import torch
import torch.utils.data as data
import torchvision
from torch.utils.data import ConcatDataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm

import py12transforms as T
from sampler import DistributedSampler, UniformClipSampler, RandomClipSampler, ConcatSampler

# normalize = T.Normalize(mean=get_mean(dataset='kinetics'),
#                         std=get_std())
normalize = T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                        std=[0.22803, 0.22145, 0.216989])
unnormalize = T.Unnormalize(mean=[0.43216, 0.394666, 0.37645],
                            std=[0.22803, 0.22145, 0.216989])
train_transform = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((128, 171)),
    T.RandomHorizontalFlip(),
    normalize,
    T.RandomCrop((112, 112))
])
test_transform = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((128, 171)),
    normalize,
    T.CenterCrop((112, 112))
])


def get_flow_histogram(flow):
    flow_magnitude = ((flow[..., 0] ** 2 + flow[..., 1] ** 2) ** 0.5).flatten()
    flow_magnitude[flow_magnitude > 99] = 99
    return torch.histc(flow_magnitude, min=0, max=100) / len(flow_magnitude)


histogram_flow_transform = lambda flow: get_flow_histogram(flow)


# CODE for 0,1,2 class imbalance (should go in init of dataset)
# y_tracker = Counter()
# fast_y_tracker = Counter()
# for video_idx, vid_clips in tqdm(enumerate(self.video_clips.clips), total=len(self.video_clips.clips)):
#     video_path = self.video_clips.video_paths[video_idx]
#     t_unit = av.open(video_path, metadata_errors='ignore').streams[0].time_base
#     t_fail = sorted(self.fails_data[os.path.splitext(os.path.basename(video_path))[0]]['t'])
#     t_fail = t_fail[len(t_fail) // 2]
#     for clip_idx, clip in enumerate(vid_clips):
#         start_pts = clip[0].item()
#         end_pts = clip[-1].item()
#         t_start = float(t_unit * start_pts)
#         t_end = float(t_unit * end_pts)
#         label = 0
#         if t_start <= t_fail <= t_end:
#             label = 1
#         elif t_start > t_fail:
#             label = 2
#         y_tracker[label] += 1
#         fast_y_tracker[self.video_clips.labels[video_idx][clip_idx]] += 1
# print({k: round(100 * v / sum(y_tracker.values()), 2) for k, v in y_tracker.items()})

class KineticsAndFails(VisionDataset):
    FLOW_FPS = 8

    def __init__(self, fails_path, kinetics_path, frames_per_clip, step_between_clips, fps, transform=None,
                 extensions=('.mp4',), video_clips=None, fails_only=False, val=False, balance_fails_only=False,
                 get_clip_times=False, fails_video_list=None, fns_to_remove=None, load_flow=False, flow_histogram=False,
                 fails_flow_path=None, all_fail_videos=False, selfsup_loss=None, clip_interval_factor=None,
                 labeled_fails=True, debug_dataset=False, anticipate_label=0, data_proportion=1, **kwargs):
        self.clip_len = frames_per_clip / fps
        self.clip_step = step_between_clips / fps
        self.clip_interval_factor = clip_interval_factor
        self.fps = fps
        self.t = transform
        self.load_flow = load_flow
        self.flow_histogram = flow_histogram
        self.video_clips = None
        self.fails_path = fails_path
        self.fails_flow_path = fails_flow_path
        self.selfsup_loss = selfsup_loss
        self.get_clip_times = get_clip_times
        self.anticipate_label = anticipate_label
        data_proportion = 1 if val else data_proportion
        if video_clips:
            self.video_clips = video_clips
        else:
            assert fails_path is None or fails_video_list is None
            video_list = fails_video_list or glob(os.path.join(fails_path, '**', '*.mp4'), recursive=True)
            if not fails_only:
                kinetics_cls = torch.load("PATH/TO/kinetics_classes.pt")
                kinetics_dist = torch.load("PATH/TO/dist.pt")
                s = len(video_list)
                for i, n in kinetics_dist.items():
                    n *= s
                    video_list += sorted(
                        glob(os.path.join(kinetics_path, '**', kinetics_cls[i], '*.mp4'), recursive=True))[
                    :round(n)]
            self.video_clips = VideoClips(video_list, frames_per_clip, step_between_clips, fps)
        with open("PATH/TO/borders.json") as f:
            self.fails_borders = json.load(f)
        with open("PATH/TO/all_mturk_data.json") as f:
            self.fails_data = json.load(f)
        self.fails_only = fails_only
        self.t_from_clip_idx = lambda idx: (
            (step_between_clips * idx) / fps, (step_between_clips * idx + frames_per_clip) / fps)
        if not balance_fails_only:  # no support for recompute clips after balance calc yet
            self.video_clips.compute_clips(frames_per_clip, step_between_clips, fps)
        if video_clips is None and fails_only and labeled_fails:
            # if True:
            if not all_fail_videos:
                idxs = []
                for i, video_path in enumerate(self.video_clips.video_paths):
                    video_path = os.path.splitext(os.path.basename(video_path))[0]
                    if video_path in self.fails_data:
                        idxs.append(i)
                self.video_clips = self.video_clips.subset(idxs)
            # if not val and balance_fails_only:  # balance dataset
            # ratios = {0: 0.3764, 1: 0.0989, 2: 0.5247}
            self.video_clips.labels = []
            self.video_clips.compute_clips(frames_per_clip, step_between_clips, fps)
            for video_idx, vid_clips in tqdm(enumerate(self.video_clips.clips), total=len(self.video_clips.clips)):
                video_path = self.video_clips.video_paths[video_idx]
                if all_fail_videos and os.path.splitext(os.path.basename(video_path))[0] not in self.fails_data:
                    self.video_clips.labels.append([-1 for _ in vid_clips])
                    continue
                t_unit = av.open(video_path, metadata_errors='ignore').streams[0].time_base
                t_fail = sorted(self.fails_data[os.path.splitext(os.path.basename(video_path))[0]]['t'])
                t_fail = t_fail[len(t_fail) // 2]
                if t_fail < 0 or not 0.01 <= statistics.median(
                        self.fails_data[os.path.splitext(os.path.basename(video_path))[0]]['rel_t']) <= 0.99 or \
                        self.fails_data[os.path.splitext(os.path.basename(video_path))[0]]['len'] < 3.2 or \
                        self.fails_data[os.path.splitext(os.path.basename(video_path))[0]]['len'] > 30:
                    self.video_clips.clips[video_idx] = torch.Tensor()
                    self.video_clips.resampling_idxs[video_idx] = torch.Tensor()
                    self.video_clips.labels.append([])
                    continue
                prev_label = 0
                first_one_idx = len(vid_clips)
                first_two_idx = len(vid_clips)
                for clip_idx, clip in enumerate(vid_clips):
                    start_pts = clip[0].item()
                    end_pts = clip[-1].item()
                    t_start = float(t_unit * start_pts)
                    t_end = float(t_unit * end_pts)
                    label = 0
                    if t_start <= t_fail <= t_end:
                        label = 1
                    elif t_start > t_fail:
                        label = 2
                    if label == 1 and prev_label == 0:
                        first_one_idx = clip_idx
                    elif label == 2 and prev_label == 1:
                        first_two_idx = clip_idx
                        break
                    prev_label = label
                self.video_clips.labels.append(
                    [0 for i in range(first_one_idx)] + [1 for i in range(first_one_idx, first_two_idx)] +
                    [2 for i in range(first_two_idx, len(vid_clips))])
                if balance_fails_only and not val:
                    balance_idxs = []
                    counts = (first_one_idx, first_two_idx - first_one_idx, len(vid_clips) - first_two_idx)
                    offsets = torch.LongTensor([0] + list(counts)).cumsum(0)[:-1].tolist()
                    ratios = (1, 0.93, 1 / 0.93)
                    labels = (0, 1, 2)
                    lbl_mode = max(labels, key=lambda i: counts[i])
                    for i in labels:
                        if i != lbl_mode and counts[i] > 0:
                            n_to_add = round(counts[i] * ((counts[lbl_mode] * ratios[i] / counts[i]) - 1))
                            tmp = list(range(offsets[i], counts[i] + offsets[i]))
                            random.shuffle(tmp)
                            tmp_bal_idxs = []
                            while len(tmp_bal_idxs) < n_to_add:
                                tmp_bal_idxs += tmp
                            tmp_bal_idxs = tmp_bal_idxs[:n_to_add]
                            balance_idxs += tmp_bal_idxs
                    if not balance_idxs:
                        continue
                    t = torch.cat((vid_clips, torch.stack([vid_clips[i] for i in balance_idxs])))
                    self.video_clips.clips[video_idx] = t
                    vid_resampling_idxs = self.video_clips.resampling_idxs[video_idx]
                    try:
                        t = torch.cat(
                            (vid_resampling_idxs, torch.stack([vid_resampling_idxs[i] for i in balance_idxs])))
                        self.video_clips.resampling_idxs[video_idx] = t
                    except IndexError:
                        pass
                    self.video_clips.labels[-1] += [self.video_clips.labels[-1][i] for i in balance_idxs]
            clip_lengths = torch.as_tensor([len(v) for v in self.video_clips.clips])
            self.video_clips.cumulative_sizes = clip_lengths.cumsum(0).tolist()
        fns_removed = 0
        if fns_to_remove and not val:
            for i, video_path in enumerate(self.video_clips.video_paths):
                if fns_removed > len(self.video_clips.video_paths)//4:
                    break
                video_path = os.path.splitext(os.path.basename(video_path))[0]
                if video_path in fns_to_remove:
                    fns_removed += 1
                    self.video_clips.clips[i] = torch.Tensor()
                    self.video_clips.resampling_idxs[i] = torch.Tensor()
                    self.video_clips.labels[i] = []
            clip_lengths = torch.as_tensor([len(v) for v in self.video_clips.clips])
            self.video_clips.cumulative_sizes = clip_lengths.cumsum(0).tolist()
            if kwargs['local_rank'] <= 0:
                print(f'removed videos from {fns_removed} out of {len(self.video_clips.video_paths)} files')
        # if not fails_path.startswith("PATH/TO/scenes"):
        for i, p in enumerate(self.video_clips.video_paths):
            self.video_clips.video_paths[i] = p.replace("PATH/TO/scenes",
                                                        os.path.dirname(fails_path))
        self.debug_dataset = debug_dataset
        if debug_dataset:
            # self.video_clips = self.video_clips.subset([0])
            pass
        if data_proportion < 1:
            rng = random.Random()
            rng.seed(23719)
            lbls = self.video_clips.labels
            subset_idxs = rng.sample(range(len(self.video_clips.video_paths)), int(len(self.video_clips.video_paths)*data_proportion))
            self.video_clips = self.video_clips.subset(subset_idxs)
            self.video_clips.labels = [lbls[i] for i in subset_idxs]

    def trim_borders(self, img, fn):
        l, r = self.fails_borders[os.path.splitext(os.path.basename(fn))[0]]
        w = img.shape[2]  # THWC
        if l > 0 and r > 0:
            img = img[:, :, round(w * l):round(w * r)]
        return img

    def __len__(self):
        return self.video_clips.num_clips()

    def compute_clip_times(self, video_idx, clip_idx):
        video_path = self.video_clips.video_paths[video_idx]
        video_path = os.path.join(self.fails_path, os.path.sep.join(video_path.rsplit(os.path.sep, 2)[-2:]))
        clip_pts = self.video_clips.clips[video_idx][clip_idx]
        start_pts = clip_pts[0].item()
        end_pts = clip_pts[-1].item()
        t_unit = av.open(video_path, metadata_errors='ignore').streams[0].time_base
        t_start = float(t_unit * start_pts)
        t_end = float(t_unit * end_pts)
        return t_start, t_end

    def __getitem__(self, idx):
        if self.load_flow:
            video_idx, clip_idx = self.video_clips.get_clip_location(idx)
            video_path = self.video_clips.video_paths[video_idx]
            video_path = os.path.join(self.fails_path, os.path.sep.join(video_path.rsplit(os.path.sep, 2)[-2:]))
            label = self.video_clips.labels[video_idx][clip_idx]
            flow_path = os.path.join(self.fails_flow_path,
                                     os.path.sep.join(os.path.splitext(video_path)[0].rsplit(os.path.sep, 2)[-2:]))
            t_start, t_end = self.compute_clip_times(video_idx, clip_idx)
            frame_start = round(t_start * self.FLOW_FPS)
            n_frames = round(self.clip_len * self.FLOW_FPS)
            flow = []
            for frame_i in range(frame_start, frame_start + n_frames):
                frame_fn = os.path.join(flow_path, f'{frame_i:06}.flo')
                try:
                    flow.append(torch.load(frame_fn, map_location=torch.device('cpu')).permute(1, 2, 0).data.numpy())
                except:
                    pass
            while len(flow) < n_frames:
                flow += flow
            flow = flow[:n_frames]
            flow = torch.Tensor(flow)
            flow = self.trim_borders(flow, video_path)
            if self.t is not None:
                flow = self.t(flow)
            return flow, label, (flow_path, t_start, t_end)
        else:
            video_idx, clip_idx = self.video_clips.get_clip_location(idx)
            if self.anticipate_label:
                assert not self.selfsup_loss, 'no anticipation with self supervision'
                video_path = self.video_clips.video_paths[video_idx]
                label = self.video_clips.labels[video_idx][clip_idx]
                idx -= round(self.anticipate_label / self.clip_step)
                new_video_idx, new_clip_idx = self.video_clips.get_clip_location(idx)
                video, *_ = self.video_clips.get_clip(idx)
                video = self.trim_borders(video, video_path)
                if self.t is not None:
                    video = self.t(video)
                new_t_start, new_t_end = self.compute_clip_times(new_video_idx, new_clip_idx)
                old_t_start, old_t_end = self.compute_clip_times(video_idx, clip_idx)
                if new_video_idx != video_idx or new_t_start > old_t_start:
                    label = -1
                return video, label, (video_path, new_t_start, new_t_end, [])

            video, audio, info, video_idx = self.video_clips.get_clip(idx)
            video_path = self.video_clips.video_paths[video_idx]
            # print(video_path)
            try:
                label = self.video_clips.labels[video_idx][clip_idx]
                # if self.anticipate_label:
                #     video_path = self.video_clips.video_paths[video_idx]
                #     t_fail = statistics.median(self.fails_data[os.path.splitext(os.path.basename(video_path))[0]]['t'])
                #     t_start, t_end = self.compute_clip_times(video_idx, clip_idx)
                #     t_start += self.anticipate_label
                #     t_end += self.anticipate_label
                #     label = 0
                #     if t_start <= t_fail <= t_end:
                #         label = 1
                #     elif t_start > t_fail:
                #         label = 2
            except:
                label = -1

            if label == 0 or self.fails_only: video = self.trim_borders(video, video_path)
            if self.debug_dataset:
                pass
                # video[:] = 0
                # video[..., 0] = 255
            if self.t is not None:
                video = self.t(video)

            t_start = t_end = -1
            if self.get_clip_times:
                t_start, t_end = self.compute_clip_times(video_idx, clip_idx)

            other = []

            if self.selfsup_loss == 'pred_middle' or self.selfsup_loss == 'sort' or self.selfsup_loss == 'ctc':
                k = round(self.clip_len / self.clip_step * self.clip_interval_factor)
                video_l = [video]
                try:
                    pvideo, paudio, pinfo, pvideo_idx = self.video_clips.get_clip(idx - k)
                except:
                    pvideo_idx = -1
                try:
                    nvideo, naudio, ninfo, nvideo_idx = self.video_clips.get_clip(idx + k)
                except:
                    nvideo_idx = -1
                t_start, _ = self.compute_clip_times(*self.video_clips.get_clip_location(idx))
                try:
                    p_t_start, _ = self.compute_clip_times(*self.video_clips.get_clip_location(idx - k))
                except:
                    p_t_start = 1000000000
                try:
                    n_t_start, _ = self.compute_clip_times(*self.video_clips.get_clip_location(idx + k))
                except:
                    n_t_start = -1000000000
                # if pvideo_idx == video_idx:
                #     assert p_t_start < t_start, f"{t_start} <= prev video time {p_t_start}"
                # if nvideo_idx == video_idx:
                #     assert t_start < n_t_start, f"{t_start} >= next video time {n_t_start}"
                if pvideo_idx == video_idx and p_t_start < t_start:
                    pvideo = self.trim_borders(pvideo, video_path)
                    if self.t is not None:
                        pvideo = self.t(pvideo)
                    video_l.insert(0, pvideo)
                else:
                    video_l.insert(0, torch.full_like(video, -1))
                if nvideo_idx == video_idx and t_start < n_t_start:
                    nvideo = self.trim_borders(nvideo, video_path)
                    if self.t is not None:
                        nvideo = self.t(nvideo)
                    video_l.append(nvideo)
                else:
                    video_l.append(torch.full_like(video, -1))
                video_l = torch.stack(video_l)
                video = video_l
                other = [nvideo_idx == video_idx and pvideo_idx == video_idx]

            if self.selfsup_loss == 'fps':
                other = [self.fps]

            other.append(idx)

            return video, label, (video_path, t_start, t_end, *other)


def get_video_loader(**kwargs):
    args = Namespace(**kwargs)
    args.fails_video_list = None
    if args.val:
        args.fails_path = os.path.join(args.fails_path, 'val')
        args.kinetics_path = os.path.join(args.kinetics_path, 'val')
    else:
        args.fails_path = os.path.join(args.fails_path, 'train')
        args.kinetics_path = os.path.join(args.kinetics_path, 'train')
    if args.fails_action_split:
        args.fails_path = None
        args.fails_video_list = torch.load(os.path.join(args.dataset_path, 'fails_action_split.pth'))[
            'val' if args.val else 'train']
    DEBUG = False
    datasets = []
    samplers = []
    for fps in args.fps_list:
        clips = None
        args.fps = fps
        args.step_between_clips = round(args.step_between_clips_sec * fps)
        cache_path = os.path.join(args.dataset_path,
                                  '{3}{2}{1}{0}{4}_videoclips.pth'.format('val' if args.val else 'train',
                                                                          f'fails_only_{"all_" if args.all_fail_videos else ""}' if args.fails_only else '',
                                                                          'bal_' if (
                                                                                  args.balance_fails_only and not DEBUG) else '',
                                                                          'actions_' if args.fails_action_split else '',
                                                                          f'{args.fps}fps'))
        if args.cache_dataset and os.path.exists(cache_path):
            clips = torch.load(cache_path)
            if args.local_rank <= 0:
                print(f'Loaded dataset from {cache_path}')
        fns_to_remove = None
        if args.flow_histogram:
            args.transform = histogram_flow_transform
        if args.remove_fns == 'action_based':
            fns_to_remove = torch.load("PATH/TO/fails_remove_fns.pth")['action_remove']
        elif args.remove_fns == 'random':
            fns_to_remove = torch.load("PATH/TO/fails_remove_fns.pth")['random_remove']
        dataset = KineticsAndFails(video_clips=clips, fns_to_remove=fns_to_remove, **vars(args))
        if not args.val:
            print(f'Dataset contains {len(dataset)} items')
        if args.cache_dataset and args.local_rank <= 0 and clips is None:  # and not args.fails_only
            torch.save(dataset.video_clips, cache_path)
        if args.val:
            sampler = UniformClipSampler(dataset.video_clips,
                                         1000000 if args.sample_all_clips else args.clips_per_video)
        else:
            sampler = RandomClipSampler(dataset.video_clips, 1000000 if args.sample_all_clips else args.clips_per_video)
        datasets.append(dataset)
        samplers.append(sampler)
    if len(args.fps_list) > 1:
        dataset = ConcatDataset(datasets)
        sampler = ConcatSampler(samplers)
    else:
        dataset = datasets[0]
        sampler = samplers[0]
    if args.local_rank != -1:
        sampler = DistributedSampler(sampler)
    return data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        # collate_fn=dataset.collate_fn,
        sampler=sampler,
        pin_memory=True,
        drop_last=False
    )
