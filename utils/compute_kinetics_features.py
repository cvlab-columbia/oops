import torch
import ipdb
# from torchvision.models.video import r3d_18
from torchvision.datasets import ImageFolder
import torchvision
import torch.utils.data.distributed as distrib
from tqdm import tqdm
import torch.utils.data as data
import os
import torch.nn as nn
import sys
import torch
import argparse
import cv2
from types import SimpleNamespace
from spatial_transforms import Normalize, Compose, ToTensor, Scale, RandomScaleCrop
from torchvision.datasets.folder import default_loader
import json
sys.path.append("PATH/TO/fails")
sys.path.append("PATH/TO/cnns")


def get_mean(norm_value=255, dataset='activitynet'):
    assert dataset in ['activitynet', 'kinetics']

    if dataset == 'activitynet':
        return [
            114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value
        ]
    elif dataset == 'kinetics':
        # Kinetics (10 videos for each class)
        return [
            110.63666788 / norm_value, 103.16065604 / norm_value,
            96.29023126 / norm_value
        ]


def get_std(norm_value=255):
    # Kinetics (10 videos for each class)
    return [
        38.7568578 / norm_value, 37.88248729 / norm_value,
        40.02898126 / norm_value
    ]


class Resize():
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)


class ActionClassifier(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model, _ = model3d.generate_model(SimpleNamespace(**{
            'model': 'resnext',
            'model_depth': 101,
            'arch': 'resnext-101',
            'resnet_shortcut': 'B',
            'resnext_cardinality': 32,
            'n_finetune_classes': cfg.n_kinetics_classes,
            'n_classes': 600,
            'sample_size': cfg.sample_size,
            'sample_duration': cfg.sample_duration,
            'pretrain_path': None,
            'no_cuda': False,
            'ft_begin_index': 0
        }))
        self.features = model.module
        self.fc = self.features.fc
        self.features.fc = nn.Sequential()

    def forward(self, vids):
        features = self.features(vids).view(len(vids), -1)
        return self.fc(features)


def get_loader(dataset):
    if args.local_rank != -1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=False)
    else:
        sampler = None
    return data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=sampler,
        pin_memory=True
    )


def trim_borders(img, fn):
    l, r = border_file[os.path.basename(os.path.dirname(fn))]
    w, h = img.size
    if l > 0 and r > 0:
        img = img.crop((w*l, 0, w*r, h))
    return img


if __name__ == "__main__":
    from cnns import model as model3d
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=4, type=int, metavar='N',
                        help='batchsize')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--learning_rate",
                        default=1e-3,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--save_path', default='test_results')
    parser.add_argument('--opt_level', default="O2")
    parser.add_argument('--sample_size', type=int, default=112)
    parser.add_argument('--sample_duration', type=int, default=16)
    parser.add_argument('--n_kinetics_classes', type=int, default=600)
    args = parser.parse_args()
    args.clip_size = args.sample_duration
    args.cuda = torch.cuda.is_available()

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')

    basepath = "PATH/TO/scenes_small"
    with open("PATH/TO/borders.json") as f:
        border_file = json.load(f)
    img_xform = Compose([RandomScaleCrop((1,), center=True), Resize((args.sample_size, args.sample_size)), ToTensor(
    ), Normalize(get_mean(dataset='kinetics'), get_std()), torchvision.transforms.Lambda(lambda img: img.unsqueeze(1))])

    xform = torchvision.transforms.Lambda(
        lambda data: (img_xform(trim_borders(data[0], data[1])), data[1]))

    def loaderfn(fn): return (default_loader(fn), fn)
    traindata = torchvision.datasets.ImageFolder(os.path.join(
        basepath, 'train', 'train'), transform=xform, loader=loaderfn)
    testdata = torchvision.datasets.ImageFolder(os.path.join(
        basepath, 'val', 'val'), transform=xform, loader=loaderfn)
    loader = get_loader(data.ConcatDataset([traindata, testdata]))
    model = ActionClassifier(args)
    model.to(device)

    if args.fp16:
        try:
            #from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
            from apex import amp, optimizers
        except ImportError:
            raise ImportError('apex not working')

        if args.loss_scale == 0:
            args.loss_scale = None

        model = amp.initialize(
            model, opt_level=args.opt_level, loss_scale=args.loss_scale)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except:
            from torch.nn.parallel import DistributedDataParallel as DDP
            print('Using PyTorch DDP - could not find Apex')
        model = DDP(model, delay_allreduce=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    checkpoint = torch.load(
        "PATH/TO/model_best.pt.tar", map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    results = []

    os.makedirs(args.save_path, exist_ok=True)
    fn_cnt = 0

    with torch.no_grad():
        for (batch, fns), _ in tqdm(loader, disable=args.local_rank > 0):
            groupedbatch = []
            groupedfns = []
            prev_fn = None
            for img, fn in zip(batch, fns):
                fn = os.path.dirname(fn)
                if fn == prev_fn:
                    groupedbatch[-1].append(img)
                else:
                    groupedbatch.append([img])
                    groupedfns.append(fn)
                    prev_fn = fn
            newbatch = []
            newfns = []
            for group, fn in zip(groupedbatch, groupedfns):
                while len(group) > args.clip_size:
                    clip = group[:args.clip_size]
                    group = group[args.clip_size:]
                    newbatch.append(clip)
                    newfns.append(fn)
                while len(group) < args.clip_size:
                    group.append(group[-1])
                newbatch.append(group)
                newfns.append(fn)
            batch = torch.stack([torch.cat(clip, dim=1) for clip in newbatch])
            out = model(batch.to(device))
            for p_vec, fn in zip(out, newfns):
                results.append([p_vec.tolist(), fn+'_{0}'.format(fn_cnt)])
                fn_cnt += 1

    with open(os.path.join(args.save_path, 'fails_kinetics_features_{0}.json'.format(args.local_rank)), 'w') as f:
        json.dump(results, f)
