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
# from spatial_transforms import Normalize, Compose, ToTensor, Scale, RandomScaleCrop
from torchvision.datasets.folder import default_loader
import json

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

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image



def trim_borders(img, fn):
    l, r = border_file[os.path.basename(os.path.dirname(fn))]
    w, h = img.size
    if l > 0 and r > 0:
        img = img.crop((w*l, 0, w*r, h))
    return img


if __name__ == "__main__":
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

    basepath = '/local/vondrick/datasets/fails/scenes_small'
    with open("PATH/TO/borders.json") as f:
        border_file = json.load(f)
    # img_xform = Compose([RandomScaleCrop((1,), center=True), Resize((args.sample_size, args.sample_size)), ToTensor(
    # ), Normalize(get_mean(dataset='kinetics'), get_std()), torchvision.transforms.Lambda(lambda img: img.unsqueeze(1))])

    # th architecture to use
    arch = 'resnet18'

    # load the pre-trained weights
    model_file = '%s_places365.pth.tar' % arch
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # load the image transformer
    centre_crop = trn.Compose([
        trn.Resize((256, 256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load the class label
    file_name = 'categories_places365.txt'
    if not os.access(file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    xform = torchvision.transforms.Lambda(
        lambda data: (centre_crop(trim_borders(data[0], data[1])), data[1]))

    def loaderfn(fn): return (default_loader(fn), fn)
    traindata = torchvision.datasets.ImageFolder(os.path.join(
        basepath, 'train', 'train'), transform=xform, loader=loaderfn)
    testdata = torchvision.datasets.ImageFolder(os.path.join(
        basepath, 'val', 'val'), transform=xform, loader=loaderfn)
    loader = get_loader(data.ConcatDataset([traindata, testdata]))
    # model = ActionClassifier(args)
    # model.to(device)

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

    # checkpoint = torch.load(
    #     '/local/vondrick/dave/fails/checkpoint_kinetics/model_best.pt.tar', map_location=device)
    # model.load_state_dict(checkpoint['state_dict'])

    results = []

    os.makedirs(args.save_path, exist_ok=True)
    fn_cnt = 0

    with torch.no_grad():
        for (batch, fns), _ in tqdm(loader, disable=args.local_rank > 0):
            out = model(batch.to(device))
            for p_vec, fn in zip(out, fns):
                results.append([p_vec.tolist(), os.path.dirname(fn), fn_cnt])
                fn_cnt += 1

    with open(os.path.join(args.save_path, 'fails_kinetics_features_{0}.json'.format(args.local_rank)), 'w') as f:
        json.dump(results, f)
