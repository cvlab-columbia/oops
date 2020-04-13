import argparse
import torch
import os
from dataloader import get_video_loader, train_transform, test_transform, unnormalize
import torch.nn.functional as F
from nets import FlowPredictor
from tqdm import trange, tqdm
from captum.attr import LayerAttribution
from captum.attr._core.grad_cam import LayerGradCam
import torch.distributed as distrib
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import shutil
import sys
from PIL import Image
from cvbase.optflow.visualize import flow2rgb
from torchvision.models.video import r3d_18
import csv
import time
from utils import AverageMeter
import torch.nn as nn
import ipdb
from types import SimpleNamespace
from utils.kinetics_utils import *

sys.path.append("PATH/TO/fails")
sys.path.append("PATH/TO/cnns")


def make_uint8(im):
    return (255 * im).astype('uint8')


def train(loader):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    end = time.time()

    loss_fn = nn.CrossEntropyLoss()

    with tqdm(loader, desc='Train batch iteration', disable=args.local_rank > 0) as t:
        for batch_idx, (xs, ys, _) in enumerate(t):
            data_time.update(time.time() - end)
            # if args.local_rank <= 0: ipdb.set_trace()
            xs = xs.to(device)
            ys = ys.to(device)

            y_hats = model(xs)
            loss = loss_fn(y_hats, ys)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            losses.update(loss.item(), len(ys))
            accs.update(accuracy(y_hats, ys)[0].item(), len(ys))

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()

            t.set_postfix(
                DataTime=data_time.avg,
                BatchTime=batch_time.avg,
                Loss=losses.avg,
                Acc=accs.avg
            )

            # break

    return (accs.avg)


def test(loader, save_flag, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()

    gc = LayerGradCam(model, model.layer4)

    loss_fn = nn.CrossEntropyLoss()

    end = time.time()

    if save_flag:
        results = [['y', 'y_hat_vec', 'y_hat', 'viz_fn', 'fn', 't_start', 't_end']]

    with tqdm(loader, desc="Test batch iteration", disable=args.local_rank > 0) as t:
        for batch_idx, (xs, ys, (fns, t_starts, t_ends)) in enumerate(t):
            data_time.update(time.time() - end)

            xs = xs.to(device)
            ys = ys.to(device)



            y_hats = model(xs)
            loss = loss_fn(y_hats, ys)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            losses.update(loss.item(), len(ys))
            accs.update(accuracy(y_hats, ys)[0].item(), len(ys))

            batch_time.update(time.time() - end)
            end = time.time()

            d = 0


            if save_flag:
                for x, y, y_hat, fn, t_s, t_e in zip(xs, ys,
                                                 F.softmax(y_hats, dim=1),
                                                 fns, t_starts, t_ends):
                    x = unnormalize(x.cpu()).permute(1, 2, 3, 0).numpy()
                    fn_=fn
                    fn = '{0:02}_{1:010}.mp4'.format(
                        args.local_rank, batch_idx * args.batch_size + d)
                    results.append((y.item(), y_hat.tolist(), y_hat.argmax().item(), fn, fn_, t_s.item(), t_e.item()))
                    clip = ImageSequenceClip(list(x), fps=args.fps).fl_image(make_uint8)
                    clip.write_videofile(os.path.join(args.save_path, 'input', fn), logger=None)
                    clip.close()
                    d += 1

            t.set_postfix(
                DataTime=data_time.avg,
                BatchTime=batch_time.avg,
                Loss=losses.avg,
                Acc=accs.avg
            )

    if save_flag == True:
        with open(os.path.join(args.save_path, 'results_{0:06}_{1:03}.csv'.format(args.local_rank, epoch)), 'w') as f:
            wr = csv.writer(f)
            wr.writerows(results)

    return accs.avg, accs.count


def save_checkpoint(state, is_best=False, checkpoint='checkpoint', filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(
            checkpoint, 'model_best.pth'))


def load_checkpoint(checkpoint, filename='model_best.pth'):
    global start_epoch
    filepath = os.path.join(checkpoint, filename)
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    if not args.test_only:
        optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']


if __name__ == "__main__":

    # from cnns import model as model3d

    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                        help='batchsize')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('-c', '--checkpoint',
                        default="PATH/TO/checkpoint_bincls_newborders", type=str,
                        metavar='PATH',
                        help='path to save checkpoint')
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--resume_latest', action='store_true')
    parser.add_argument('--fused_optimizer', action='store_true')
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
    parser.add_argument('--opt_level', default="O2")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--save_test', action='store_true')
    parser.add_argument('--sample_all_clips', action='store_true')
    parser.add_argument('--linear_model', action='store_true')
    parser.add_argument('--cache_dataset', action='store_false')
    parser.add_argument('--frames_per_clip', type=int, default=4)
    parser.add_argument('--step_between_clips', type=int, default=1)
    parser.add_argument('--fps', type=int, default=8)
    # parser.add_argument('--subtract_mean', action='store_true')
    # parser.add_argument('--backward_predict', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--clips_per_video', type=int, default=5)
    # parser.add_argument('--in_vids', type=int, default=4)
    # parser.add_argument('--in_step', type=int, default=1)
    parser.add_argument('--save_path', default="PATH/TO/test_results_bincls_newborders")
    parser.add_argument(
        '--fails_path', default="PATH/TO/scenes")
    parser.add_argument(
        '--kinetics_path', default="PATH/TO/data")
    parser.add_argument('--border_path', default="PATH/TO/borders.json")
    parser.add_argument('--pretrain_path', default="PATH/TO/model_best.pt.tar")

    parser.add_argument('--dataset_path', default="PATH/TO/datasets")
    # '--data_root', default="PATH/TO/scenes_flow")
    parser.add_argument('--sample_size', type=int, default=112)
    parser.add_argument('--sample_duration', type=int, default=16)
    parser.add_argument('--n_kinetics_classes', type=int, default=600)
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    args.fails_only = False

    best_acc = float("inf")
    best_acc = float("-inf")
    start_epoch = 0

    os.makedirs(args.checkpoint, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.dataset_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'input'), exist_ok=True)

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')

    trainloader = get_video_loader(**vars(args), val=False, transform=train_transform)
    testloader = get_video_loader(**vars(args), val=True, transform=test_transform)

    model = r3d_18(pretrained=True)
    model.fc = nn.Linear(512, 2)
    model.to(device)

    if args.linear_model:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    # if args.local_rank <= 0: ipdb.set_trace()

    if args.fp16:
        try:
            # from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
            from apex import amp, optimizers
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        if args.fused_optimizer:
            optimizer = FusedAdam(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  # max_grad_norm=1.0,
                                  eps=1e-4)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), eps=1e-4)

        if args.loss_scale == 0:
            args.loss_scale = None

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level,
                                          loss_scale=args.loss_scale,
                                          keep_batchnorm_fp32=False if args.opt_level == "O2" else None)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), eps=1e-4)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except:
            from torch.nn.parallel import DistributedDataParallel as DDP

            print('Using PyTorch DDP - could not find Apex')
        model = DDP(model, delay_allreduce=False)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.resume_training:
        try:
            load_checkpoint(
                args.checkpoint, filename='checkpoint.pth' if args.resume_latest else 'model_best.pth')
        except FileNotFoundError:
            print('file not found!!!')
        if args.local_rank <= 0:
            print('Resumed from checkpoint, now at epoch', start_epoch)
    # else:
    #     checkpoint = torch.load(args.pretrain_path, map_location=device)
    #     model.load_state_dict(checkpoint['state_dict'])

    for epoch in trange(start_epoch, args.epochs, desc='Main epoch', disable=args.local_rank > 0):
        if args.local_rank != -1:
            trainloader.sampler.set_epoch(epoch)

        if not args.test_only:
            train_acc = train(trainloader)
            if args.local_rank <= 0:
                print()
                print('Training accuracy after epoch {0}: {1}'.format(
                    epoch, train_acc))
                print()

        with torch.no_grad():
            test_acc, test_cnt = test(testloader, args.save_test, epoch)
            test_acc = torch.Tensor([test_acc * test_cnt]).to(device)
            test_cnt = torch.Tensor([test_cnt]).to(device)
            if args.local_rank >= 0:
                distrib.reduce(test_acc, 0)
                distrib.reduce(test_cnt, 0)
                if args.local_rank == 0:
                    test_acc /= test_cnt
                test_acc = test_acc.item()
            if args.local_rank <= 0:
                print()
                print('Accuracy after epoch {0}: {1}'.format(epoch, test_acc))
                print()
            if args.test_only:
                break

        if args.local_rank <= 0 and not args.test_only:
            is_best = test_acc > best_acc
            best_acc = max(best_acc, test_acc)
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc
            }, checkpoint=args.checkpoint, is_best=is_best)
