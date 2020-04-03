import argparse
import csv
import itertools
import os
import shutil
import time
import types

import ipdb
import numpy as np
import torch
import torch.distributed as distrib
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import LayerAttribution
from captum.attr._core.grad_cam import LayerGradCam
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from torchvision.models.video import r3d_18
from torchvision.utils import save_image, make_grid
from tqdm import trange, tqdm

from dataloader import get_video_loader, train_transform, test_transform, unnormalize
from nets import OrderPredictionNetwork
from utils import AverageMeter
from utils.kinetics_utils import accuracy


# from utils.kinetics_utils import *


# sys.path.append('/local/vondrick/dave/fails')
# sys.path.append('/local/vondrick/dave/fails/cnns')


def make_uint8(im):
    return (255 * im).astype('uint8')


def train(loader):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    sup_losses = AverageMeter()
    selfsup_acc = AverageMeter()
    selfsup_losses = AverageMeter()
    accs = AverageMeter()
    end = time.time()

    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    if args.selfsup_loss == 'pred_middle':
        selfsup_loss_fn = nn.MSELoss()
    elif args.selfsup_loss == 'sort' or args.selfsup_loss == 'fps':
        selfsup_loss_fn = loss_fn
    elif args.selfsup_loss == 'ctc':
        selfsup_loss_fn = ctc_loss
    # ys_tracker = defaultdict(int)

    with tqdm(loader, desc='Train batch iteration', disable=args.local_rank > 0) as t:
        for batch_idx, (xs, ys, (fns, t_starts, t_ends, selfsup_info, *_)) in enumerate(t):
            # for y in ys: ys_tracker[y.item()] += 1
            # continue
            data_time.update(time.time() - end)
            # if args.local_rank <= 0: ipdb.set_trace()
            xs = xs.to(device)
            ys = ys.to(device)

            # print(xs.shape, ys.shape)

            if args.selfsup_loss:
                if args.selfsup_loss == 'pred_middle' or args.selfsup_loss == 'ctc':
                    _, prev_feats = model(xs[:, 0])
                    y_hats, mid_feats = model(xs[:, 1])
                    _, next_feats = model(xs[:, 2])
                    feats = torch.cat((prev_feats, next_feats), dim=1)
                    pred_mid_feats = selfsup_model(feats)
                    valid_pred_locs = (xs[:, 0].mean(dim=(1, 2, 3, 4)) > -0.999) & (
                            xs[:, 2].mean(dim=(1, 2, 3, 4)) > -0.999)
                    if args.selfsup_loss == 'pred_middle':
                        selfsup_loss = selfsup_loss_fn(pred_mid_feats[valid_pred_locs], mid_feats[valid_pred_locs])
                    elif args.selfsup_loss == 'ctc':
                        selfsup_loss = selfsup_loss_fn(pred_mid_feats[valid_pred_locs], mid_feats[valid_pred_locs],
                                                       feats[valid_pred_locs])
                    selfsup_len = valid_pred_locs.sum().item()
                elif args.selfsup_loss == 'sort':
                    sort_ys = torch.zeros_like(ys)
                    valid_pred_locs = (xs[:, 0].mean(dim=(1, 2, 3, 4)) > -0.999) & (
                            xs[:, 2].mean(dim=(1, 2, 3, 4)) > -0.999)

                    for i in range(len(xs)):
                        p = torch.randperm(3)
                        xs[i] = xs[i][p]
                        s = ''.join(map(str, p.tolist()))
                        try:
                            sort_ys[i] = sort_y_vocab.index(s)
                        except:
                            sort_ys[i] = sort_y_vocab.index(s[::-1])

                    _, prev_feats = model(xs[:, 0])
                    y_hats, mid_feats = model(xs[:, 1])  # nonsense, can't co train with sort for now
                    _, next_feats = model(xs[:, 2])
                    feats = torch.stack((prev_feats, mid_feats, next_feats), dim=1)
                    pred_perms = selfsup_model(feats)
                    sort_ys[~valid_pred_locs] = -1
                    selfsup_loss = selfsup_loss_fn(pred_perms, sort_ys)
                    selfsup_len = valid_pred_locs.sum().item()
                    selfsup_acc.update(accuracy(pred_perms[valid_pred_locs], sort_ys[valid_pred_locs])[0].item(),
                                       selfsup_len)
                elif args.selfsup_loss == 'fps':
                    fps_ys = torch.LongTensor([args.fps_list.index(_) for _ in selfsup_info]).to(device)
                    y_hats, feats = model(xs)
                    pred_fps = selfsup_model(feats)
                    selfsup_loss = selfsup_loss_fn(pred_fps, fps_ys)
                    selfsup_len = len(ys)
                    selfsup_acc.update(accuracy(pred_fps, fps_ys)[0].item(), selfsup_len)
                suploss = loss_fn(y_hats, ys)
                loss = suploss + args.selfsup_lambda * selfsup_loss
            else:
                y_hats = model(xs)
                suploss = loss_fn(y_hats, ys)
                loss = suploss

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            losses.update(loss.item(), len(ys))
            if args.selfsup_loss:
                if (ys != -1).sum() > 0:
                    sup_losses.update(suploss.item(), (ys != -1).sum().item())
                    accs.update(accuracy(y_hats[ys != -1], ys[ys != -1])[0].item(), len(ys))
                selfsup_losses.update(selfsup_loss.item(), selfsup_len)
            else:
                accs.update(accuracy(y_hats[ys != -1], ys[ys != -1])[0].item(), len(ys))

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()

            postfix_kwargs = {}
            if args.selfsup_loss:
                postfix_kwargs = {'SelfsupLoss': selfsup_losses.avg, 'SupLoss': sup_losses.avg}
                if args.selfsup_loss == 'sort' or args.selfsup_loss == 'fps':
                    postfix_kwargs['SelfsupAcc'] = selfsup_acc.avg

            t.set_postfix(
                DataTime=data_time.avg,
                BatchTime=batch_time.avg,
                Loss=losses.avg,
                Acc=accs.avg,
                **postfix_kwargs
            )

            # break
    # if args.local_rank <= 0:
    #     print('\nObserved label distribution:',
    #           {k: round(100 * v / sum(ys_tracker.values()), 2) for k, v in ys_tracker.items()}, '\n')
    if args.selfsup_loss == 'ctc':
        return selfsup_losses.avg
    if accs.count > 0:
        return accs.avg
    else:
        return selfsup_acc.avg


# gt_scores = torch.exp(gt_dots)
# nce = gt_scores / sum_all_scores
# loss = -torch.log(nce)

def ctc_loss(pred, gt, model_in):
    all_examples = torch.cat([*model_in.chunk(2, dim=1), gt])
    n_dim = pred.shape[1]
    gt_dots = (pred * gt).sum(-1) / n_dim
    all_dots = pred @ all_examples.t() / n_dim
    all_scores = torch.exp(all_dots)
    sum_all_scores = all_scores.sum(-1)
    loss = torch.log(sum_all_scores) - gt_dots
    return loss.mean()


def test(loader, save_flag, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    sup_losses = AverageMeter()
    selfsup_acc = AverageMeter()
    selfsup_losses = AverageMeter()
    accs = AverageMeter()
    if not args.save_attn:
        model.eval()
    else:
        # ipdb.set_trace()
        model.module.fc = selfsup_model[0]
        gc = LayerGradCam(model, model.module.layer4)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    if args.selfsup_loss == 'pred_middle':
        selfsup_loss_fn = nn.MSELoss()
    elif args.selfsup_loss == 'sort' or args.selfsup_loss == 'fps':
        selfsup_loss_fn = loss_fn
    elif args.selfsup_loss == 'ctc':
        selfsup_loss_fn = ctc_loss
    end = time.time()

    if save_flag:
        results = [['y', 'y_hat_vec', 'y_hat', 'viz_fn', 'fn', 't_start', 't_end']]
        if args.flow_histogram:
            results = [['x', 'y', 'y_hat_vec', 'y_hat', 'viz_fn', 'fn', 't_start', 't_end']]

    featsarr = []
    predsarr =[]

    with tqdm(loader, desc="Test batch iteration", disable=args.local_rank > 0) as t:
        for batch_idx, (xs, ys, (fns, t_starts, t_ends, selfsup_info, *_)) in enumerate(t):
            data_time.update(time.time() - end)

            xs = xs.to(device)
            ys = ys.to(device)

            if args.get_features:
                _, feats = model(xs)
                for feat, fn, t_start, t_end in zip(feats.detach().cpu(), fns, t_starts, t_ends):
                    featsarr.append((feat, fn, t_start, t_end))
                continue

            if args.save_preds:
                _, feats = model(xs)
                pred_fps = selfsup_model(feats).argmax(1)
                for pred, fn, t_start, t_end in zip(pred_fps.detach().cpu(), fns, t_starts, t_ends):
                    predsarr.append((pred.item(), fn, t_start.item(), t_end.item()))
                continue

            if args.save_attn:
                with torch.no_grad():
                    y_hats = model(xs)
                    if args.local_rank <= 0: ipdb.set_trace()
                    yh_argmax = y_hats.argmax(dim=1)
                xs.requires_grad = True
                fps_ys = torch.LongTensor([args.fps_list.index(_) for _ in selfsup_info]).to(device)
                attr = gc.attribute(xs, yh_argmax)
                up_attr = LayerAttribution.interpolate(attr, (16, 112, 112), interpolate_mode='trilinear').to(torch.float)
                xs_ = torch.stack([unnormalize(x.cpu()) for x in xs])
                acts = xs_.cpu() * up_attr.cpu()
                acts = acts.cpu().detach().clamp(min=0)
                for act, fn, t_s, t_e, yh, y in zip(acts, fns, t_starts, t_ends, yh_argmax.tolist(), fps_ys.tolist()):
                    # if args.local_rank <= 0: ipdb.set_trace()
                    save_image(act.permute(1, 0, 2, 3), os.path.join(args.save_path, 'input', f'{os.path.splitext(os.path.basename(fn))[0]}_{int(1000*t_s)}_{int(1000*t_e)}_pred{yh}_gt{y}.png'),  normalize=True)
                accs.update(accuracy(y_hats, fps_ys)[0].item(), len(fps_ys))
                t.set_postfix(
                    Acc=accs.avg
                )
                continue

            if args.selfsup_loss:
                if args.selfsup_loss == 'pred_middle' or args.selfsup_loss == 'ctc':
                    _, prev_feats = model(xs[:, 0])
                    y_hats, mid_feats = model(xs[:, 1])
                    _, next_feats = model(xs[:, 2])
                    feats = torch.cat((prev_feats, next_feats), dim=1)
                    pred_mid_feats = selfsup_model(feats)
                    valid_pred_locs = (xs[:, 0].mean(dim=(1, 2, 3, 4)) > -0.999) & (
                            xs[:, 2].mean(dim=(1, 2, 3, 4)) > -0.999)
                    if args.selfsup_loss == 'pred_middle':
                        selfsup_loss = selfsup_loss_fn(pred_mid_feats[valid_pred_locs], mid_feats[valid_pred_locs])
                    elif args.selfsup_loss == 'ctc':
                        selfsup_loss = selfsup_loss_fn(pred_mid_feats[valid_pred_locs], mid_feats[valid_pred_locs],
                                                       feats[valid_pred_locs])
                    selfsup_len = valid_pred_locs.sum().item()
                elif args.selfsup_loss == 'sort':
                    sort_ys = torch.zeros_like(ys)
                    valid_pred_locs = (xs[:, 0].mean(dim=(1, 2, 3, 4)) > -0.999) & (
                            xs[:, 2].mean(dim=(1, 2, 3, 4)) > -0.999)

                    for i in range(len(xs)):
                        p = torch.randperm(3)
                        xs[i] = xs[i][p]
                        s = ''.join(map(str, p.tolist()))
                        try:
                            sort_ys[i] = sort_y_vocab.index(s)
                        except:
                            sort_ys[i] = sort_y_vocab.index(s[::-1])

                    _, prev_feats = model(xs[:, 0])
                    y_hats, mid_feats = model(xs[:, 1])  # nonsense, can't co train with sort for now
                    _, next_feats = model(xs[:, 2])
                    feats = torch.stack((prev_feats, mid_feats, next_feats), dim=1)
                    pred_perms = selfsup_model(feats)
                    sort_ys[~valid_pred_locs] = -1
                    selfsup_loss = selfsup_loss_fn(pred_perms, sort_ys)
                    selfsup_len = valid_pred_locs.sum().item()
                    selfsup_acc.update(accuracy(pred_perms[valid_pred_locs], sort_ys[valid_pred_locs])[0].item(),
                                       selfsup_len)
                elif args.selfsup_loss == 'fps':
                    fps_ys = torch.LongTensor([args.fps_list.index(_) for _ in selfsup_info]).to(device)
                    y_hats, feats = model(xs)
                    pred_fps = selfsup_model(feats)
                    selfsup_loss = selfsup_loss_fn(pred_fps, fps_ys)
                    selfsup_len = len(ys)
                    selfsup_acc.update(accuracy(pred_fps, fps_ys)[0].item(), selfsup_len)
                suploss = loss_fn(y_hats, ys)
                loss = suploss + args.selfsup_lambda * selfsup_loss
            else:
                y_hats = model(xs)
                suploss = loss_fn(y_hats, ys)
                loss = suploss
                # print(loss, y_hats, ys)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            losses.update(loss.item(), len(ys))
            if args.selfsup_loss:
                if (ys != -1).sum() > 0:
                    sup_losses.update(suploss.item(), (ys != -1).sum().item())
                    accs.update(accuracy(y_hats[ys != -1], ys[ys != -1])[0].item(), len(ys))
                selfsup_losses.update(selfsup_loss.item(), selfsup_len)
            else:
                accs.update(accuracy(y_hats[ys != -1], ys[ys != -1])[0].item(), len(ys))

            batch_time.update(time.time() - end)
            end = time.time()

            d = 0

            if save_flag:
                # TODO for self-supervised losses
                for x, y, y_hat, fn, t_start, t_end in zip(xs, ys,
                                                           F.softmax(y_hats, dim=1),
                                                           fns, t_starts, t_ends):
                    fn_ = fn
                    fn = '{0:02}_{1:010}.mp4'.format(
                        args.local_rank, batch_idx * args.batch_size + d)
                    other = ()
                    if args.flow_histogram:
                        other = (x.tolist(),)
                    results.append(
                        (
                            *other, y.item(), y_hat.tolist(), y_hat.argmax().item(), fn, fn_, t_start.item(),
                            t_end.item()))
                    if args.save_test_vids:
                        x = unnormalize(x.cpu()).permute(1, 2, 3, 0).numpy()
                        tt = ImageSequenceClip(list(x), fps=args.fps).fl_image(make_uint8)
                        tt.write_videofile(os.path.join(args.save_path, 'input', fn), logger=None)
                        tt.close()
                    d += 1

            postfix_kwargs = {}
            if args.selfsup_loss:
                postfix_kwargs = {'SelfsupLoss': selfsup_losses.avg, 'SupLoss': sup_losses.avg}
                if args.selfsup_loss == 'sort' or args.selfsup_loss == 'fps':
                    postfix_kwargs['SelfsupAcc'] = selfsup_acc.avg

            t.set_postfix(
                DataTime=data_time.avg,
                BatchTime=batch_time.avg,
                Loss=losses.avg,
                Acc=accs.avg,
                **postfix_kwargs
            )

    if args.get_features:
        torch.save(featsarr, os.path.join(args.save_path, 'input', 'features_and_fns.pt'))

    if args.save_preds:
        torch.save(predsarr, os.path.join(args.save_path, 'input', 'preds_and_fns.pt'))

    if save_flag == True:
        with open(os.path.join(args.save_path, 'results_{0:06}_{1:03}.csv'.format(args.local_rank, epoch)), 'w') as f:
            wr = csv.writer(f)
            wr.writerows(results)

    if args.selfsup_loss == 'ctc':
        return selfsup_losses.avg * -1, selfsup_losses.count

    if accs.count > 0:
        return accs.avg, accs.count
    else:
        return selfsup_acc.avg, selfsup_acc.count


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
    if 'selfsup_state_dict' in checkpoint:
        try:
            selfsup_model.load_state_dict(checkpoint['selfsup_state_dict'])
        except:
            print('could not load self-supervised weights, did you specify a selfsup_loss?')
    if not args.test_only:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:  # may occur when going selfsup -> sup
            pass
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


if __name__ == "__main__":

    # from cnns import model as model3d

    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                        help='batchsize')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('-c', '--checkpoint',
                        type=str,
                        metavar='PATH',
                        help='path to save checkpoint')
    parser.add_argument('-l', '--load_dir',
                        type=str,
                        metavar='PATH',
                        help='path to load checkpoint from')
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--resume_latest', action='store_true')
    parser.add_argument('--no_fails_balance', action='store_true')
    parser.add_argument('--fused_optimizer', action='store_true')
    parser.add_argument('--save_attn', action='store_true')
    parser.add_argument('--debug_dataset', action='store_true')
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
    parser.add_argument('--anticipate_label', type=float, default=0)
    parser.add_argument('--data_proportion', type=float, default=1)
    parser.add_argument('--selfsup_loss', type=str, default=None)
    parser.add_argument('--selfsup_sort_nomerge', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--save_test', action='store_true')
    parser.add_argument('--sample_all_clips', action='store_true')
    parser.add_argument('--get_clip_times', action='store_true')
    parser.add_argument('--start_from_pretrained', action='store_true')
    parser.add_argument('--linear_model', action='store_true')
    parser.add_argument('--flow_histogram', action='store_true')
    parser.add_argument('--cache_dataset', action='store_false')
    parser.add_argument('--fails_action_split', action='store_true')
    parser.add_argument('--no_labeled_fails', action='store_true')
    parser.add_argument('--all_fail_videos', action='store_true')
    parser.add_argument('--save_test_vids', action='store_true')
    parser.add_argument('--get_features', action='store_true')
    parser.add_argument('--save_preds', action='store_true')
    parser.add_argument('--load_flow', action='store_true')
    parser.add_argument('--frames_per_clip', type=int, default=16)
    parser.add_argument('--step_between_clips_sec', type=float, default=0.25)
    # parser.add_argument('--fps', type=int, default=16)
    parser.add_argument('--fps_list', type=int, nargs='+', default=[16])
    # parser.add_argument('--subtract_mean', action='store_true')
    # parser.add_argument('--backward_predict', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--clips_per_video', type=int, default=10)
    parser.add_argument('--clip_interval_factor', type=float, default=1)
    parser.add_argument('--selfsup_lambda', type=float, default=1)
    # parser.add_argument('--in_vids', type=int, default=4)
    # parser.add_argument('--in_step', type=int, default=1)
    parser.add_argument('--save_path')
    parser.add_argument(
        '--fails_path', default='/local/vondrick/datasets/fails/scenes')
    parser.add_argument(
        '--fails_flow_path', default='/local/vondrick/datasets/fails/scenes_flow_small')
    parser.add_argument(
        '--kinetics_path', default='/local/vondrick/datasets/Kinetics-600/data')
    parser.add_argument('--border_path', default="PATH/TO/borders.json")
    parser.add_argument('--pretrain_path', default='/local/vondrick/dave/fails/checkpoint_kinetics/model_best.pt.tar')

    parser.add_argument('--dataset_path', default='/proj/vondrick/dave/datasets')
    parser.add_argument('--remove_fns')
    # '--data_root', default='/proj/vondrick/datasets/fails/scenes_flow')
    parser.add_argument('--sample_size', type=int, default=112)
    parser.add_argument('--sample_duration', type=int, default=16)
    parser.add_argument('--n_kinetics_classes', type=int, default=600)
    args = parser.parse_args()

    # print(args.local_rank)

    # if args.selfsup_loss:
    #     args.all_fail_videos = True

    args.load_dir = args.load_dir or args.checkpoint
    args.cuda = torch.cuda.is_available()
    args.fails_only = True
    args.balance_fails_only = not args.no_fails_balance
    args.labeled_fails = not args.no_labeled_fails

    if args.flow_histogram:
        args.load_flow = True

    best_acc = float("inf")
    best_acc = float("-inf")
    start_epoch = 0

    if args.save_attn:
        import matplotlib.pyplot as plt

    try:
        os.makedirs(args.checkpoint, exist_ok=True)
    except:
        pass
    try:
        os.makedirs(args.save_path, exist_ok=True)
    except:
        pass
    try:
        os.makedirs(args.dataset_path, exist_ok=True)
    except:
        pass
    try:
        os.makedirs(os.path.join(args.save_path, 'input'), exist_ok=True)
    except:
        pass

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

    if args.flow_histogram:
        model = nn.Sequential(nn.Linear(100, 256), nn.Linear(256, 3))
    else:
        model = r3d_18(pretrained=args.start_from_pretrained)
        model.fc = nn.Linear(512, 3)
        if args.selfsup_loss:
            def patched_forward(self, x):
                x = self.stem(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.avgpool(x)
                # Flatten the layer to fc
                x = x.flatten(1)
                features = x
                x = self.fc(x)

                return x, features


            if not args.save_attn:
                model.forward = types.MethodType(patched_forward, model)
            if args.selfsup_loss == 'pred_middle' or args.selfsup_loss == 'ctc':
                selfsup_model = nn.Sequential(nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 512))
                selfsup_model.to(device)
            elif args.selfsup_loss == 'sort':
                # selfsup_model = nn.Sequential(nn.Linear(1536, 512), nn.ReLU(), nn.Linear(512, 6))
                # selfsup_model.to(device)
                # sort_y_vocab = ['012', '021', '102', '120', '201', '210']
                selfsup_model = OrderPredictionNetwork(nin=3, ndim=512, merge_reverse=not args.selfsup_sort_nomerge)
                selfsup_model.to(device)
                sort_y_vocab = ['012', '120', '201']
                if args.selfsup_sort_nomerge:
                    sort_y_vocab = ['012', '021', '102', '120', '201', '210']
            elif args.selfsup_loss == 'fps':
                selfsup_model = nn.Sequential(nn.Linear(512, len(args.fps_list)))
                selfsup_model.to(device)
    model.to(device)

    if args.linear_model:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    # if args.local_rank <= 0: ipdb.set_trace()

    params = model.parameters()

    if args.selfsup_loss:
        params = itertools.chain(params, selfsup_model.parameters())
    if args.fp16:
        try:
            # from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
            from apex import amp, optimizers
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        if args.fused_optimizer:
            optimizer = FusedAdam(filter(lambda p: p.requires_grad, params),
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  # max_grad_norm=1.0,
                                  eps=1e-4)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params), eps=1e-4)

        if args.loss_scale == 0:
            args.loss_scale = None

        if args.selfsup_loss:
            [model, selfsup_model], optimizer = amp.initialize([model, selfsup_model], optimizer,
                                                               opt_level=args.opt_level,
                                                               loss_scale=args.loss_scale,
                                                               keep_batchnorm_fp32=False if args.opt_level == "O2" else None)
        else:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level=args.opt_level,
                                              loss_scale=args.loss_scale,
                                              keep_batchnorm_fp32=False if args.opt_level == "O2" else None)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params), eps=1e-4)

    if args.local_rank != -1:
        try:
            # raise Exception('apex parallel doesnt mesh well with pytorch 1.2')
            from apex.parallel import DistributedDataParallel as DDP

            ddp_kwargs = {}
        except:
            from torch.nn.parallel import DistributedDataParallel as DDP

            ddp_kwargs = {'device_ids': [args.local_rank], 'output_device': args.local_rank}

            print('Using PyTorch DDP')
        model = DDP(model, **ddp_kwargs)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.resume_training:
        try:
            load_checkpoint(
                args.load_dir, filename='checkpoint.pth' if args.resume_latest else 'model_best.pth')
        except FileNotFoundError:
            print('file not found!!!')
            if args.local_rank <= 0:
                ipdb.set_trace()
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

        with torch.set_grad_enabled(args.save_attn):
            test_acc, test_cnt = test(testloader, args.save_test, epoch)
            if args.local_rank >= 0:
                test_acc = torch.Tensor([test_acc * test_cnt]).to(device)
                test_cnt = torch.Tensor([test_cnt]).to(device)
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
            other = {}
            if args.selfsup_loss:
                other = {'selfsup_state_dict': selfsup_model.state_dict()}
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
                **other
            }, checkpoint=args.checkpoint, is_best=is_best)
