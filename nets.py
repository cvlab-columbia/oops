from math import factorial

import torch
from torch import nn
from torchvision.models import resnext50_32x4d

from modules import conv2d_bn_relu, deconv_sigmoid, deconv_relu


class ErrorPredictor(torch.nn.Module):
    """ graph convolution network"""

    def __init__(self, args, n_channels=3, n_imgs=5):
        super(ErrorPredictor, self).__init__()

        self.net = resnext50_32x4d(num_classes=1)
        self.net.conv1 = torch.nn.Conv2d(n_channels * n_imgs, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False)

    def forward(self, x):
        return self.net(x)


class OrderPredictionNetwork(torch.nn.Module):
    def __init__(self, nin, ndim, merge_reverse=True):
        # nin - num reprs to sort
        # ndim - dim of each repr
        super(OrderPredictionNetwork, self).__init__()
        self.nin = nin
        self.actfn = nn.ReLU()
        self.pairwise = nn.Linear(ndim * 2, ndim // 2)
        npairs = nin * (nin - 1) // 2
        nout = factorial(nin)
        if merge_reverse:
            nout //= 2
        self.out = nn.Linear(npairs * ndim // 2, nout)

    def forward(self, xs):
        # xs - batch size by nin by ndim
        xs = self.actfn(xs)
        pairwise = []
        for i in range(self.nin):
            for j in range(i + 1, self.nin):
                pairwise.append(self.pairwise(torch.cat((xs[:, i], xs[:, j]), dim=1)))
        pairwise = torch.cat(pairwise, dim=1)
        pairwise = self.actfn(pairwise)
        out = self.out(pairwise)
        return out


class FlowPredictor(torch.nn.Module):
    """ graph convolution network"""

    def __init__(self, args, n_channels=2, n_imgs=4):
        super(FlowPredictor, self).__init__()

        self.conv_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(n_channels * n_imgs, 32, 4, stride=2),
            conv2d_bn_relu(32, 32, 3)
        )
        self.conv_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(32, 32, 4, stride=2),
            conv2d_bn_relu(32, 32, 3)
        )
        self.conv_stack3 = torch.nn.Sequential(
            conv2d_bn_relu(32, 64, 4, stride=2),
            conv2d_bn_relu(64, 64, 3)
        )
        self.conv_stack4 = torch.nn.Sequential(
            conv2d_bn_relu(64, 128, 4, stride=2),
            conv2d_bn_relu(128, 128, 3),
        )

        self.deconv_4 = deconv_relu(128, 64, 4, stride=2)
        self.deconv_3 = deconv_relu(66, 32, 4, stride=2)
        self.deconv_2 = deconv_relu(34, 16, 4, stride=2)
        self.deconv_1 = deconv_sigmoid(
            18, n_channels, 4, stride=2, sigmoid=False)

        self.predict_4 = torch.nn.Conv2d(
            128, n_channels, 3, stride=1, padding=1)
        self.predict_3 = torch.nn.Conv2d(
            66, n_channels, 3, stride=1, padding=1)
        self.predict_2 = torch.nn.Conv2d(
            34, n_channels, 3, stride=1, padding=1)

        self.up_sample_4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                n_channels, n_channels, 4, stride=2, padding=1, bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                n_channels, n_channels, 4, stride=2, padding=1, bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                n_channels, n_channels, 4, stride=2, padding=1, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # x: B x input_channel (3, 6) x W (128) x H (128)
        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(conv1_out)
        conv3_out = self.conv_stack3(conv2_out)
        conv4_out = self.conv_stack4(conv3_out)

        deconv4_out = self.deconv_4(conv4_out)
        predict_4_out = self.up_sample_4(self.predict_4(conv4_out))

        concat_4 = torch.cat([deconv4_out, predict_4_out], dim=1)
        deconv3_out = self.deconv_3(concat_4)
        predict_3_out = self.up_sample_3(self.predict_3(concat_4))

        concat2 = torch.cat([deconv3_out, predict_3_out], dim=1)
        deconv2_out = self.deconv_2(concat2)
        predict_2_out = self.up_sample_2(self.predict_2(concat2))

        concat1 = torch.cat([deconv2_out, predict_2_out], dim=1)
        predict_out = self.deconv_1(concat1)

        return predict_out
