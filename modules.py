import torch


def conv2d_bn_leakrelu(inch, outch, kernel_size, stride=1, padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch, outch, kernel_size=kernel_size,
                        stride=stride, padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.LeakyReLU()
    )
    return convlayer


def conv2d_bn_relu(inch, outch, kernel_size, stride=1, padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch, outch, kernel_size=kernel_size,
                        stride=stride, padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer


def deconv_tanh(inch, outch, kernel_size, stride=1, padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(
            inch, outch, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.Tanh()
    )
    return convlayer


def deconv_sigmoid(inch, outch, kernel_size, stride=1, padding=1, sigmoid=True):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(
            inch, outch, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.Sigmoid() if sigmoid else torch.nn.Sequential()
    )
    return convlayer


def deconv_leakrelu(inch, outch, kernel_size, stride=1, padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(
            inch, outch, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.LeakyReLU()
    )
    return convlayer


def deconv_relu(inch, outch, kernel_size, stride=1, padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(
            inch, outch, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer
