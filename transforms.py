import cv2
import random
import torch
from PIL import Image, ImageFile
import numpy as np


class RandomScaleCrop():
    def __init__(self, scales, center=False):
        self.scales = list(scales)
        self.scale = random.choice(self.scales)
        self.center = center

    def __refresh__(self):
        # if torch.distributed.get_rank() <= 0:
        #    print('refresh')
        self.scale = random.choice(self.scales)

    def __call__(self, img):
        # assuming HWC not CHW
        if isinstance(img, ImageFile.ImageFile):
            img = np.array(img)
        short_side = min(img.shape[:-1])
        # if torch.distributed.get_rank() <= 0:
        #    print(self.scale)
        d = int(self.scale * short_side)
        y, x = img.shape[:-1]
        if not self.center:
            y0, x0 = random.choice([
                (0, 0),
                (y-d, 0),
                (0, x-d),
                (y-d, x-d),
                (int((y-d)/2), int((x-d)/2))
            ])
        else:
            y0, x0 = int((y-d)/2), int((x-d)/2)
        return img[y0:y0+d, x0:x0+d, :]


class Resize():
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)


# transform = Compose([RandomScaleCrop((1,), center=True),
# #                     Resize((224, 224)), ToTensor()])
# def transform_(s): return Compose([RandomScaleCrop((0.5**i for i in np.linspace(0, 1, 5)),
#                                                    center=True), Resize((s, s)), ToTensor(), Normalize(get_mean(dataset='kinetics'), get_std())])


# class VideoTransform():
#     def __init__(self, img_transform):
#         self.t = img_transform
#         try:
#             self.transforms = self.t.transforms
#         except:
#             pass

#     def __call__(self, vid):
#         try:
#             self.t.__refresh__()
#         except:
#             pass

#         return [self.t(img) for img in vid]


# def transform(s): return VideoTransform(transform_(s))
