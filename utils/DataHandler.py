import cv2
import numpy as np
# import torch
import torchvision.transforms as trans
from config import H, W


_OH = 64 - 1
_OW = 48 - 1
_CH = 12
_CW = 12
MinS = 112
MaxS = 128


class Augment:
    rng = np.random

    def cutout(self, img):
        if self.rng.rand() < 0.5:
            y = int(self.rng.rand() * _OH)
            x = int(self.rng.rand() * _OW)
            img[y:y+_CH, x:x+_CW, :] = 255.
        return img

    def resize(self, img):
        new_size = self.rng.randint(MinS, MaxS+1)
        # new_size = 256
        new_size = (new_size, new_size)
        img = cv2.resize(img, new_size)
        return img

    def crop(self, img):
        dh = img.shape[1] - H
        dw = img.shape[0] - W
        y = int(self.rng.rand()*dh)
        x = int(self.rng.rand()*dw)
        return img[y:y+H, x:x+W, :]

    def rotate(self, img):
        if self.rng.rand() < 0.3:
            img = trans.RandomRotation(img, 15)  # -15 -> +15
        return img

    def flip(self, img):
        if self.rng.rand() < 0.5:
            img = img[:, ::-1, :]
        return img

    def trans(self, img):
        pass

    def run(self, img, label):
        # img = cv2.resize(img, (H, W))
        # img = self.cutout(img)
        img = self.resize(img)
        img = self.crop(img)
        img = np.transpose(img, [2, 0, 1])
        img = (img - 127.5) / 128.0
        return img, label
