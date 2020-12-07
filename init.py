import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import progressbar as pb
from utils.DataHandler import Augment, MaxS, MinS
from config import trainPath, widgets, testPath, train_size, test_size, H, W
from utils.cut import num
# lfw: 5749, 13233
# webface: 10575, 494414
# clean-webface: 10575, 455594
# dmo-captcha: 65, 4122


aug = Augment()


class DataReader(Dataset):
    type = 0
    sample = 0
    dataset = []
    label = []
    rng = np.random
    st = ''

    def __init__(self, st):
        self.st = st
        if self.st == 'train':
            file_path = trainPath
            total = train_size
        else:
            file_path = testPath
            total = test_size
        path_dir = os.listdir(file_path)
        print('Data Path:', file_path)
        pgb = pb.ProgressBar(widgets=widgets, maxval=total).start()
        for allDir in path_dir:
            child = os.path.join('%s/%s' % (file_path, allDir))
            child_dir = os.listdir(child)
            for allSon in child_dir:
                son = os.path.join('%s/%s' % (child, allSon))
                if self.st == 'train':
                    self.dataset.append(cv2.imread(son))
                elif self.st == 'test':
                    self.dataset.append(cv2.imread(son))
                self.label.append(num[ord(allDir)])
                pgb.update(self.sample)
                self.sample += 1
            self.type += 1
        pgb.finish()
        print('Processing Method:', self.st)
        if self.st == 'train':
            self.dataset = np.array(self.dataset, dtype=float)
        elif self.st == 'test':
            self.dataset = np.array(self.dataset, dtype=float)
        self.label = np.array(self.label)
        print('Types:', self.type)
        print('Label:', self.label.shape)
        print('Label_value:', self.label[345:350])
        self.y = torch.from_numpy(self.label).long()

    def __getitem__(self, index):
        if self.st == 'train':
            image = self.dataset[index]
            image, label = aug.run(image, self.y[index])
            image = torch.from_numpy(image).float()
            return image, label
        elif self.st == 'test':
            size = (MinS+MaxS) // 2
            idx = (size-W) // 2
            img = self.dataset[index]
            image = cv2.resize(img, (size, size))
            image = image[idx:idx+H, idx:idx+W, :]
            image = np.transpose(image, [2, 0, 1])
            image = torch.from_numpy((image-127.5)/128).float()
            return image, self.y[index]
        else:
            exit(-1)

    def __len__(self):
        return self.sample


if __name__ == '__main__':
    DataReader('train')
