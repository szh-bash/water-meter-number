import os
import cv2
import numpy as np
import progressbar as pb
from config import train_origin_path, trainPath, test_origin_path, testPath, widgets, modelPath, label_path
import re
import torch
from model.vggnet.vgg16 import Vgg16
from model.resnet.resnet import resnet50
from loss import ArcMarginProduct as ArcFace
from utils.DataHandler import MinS, MaxS

# H = 64
# W = 42
# index = [14, 54, 94, 124, 164, 204]
classes = 10
cnt = [0]*classes
print("Default Class:", classes)


def get_label(filename):
    filename = filename.split('.')
    filename = label_path+'/'+filename[0]+'.txt'
    f = open(filename, 'r')
    st = f.read()
    f.close()
    st = st.split(' ')[8:]
    # print(st)
    return st


def build(filename, target, origin):
    img_path = os.path.join("%s/%s" % (origin, filename))
    img = np.array(cv2.imread(img_path), dtype=float)
    lb = get_label(filename)
    w = img.shape[1]
    dw = w / 5
    res = 0
    for idx in range(5):
        flag = [0]*10
        for label in lb:
            if flag[int(label[idx])] == 0:
                img_single = img[:, int(idx*dw):int((idx+1)*dw), :]
                img_single = cv2.resize(img_single, (112, 112))
                dir_path = os.path.join("%s/%s" % (target, label[idx]))
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                cv2.imwrite(os.path.join("%s/%d.jpg" % (dir_path, cnt[int(label[idx])])), img_single)
                cnt[int(label[idx])] += 1
                flag[int(label[idx])] = 1
                res += 1
    return res


def cut(target, origin):
    res = 0
    count = 0
    if not os.path.exists(target):
        os.mkdir(target)
    print(target)
    path_dir = os.listdir(origin)
    pgb = pb.ProgressBar(widgets=widgets, maxval=1000).start()
    for allDir in path_dir:
        res += build(allDir, target, origin)
        count += 1
        pgb.update(count)
    pgb.finish()
    print('origin:', count)
    print('result:', res)
    print(np.sort(cnt[0:classes]))


if __name__ == '__main__':
    cut(trainPath, train_origin_path)
    # cut(testPath, test_origin_path)
    # label_image()
