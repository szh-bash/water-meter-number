import os
import cv2
import numpy as np
import progressbar as pb
from config import train_origin_path, trainPath, test_origin_path, testPath, widgets, modelPath
import re
import torch
from model.vggnet.vgg16 import Vgg16
from model.resnet.resnet import resnet50
from loss import ArcMarginProduct as ArcFace
from utils.DataHandler import MinS, MaxS

H = 64
W = 42
index = [14, 54, 94, 124, 164, 204]
num = [0]*132
cnt = [0]*132
trans = [0]*40
classes = 0
for i in range(132):
    if re.match(r'[\da-ce-pr-z%@#]', chr(i)):
        num[i] = classes
        trans[classes] = i
        classes += 1
print("Default Class:", classes)


def build(filename):
    img_path = os.path.join("%s/%s" % (train_origin_path, filename))
    img = np.array(cv2.imread(img_path), dtype=float)
    for idx in range(6):
        label = filename[idx]
        if label.isupper():
            label = label.lower()
        img_single = img[:, index[idx]:index[idx] + W, :]
        dir_path = os.path.join("%s/%s" % (trainPath, label))
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        cv2.imwrite(os.path.join("%s/%d.jpg" % (dir_path, cnt[num[ord(label)]])), img_single)
        cnt[num[ord(label)]] += 1


def cut():
    res = 0
    count = 0
    if not os.path.exists(trainPath):
        os.mkdir(trainPath)
    print(trainPath)
    path_dir = os.listdir(train_origin_path)
    pgb = pb.ProgressBar(widgets=widgets, maxval=687).start()
    for allDir in path_dir:
        build(allDir)
        res += 6
        count += 1
        pgb.update(count)
    pgb.finish()
    print('origin:', count)
    print('result:', res)
    print(np.sort(cnt[0:classes]))


def mark(filename, net, arc, device):
    img = np.array(cv2.imread(test_origin_path+'/'+filename), dtype=float)
    # print(img.shape)
    # print(test_origin_path+'/'+filename)
    for idx in range(6):
        img_single = img[:, index[idx]:index[idx] + W, :]
        size = (MaxS+MinS) // 2
        st = (size-224) // 2
        img_single = cv2.resize(img_single, (size, size))
        img_single = img_single[st:st+224, st:st+224, :]
        img_single = (np.transpose(img_single, [2, 0, 1])-127.5)/128
        img_single = torch.from_numpy(np.array([img_single])).float()
        label = torch.from_numpy(np.array([0])).long()
        feat = arc(net(img_single.to(device)), label.to(device), 'test')
        label = torch.argmax(feat, dim=1).cpu().numpy()
        img_single = img[:, index[idx]:index[idx] + W, :]
        label_chr = chr(trans[label[0]])
        dir_path = os.path.join("%s/%c" % (testPath, label_chr))
        # print(label[0]+1, dir_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        cv2.imwrite(os.path.join("%s/%d.jpg" % (dir_path, cnt[label[0]])), img_single)
        cnt[label[0]] += 1


def label_image():
    res = 0
    count = 0
    device = torch.device('cuda:0')
    model = Vgg16().cuda()
    print(model)
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(modelPath)['net'].items()})
    model.eval()  # DropOut/BN
    arc = ArcFace(4096, classes).cuda()
    arc.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(modelPath)['arc'].items()})
    if not os.path.exists(testPath):
        os.mkdir(testPath)
    path_dir = os.listdir(test_origin_path)
    pgb = pb.ProgressBar(widgets=widgets, maxval=1433).start()
    for allDir in path_dir:
        mark(allDir, model, arc, device)
        res += 6
        count += 1
        pgb.update(count)
    pgb.finish()
    print('origin:', count)
    print('result:', res)
    print(np.sort(cnt[0:classes]))


if __name__ == '__main__':
    label_image()
