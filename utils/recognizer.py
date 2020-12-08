import re
import os
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import pandas as pd

from model.resnet.resnet import resnet50
from config import dataProject, modelPath, test_origin_path, modelName


H = 112
W = 112
MinS = 112
MaxS = 128
size = (MinS+MaxS) // 2
st = (size - MinS) // 2
classes = 10


class ArcMarginProduct(nn.Module):

    def __init__(self, in_features, out_features):
        super(ArcMarginProduct, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs):
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        return cosine


def predict(filepath):
    res = ''
    img = np.array(cv2.imread(filepath), dtype=float)
    dw = img.shape[1] / 5

    for idx in range(5):
        image = img[:, int(dw*idx):int(dw*(idx+1)), :]
        image = cv2.resize(image, (size, size))
        image = image[st:st+MinS, st:st+MinS, :]
        image = (np.transpose(image, [2, 0, 1])-127.5)/128
        image = torch.from_numpy(np.array([image])).float().cuda()
        feat = arc(net(image))
        lb = torch.argmax(feat, dim=1).cpu().numpy()[0]
        res += str(lb)
    res = [filepath.split('/')[-1], res]
    return res


if __name__ == '__main__':
    timer = time.time()
    net = resnet50().cuda()
    net.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(modelPath, map_location='cpu')['net'].items()})
    net.eval()
    arc = ArcMarginProduct(2048 * 7 * 7, classes).cuda()
    arc.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(modelPath, map_location='cpu')['arc'].items()})
    arc.eval()
    print('Load time:', time.time() - timer)

    path = test_origin_path
    pred = [['filename', 'result']]
    for idx in range(500):
        prd = predict(path+'/test_'+str(idx+1)+'.jpg')
        print(prd)
        pred.append(prd)
    pred = np.array(pred)
    dt = pd.DataFrame(pred)
    dt.to_csv(dataProject+'/result/'+modelName+'_ep156.csv', header=False, index=False)
