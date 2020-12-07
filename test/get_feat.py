import sys
import torch
import numpy as np
import progressbar as pb

sys.path.append("/home/shenzhonghai/dmo-captcha")
from init import DataReader
from model.resnet.resnet import resnet50
from torch.utils.data import DataLoader
from loss import ArcMarginProduct as ArcFace

from config import modelPath, widgets
from utils.cut import trans


feats = []
test_Y = []


def save_feat(ft, lb, lim):
    ft = ft.cpu()
    lb = lb.cpu()
    for dx in range(lim):
        feats.append(ft[dx].data.numpy())
        test_Y.append(lb[dx].data.numpy())


# load data
batch_size = 1
data = DataReader('test')
data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, pin_memory=True)

# load model
device = torch.device('cuda:0')
model = resnet50().to(device)
model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(modelPath)['net'].items()})
model.eval()  # DropOut/BN
arc = ArcFace(2048 * 7 * 7, data.type).to(device)
arc.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(modelPath)['arc'].items()})
print('Calculating Feature Map...')-
ids = 0
Total = (data.sample - 1) / batch_size + 1
pgb = pb.ProgressBar(widgets=widgets, maxval=Total).start()
for i, (inputs, labels) in enumerate(data_loader):
    feat = arc(model(inputs.to(device)), labels.to(device), 'test')
    save_feat(feat, labels, labels.size(0))
    pgb.update(i)
pgb.finish()
test_Y = np.array(test_Y)
feats = np.array(feats)
pred = np.argmax(feats, axis=1)
index = np.arange(0, data.sample)
index = index[pred != test_Y]
print('Model:', modelPath)
print('Test Accuracy:', np.sum(pred == test_Y)/Total)
print([chr(trans[pred[x]]).upper() for x in index])
print([chr(trans[test_Y[x]]).upper() for x in index])
