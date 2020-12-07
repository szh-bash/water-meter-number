# Configurations

import progressbar as pb

dataPath = '/data/shenzhonghai'
dataProject = '/data/shenzhonghai/water-meter-number'

# build data
label_path = dataPath+'/data/train_labels/labels'

train_origin_path = dataPath+'/data/crop_train'
trainPath = dataProject+'/data/part0'
train_size = 5266

test_origin_path = dataPath+'/data/crop_test'
testPath = dataProject+'/data/part1'
test_size = 500

modelSavePath = dataProject+'/models/demo'
modelPath = dataProject + '/models/demo.tar'

H = 112
W = 112

# train
Total = 150
batch_size = 64
learning_rate = 0.001
weight_decay = 0.00000
dp = 0.00

widgets = ['Data Loading: ', pb.Percentage(),
           ' ', pb.Bar(marker='>', left='[', right=']', fill='='),
           ' ', pb.Timer(),
           ' ', pb.ETA(),
           ' ', pb.FileTransferSpeed()]
