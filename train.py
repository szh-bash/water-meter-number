import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.vggnet.vgg16 import Vgg16
from model.resnet.resnet import resnet50
from loss import ArcMarginProduct as ArcFace

from config import learning_rate, batch_size, weight_decay, Total, modelSavePath
from init import DataReader


def get_label(output):
    # print(output.shape)
    return torch.argmax(output, dim=1)


def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


def get_max_gradient(g):
    # print('Gradient:', g)
    pm = torch.max(g)
    nm = torch.min(g)
    if -nm > pm:
        return nm
    else:
        return pm


if __name__ == '__main__':
    # set config
    data = DataReader('train')
    slides = (data.sample - 1) // batch_size + 1
    grads = {}

    # Some Args setting
    # net = Vgg16()
    net = resnet50()
    device = torch.device("cuda:0")
    if torch.cuda.device_count() > 1:
        devices_ids = [i for i in range(torch.cuda.device_count())]
        net = nn.DataParallel(net, device_ids=devices_ids)
        print("Let's use %d/%d GPUs!" % (len(devices_ids), torch.cuda.device_count()))
    net.to(device)
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    arcFace = ArcFace(2048 * 7 * 7, data.type).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam([{'params': net.parameters()},
                            {'params': arcFace.parameters()}],
                           lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3600, 5600], gamma=0.1, last_epoch=-1)
    print(net.parameters())
    print(arcFace.parameters())
    if os.path.exists(modelSavePath+'.tar'):
        checkpoint = torch.load(modelSavePath+'.tar')
        net.load_state_dict(checkpoint['net'])
        arcFace.load_state_dict(checkpoint['arc'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch_start = checkpoint['epoch']
        iter_start = checkpoint['iter']
        print('Load checkpoint Successfully!')
        print('epoch: %d\niter: %d' % (epoch_start, iter_start))
        # scheduler.state_dict()['milestones'][1500] = 1
        # scheduler.state_dict()['milestones'].pop(2800)
        # scheduler.state_dict()['milestones'][3000] = 1
        # scheduler.state_dict()['milestones'].pop(4000)
        # print(scheduler.state_dict())
    else:
        epoch_start = 0
        iter_start = 0
        # torch.save({'net': net.state_dict(),
        #             'epoch': 0,
        #             'iter': 0
        #             }, modelSavePath+str(0)+'.tar')
        print('Model saved to %s' % (modelSavePath + '.tar'))

    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    for param in arcFace.parameters():
        num_params += param.numel()
    print("Parameter Number: %d M" % (num_params / 1e6))

    print("Training Started!")
    iterations = iter_start
    for epoch in range(epoch_start, Total):
        data_time, train_time = 0, 0
        pred, train_x, train_y, loss = None, None, None, None

        batch_data_time = time.time()
        acc_bc, loss_bc = 0, 0
        for i, (inputs, labels) in enumerate(data_loader):
            train_x, train_y = inputs.to(device), labels.to(device)
            dt = time.time() - batch_data_time
            data_time = data_time + dt

            batch_train_time = time.time()
            # learning_rate = adjust_lr(optimizer, epoch, learning_rate)
            # print(optimizer.state_dict()['param_groups'])
            feat = net(train_x)
            feat = arcFace(feat, train_y)
            feat.register_hook(save_grad('feat_grad'))
            loss = criterion(feat, train_y)
            optimizer.zero_grad()   # zero the gradient buffers
            loss.backward()
            optimizer.step()    # Does the update
            scheduler.step()
            tt = time.time() - batch_train_time
            train_time = train_time + tt

            iterations += 1
            # if iterations % 200 == 0:
            #     print('Output Max pred: %s' % torch.max(feat.gather(1, train_y.view(-1, 1))))
            #     print('Abs Max Gradient of net_output:',
            #           get_max_gradient(grads['feat_grad'].gather(1, train_y.view(-1, 1))))

            if iterations % 1000 == 0:
                torch.save({'net': net.state_dict(), 'arc': arcFace.state_dict()},
                           modelSavePath+'_'+str(iterations)+'.pt')
                print('Model saved to '+modelSavePath+'_'+str(iterations)+'.pt')
            # if iterations % 1 == 0:
            pred = get_label(feat)
            acc = (pred == train_y).sum().float() / train_y.size(0) * 100
            print('epoch: %d/%d, iters: %d, lr: %.5f, '
                  'loss: %.5f, acc: %.5f, train_time: %.5f, data_time: %.5f' %
                  (epoch, Total, iterations, scheduler.get_lr()[0],
                   float(loss), float(acc), tt, dt))
            loss_bc += float(loss)
            acc_bc += float(acc)
            batch_data_time = time.time()

        epoch += 1
        loss_bc /= slides
        acc_bc /= slides
        print('epoch: %d/%d, loss: %.5f, acc: %.5f, train_time: %.5f, data_time: %.5f' %
              (epoch, Total, loss_bc, acc_bc, train_time, data_time))
        state = {'net': net.state_dict(),
                 'arc': arcFace.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(),
                 'epoch': epoch,
                 'iter': iterations,
                 'loss': loss_bc,
                 'acc': acc_bc}
        torch.save(state, modelSavePath+'.tar')
        print('Model saved to %s' % (modelSavePath+'.tar'))
        # if acc_bc > 99.99:
        #     break

    print('fydnb!')
