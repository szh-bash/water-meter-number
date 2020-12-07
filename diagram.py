import re
import numpy as np
import matplotlib.pyplot as plt


def smooth(seq):
    m = 10
    n = len(seq)
    res = []
    for i in range(n-m):
        if i == 120000:
            print(seq[i])
        sum_ = 0
        for j in range(m):
            sum_ += seq[i+j]
        res.append(sum_/m)
    return res


# log_path = '/data/shenzhonghai/dmo-captcha/logs/vgg16_32_48.log'
log_path = '/data/shenzhonghai/dmo-captcha/logs/resnet_36_56_m30_co_clean6.log'
acc = []
loss = []
with open(log_path, 'r') as f:
    for st in f.readlines():
        if re.search('iters', st) is None:
            continue
        loc = re.search(r'loss: [\d]*\.[\d]*', st).span()
        loss.append(float(st[loc[0]+6:loc[1]]))
        loc = re.search(r'acc: [\d]*\.[\d]', st).span()
        acc.append(float(st[loc[0]+5:loc[1]]))
loss = smooth(loss)
acc = smooth(acc)
print(log_path)

iterations = len(acc)
x = np.linspace(0, iterations, iterations)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, loss, label='loss', color='r')
ax2.plot(x, acc, label='train_acc', color='b')
ax1.set_xlim(0, iterations)
ax1.set_ylim(0., min(1, np.max(loss)))
ax2.set_ylim(80., 100.)
ax1.set_ylabel('loss')
ax2.set_ylabel('train_acc')
plt.xlabel('iterations')
plt.title(log_path.split('/')[-1])
fig.legend(bbox_to_anchor=(1., 0.6), bbox_transform=ax1.transAxes)

plt.show()
