from __future__ import print_function, absolute_import, division
from progress.bar import Bar
import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader,TensorDataset,Dataset
from torch.autograd import Variable

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Ax3DPose17(object):
  def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Create a 3d pose visualizer that can be updated with new poses.
    Args
      ax: 3d axis to plot the 3d pose on
      lcolor: String. Colour for the left part of the body
      rcolor: String. Colour for the right part of the body
    """

    # Start and endpoints of our representation
    self.I   = np.array([1,5,6,1,2,3,1,8,9,10, 9,12,13,9,15,16])-1
    self.J   = np.array([5,6,7,2,3,4,8,9,10,11,12,13,14,15,16,17])-1
    # Left / right indicator
    self.LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=bool)
    self.ax = ax

    vals = np.zeros((17, 3))

    # Make connection matrix
    self.plots = []
    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots.append(self.ax.plot(x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor))

    self.ax.set_xlabel("x")
    self.ax.set_ylabel("y")
    self.ax.set_zlabel("z")

  def update(self, channels, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Update the plotted 3d pose.
    Args
      channels: 96-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """

    assert channels.size == 51, "channels should have 51 entries, it has %d instead" % channels.size
    vals = np.reshape( channels, (17, -1) )

    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots[i][0].set_xdata(x)
      self.plots[i][0].set_ydata(y)
      self.plots[i][0].set_3d_properties(z)
      self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)

    r = 750
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    self.ax.set_xlim3d([-r+xroot, r+xroot])
    self.ax.set_zlim3d([-r+zroot, r+zroot])
    self.ax.set_ylim3d([-r+yroot, r+yroot])

    self.ax.set_aspect('auto')


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)

class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class LinearModel(nn.Module):
    def __init__(self,
                 linear_size=1024,
                 num_stage=2,
                 p_dropout=0.5):
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size =  17 * 2
        # 3d joints
        self.output_size = 17 * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)  ## 组装模型的容器，容器内的模型只是被存储在ModelList里并没有像nn.Sequential那样严格的模型与模型之间严格的上一层的输出等于下一层的输入

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        # post processing
        y = self.w2(y)

        return y

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = LinearModel().to(device)  # 初始化模型
model.apply(weight_init) # 初始化模型参数
criterion = nn.MSELoss(size_average=True).cuda() # 均方损失，size_average=True ： 返回loss.mean()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 学习率：0.001


class DealDataset(Dataset):
    def __init__(self, fname):
        path_data = os.getcwd()
        (inp, tar) = torch.load(os.path.join(path_data, fname))  # 读取数据
        self.x_data = torch.from_numpy(inp)  # from numpy to sensor
        self.y_data = torch.from_numpy(tar)
        self.len = tar.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
Dataset_train = DealDataset('train_s1.pth.tar')
Dataset_test = DealDataset('train_s5.pth.tar')
train_loader = DataLoader(
        dataset=Dataset_train,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True)
test_loader = DataLoader(
        dataset=Dataset_test,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True)

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def LR_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step/decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(train_loader, model, criterion, optimizer,
          lr_init=None, lr_now=None, glob_step=None, lr_decay=None, gamma=None,
          max_norm=True):
    losses = AverageMeter()

    model.train()

    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(train_loader))

    for i, (inps, tars) in enumerate(train_loader):
        glob_step += 1
        if glob_step % lr_decay == 0 or glob_step == 1:
            lr_now = LR_decay(optimizer, glob_step, lr_init, lr_decay, gamma)
        inputs = Variable(inps.cuda())
        targets = Variable(tars.cuda(non_blocking=True))

        outputs = model(inputs)

        # calculate loss
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm(model.parameters(),
                                    max_norm=1)  # 设置一个梯度剪切的阈值，如果在更新梯度的时候，梯度超过这个阈值，则会将其限制在这个范围之内，防止梯度爆炸
        optimizer.step()

        # update summary
        if (i + 1) % 100 == 0:
            batch_time = time.time() - start
            start = time.time()

        bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.4f}' \
            .format(batch=i + 1,
                    size=len(train_loader),
                    batchtime=batch_time * 10.0,
                    ttl=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg)
        bar.next()

    bar.finish()
    return glob_step, lr_now, losses.avg

def test(test_loader, model, criterion):
    losses = AverageMeter()

    model.eval()

    all_dist = []
    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(test_loader))

    for i, (inps, tars) in enumerate(test_loader):
        inputs = Variable(inps.cuda())
        targets = Variable(tars.cuda(non_blocking=True))

        outputs = model(inputs)
        outputs_plot = outputs
        outputs_plot  = outputs_plot.data.cpu().numpy().squeeze()
        # print(outputs.shape)

        ##################################################
        # draw skelo
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ob = Ax3DPose17(ax)

        # Plot the conditioning ground truth
        for i in tqdm(range(outputs_plot.shape[0])):
            ob.update(outputs_plot[i])
            plt.show(block=False)
            fig.canvas.draw()
            plt.pause(0.1)
        #################################################
        # calculate loss
        outputs_coord = outputs
        loss = criterion(outputs_coord, targets)

        losses.update(loss.item(), inputs.size(0))

        # calculate absolute distance from output to target
        out = outputs.data.cpu().numpy()
        tar = targets.data.cpu().numpy()
        sqerr = (out - tar) ** 2
        distance = np.zeros((sqerr.shape[0], 17))
        dist_idx = 0
        for k in np.arange(0, 17 * 3, 3):
            distance[:, dist_idx] = np.sqrt(np.sum(sqerr[:, k:k + 3], axis=1))
            dist_idx += 1
        all_dist.append(distance)

        # update summary
        if (i + 1) % 100 == 0:
            batch_time = time.time() - start
            start = time.time()

        bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.6f}' \
            .format(batch=i + 1,
                    size=len(test_loader),
                    batchtime=batch_time * 10.0,
                    ttl=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg)
        bar.next()

    all_dist = np.vstack(all_dist)
    joint_err = np.mean(all_dist, axis=0)  # the mean of every row
    ttl_err = np.mean(all_dist)  # the mean of all data
    bar.finish()
    print (">>> error: {} <<<".format(ttl_err))
    return losses.avg, ttl_err

err_best = 1000
glob_step = 0
lr_now = 1.0e-3
lr_decay = 100000
lr_init = 1.0e-3
gamma = 0.96
epoch = 100
# load Pre-training model
path = 'ckpt_best.pth.tar'
if os.path.exists(path):
  ckpt = torch.load(path)
  start_epoch = ckpt['epoch']
  err_best = ckpt['err']
  glob_step = ckpt['step']
  # lr_now = ckpt['lr']
  model.load_state_dict(ckpt['state_dict'])
  optimizer.load_state_dict(ckpt['optimizer'])

cudnn.benchmark = True  # improve computer speed
for epoch in range(0, epoch):
  print('==========================')
  print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
  # per epoch

  # glob_step, lr_now, loss_train = train(
  #           train_loader, model, criterion, optimizer,
  #           lr_init, lr_now, glob_step, lr_decay, gamma)

  loss_test, err_test = test(test_loader, model, criterion)

  # save ckpt
  is_best = err_test < err_best
  err_best = min(err_test, err_best)

  if is_best:
    print("The model is best,saving...")
    file_path = 'ckpt_best.pth.tar'
    state = {'epoch': epoch + 1,
              'lr': lr_now,
              'step': glob_step,
              'err': err_best,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict()}
    torch.save(state, file_path)
  else:
      print("The model is not good ,don't save")

print("The best error: ",err_best)