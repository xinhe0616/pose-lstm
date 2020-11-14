# https://github.com/HowieMa/lstm_pm_pytorch.git
import argparse
# from model.lstm_pm import LSTM_PM
from DHP19Data import Dhp19PoseDataset
from pose_resnet import get_pose_net
from utils import JointsMSELoss,get_optimizer,save_loss
from config import config
from utils import AverageMeter, accuracy
# from src.utils import *
import os
import torch
import torch.optim as optim
import torch.nn as nn
import time

from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

device_ids = [0]

# hyper parameter
temporal = 5
train_data_dir = 'dhp_lstm/train/'
train_label_dir = 'dhp_lstm/train/'

learning_rate = 8e-6
batch_size = 1
epochs = 50
begin_epoch = 0
save_dir = './ckpt2/'
cuda = 1

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

transform = transforms.Compose([transforms.ToTensor()])

# Build dataset
train_data = Dhp19PoseDataset(data_dir=train_data_dir, label_dir=train_label_dir, temporal=temporal, train=True)
print('Train dataset total number of images sequence is ----' + str(len(train_data)))

# Data Loader
train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=False)

net = get_pose_net()
gpus = [int(i) for i in '0'.split(',')]
model = torch.nn.DataParallel(net, device_ids=gpus).cuda()

def train():
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    # optimizer = get_optimizer(config, model)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    # )
    # initialize optimizer
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # optimizer = optim.SGD(params=net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # scheduler = StepLR(optimizer, step_size=40000, gamma=0.333)

    # criterion = nn.MSELoss(size_average=True)
    for epoch in range(begin_epoch, epochs + 1):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        model.train()
        end = time.time()
        print('epoch....................................' + str(epoch))
        for i, (inputs, targets, target_weights) in enumerate(train_dataset):
            outputs = model(inputs.cuda())
            targets = targets.cuda()
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), inputs.size(0))
            for j in range(5):
                out = outputs[j]
                tar = targets[:, j]
                _, avg_acc, cnt, pred = accuracy(out.detach().cpu().numpy(),
                                                 tar.detach().cpu().numpy())
                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if i % config.PRINT_FREQ == 0:
                    msg = 'Epoch: [{0}][{1}/{2}]\t' \
                          'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                          'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch, i, len(train_dataset),
                        data_time=data_time, loss=losses, acc=acc)
                    print(msg)
                #     msg = 'Epoch: [{0}][{1}/{2}]\t' \
                #           'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                #           'Speed {speed:.1f} samples/s\t' \
                #           'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                #           'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                #           'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                #         epoch, i, len(train_dataset), batch_time=batch_time,
                #         speed=inputs.size(0) / batch_time.val,
                #         data_time=data_time, loss=losses, acc=acc)
                #     print(msg)

        #  ************************* save model per 10 epochs  *************************
        if epoch % 1 == 0:
            torch.save(net.state_dict(), os.path.join(save_dir, 'ucihand_lstm_pm{:d}.pth'.format(epoch)))

    print('train done!')

if __name__ == '__main__':
    train()








