#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data import ModelNet40, SceneflowDataset
from model_reg import FlowNet3D, cal_loss
import numpy as np
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from tqdm import tqdm
from util import npmat2euler


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def SVD(src, target):
    '''
    input:
        src: [B, 3, N]
        target: [B, 3, N]
    '''
    batch_size = src.shape[0]
    reflect = nn.Parameter(torch.eye(3), requires_grad=False).cuda()
    reflect[2, 2] = -1
    src_centered = src - src.mean(dim=2, keepdim=True)
    target_centered = target - target.mean(dim=2, keepdim=True)

    H = torch.matmul(src_centered, target_centered.transpose(2, 1).contiguous()) + 1e-6
    U, S, V = [], [], []
    R = []
    for i in range(src.size(0)):
        u, s, v = torch.svd(H[i])
        r = torch.matmul(v, u.transpose(1, 0).contiguous())
        r_det = torch.det(r)
        if r_det < 0:
            u, s, v = torch.svd(H[i])
            v = torch.matmul(v, reflect)
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            # r = r * self.reflect
        R.append(r)

        U.append(u)
        S.append(s)
        V.append(v)

    U = torch.stack(U, dim=0)
    V = torch.stack(V, dim=0)
    S = torch.stack(S, dim=0)
    R = torch.stack(R, dim=0)
    t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + target.mean(dim=2, keepdim=True)
    return R, t.view(batch_size, 3)

def test_one_epoch(args, net, criterion, test_loader):
    net.eval()

    total_loss = 0
    total_mse_loss = 0
    total_epe_loss = 0
    total_attn_loss = 0
    num_examples = 0
    eulers_ab = []
    eulers_ab_pred = []
    for i, data in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        pc1, pc2, R_ab, t_ab, R_ba, t_ba, euler_ab, euler_ba = data
        pc1 = pc1.cuda()
        pc2 = pc2.cuda()
        
        batch_size = pc1.size(0)
        num_examples += batch_size
        src, src_corr, src_ds, tgt_ds, attn, attn1 = net(pc1, pc2, None, None)

        epe_loss, attn_loss = criterion(src, src_corr, R_ab.cuda(), t_ab.cuda(), src_ds, tgt_ds, attn)
        R, t = SVD(src, src_corr)
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        mse_loss = F.mse_loss(torch.matmul(R.transpose(2, 1), R_ab.cuda()), identity) \
                    + F.mse_loss(t, t_ab.cuda())
        loss = attn_loss + mse_loss+ epe_loss
        total_loss += loss.item() * batch_size
        total_mse_loss += mse_loss.item() * batch_size
        total_epe_loss += epe_loss.item() * batch_size
        total_attn_loss += attn_loss.item() * batch_size
        eulers_ab.append(euler_ab)
        eulers_ab_pred.append(npmat2euler(R.detach().cpu().numpy()))
    eulers_ab = np.concatenate(eulers_ab, axis=0)
    eulers_ab_pred = np.concatenate(eulers_ab_pred, axis=0) 
    return total_loss / num_examples, total_mse_loss / num_examples, total_epe_loss / num_examples, total_attn_loss / num_examples, eulers_ab, eulers_ab_pred


def train_one_epoch(args, net, criterion, train_loader, opt):
    net.train()
    num_examples = 0
    total_loss = 0
    total_mse_loss = 0
    total_epe_loss = 0
    total_attn_loss = 0
    eulers_ab = []
    eulers_ab_pred = []
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        pc1, pc2, R_ab, t_ab, R_ba, t_ba, euler_ab, euler_ba = data
        pc1 = pc1.cuda()
        pc2 = pc2.cuda()

        batch_size = pc1.size(0)
        opt.zero_grad()     # opt.optimizer.zero_grad()
        num_examples += batch_size
        with torch.autograd.detect_anomaly():
            src, src_corr, src_ds, tgt_ds, attn, attn1 = net(pc1, pc2, None, None)

            epe_loss, attn_loss = criterion(src, src_corr, R_ab.cuda(), t_ab.cuda(), src_ds, tgt_ds, attn)
            if args.cycle:
                cycle_epe_loss, cycle_attn_loss = criterion(src_corr, src, R_ba.cuda(), t_ba.cuda(), tgt_ds, src_ds, attn1)
            R, t = SVD(src, src_corr)
            identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
            mse_loss = F.mse_loss(torch.matmul(R.transpose(2, 1), R_ab.cuda()), identity) \
                        + F.mse_loss(t, t_ab.cuda())
            loss = attn_loss + mse_loss + epe_loss
            loss.backward()
        
        opt.step()
        total_loss += loss.item() * batch_size
        total_mse_loss += mse_loss.item() * batch_size
        total_epe_loss += epe_loss.item() * batch_size
        total_attn_loss += attn_loss.item() * batch_size
        eulers_ab.append(euler_ab)
        eulers_ab_pred.append(npmat2euler(R.detach().cpu().numpy()))
        # if (i+1) % 100 == 0:
        #     print("batch: %d, mean loss: %f" % (i, total_loss / 100 / batch_size))
        #     total_loss = 0
    eulers_ab = np.concatenate(eulers_ab, axis=0)
    eulers_ab_pred = np.concatenate(eulers_ab_pred, axis=0)
    
    return total_loss / num_examples, total_mse_loss / num_examples, total_epe_loss / num_examples, total_attn_loss / num_examples, eulers_ab, eulers_ab_pred


def test(args, net, test_loader, boardio, textio):

    test_loss, test_mse_loss, test_epe_loss, test_attn_loss, test_eulers_ab, test_eulers_ab_pred = test_one_epoch(args, net, cal_loss, test_loader)
    test_r_mse = np.mean((test_eulers_ab_pred - np.degrees(test_eulers_ab)) ** 2)
    textio.cprint('mean test loss: %f, EPE loss: %f, MSE loss: %f, attn loss: %f, rot_MSE: %f'%(test_loss, test_epe_loss, test_mse_loss, test_attn_loss, test_r_mse))


def train(args, net, train_loader, test_loader, boardio, textio):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)
    # Transformer 的 opt函数
    # opt = NoamOpt(args.emb_dims, 1, 9000,
    #         torch.optim.Adam(net.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    criterion = cal_loss

    best_test_loss = np.inf
    for epoch in range(args.epochs):
        textio.cprint('== epoch: %d, lr: %f =='%(epoch, opt.param_groups[0]['lr']))
        train_loss, train_mse_loss, train_epe_loss, train_attn_loss, train_eulers_ab, train_eulers_ab_pred = train_one_epoch(args, net, criterion, train_loader, opt)
        train_r_mse = np.mean((train_eulers_ab_pred - np.degrees(train_eulers_ab)) ** 2)
        textio.cprint('mean train loss: %f, EPE loss: %f, MSE loss: %f, attn loss: %f, rot_MSE: %f'%(train_loss, train_epe_loss, train_mse_loss, train_attn_loss, train_r_mse))

        test_loss, test_mse_loss, test_epe_loss, test_attn_loss, test_eulers_ab, test_eulers_ab_pred = test_one_epoch(args, net, criterion, test_loader)
        test_r_mse = np.mean((test_eulers_ab_pred - np.degrees(test_eulers_ab)) ** 2)
        textio.cprint('mean test loss: %f, EPE loss: %f, MSE loss: %f, attn loss: %f, rot_MSE: %f'%(test_loss, test_epe_loss, test_mse_loss, test_attn_loss, test_r_mse))
        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            textio.cprint('best test loss till now: %f'%test_loss)
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
        
        scheduler.step()
        # if torch.cuda.device_count() > 1:
        #     torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        # else:
        #     torch.save(net.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='test', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='flownet', metavar='N',
                        choices=['flownet'],
                        help='Model to use, [flownet]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Point Number [default: 1024]')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=10, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--cycle', type=bool, default=True, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--gaussian_noise', action='store_true', default=False,
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Whether to test on unseen category')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    parser.add_argument('--dataset', type=str, default='modelnet40',
                        choices=['modelnet40'], metavar='N',
                        help='dataset to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='the number of the DataLoader')
    parser.add_argument('--gpu', type=str, default='0,1',
                        help='GPUS to use')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # CUDA settings
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    boardio = []
    _init_(args)

    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    if args.dataset == 'modelnet40':
        train_loader = DataLoader(
            ModelNet40(num_points=args.num_points, partition='train', gaussian_noise=args.gaussian_noise,
                       unseen=args.unseen, factor=args.factor),
            batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
        test_loader = DataLoader(
            ModelNet40(num_points=args.num_points, partition='test', gaussian_noise=args.gaussian_noise,
                       unseen=args.unseen, factor=args.factor),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    else:
        raise Exception("not implemented")

    if args.model == 'flownet':
        net = FlowNet3D(args).cuda()
        net.apply(weights_init)
        if args.eval:
            if args.model_path is '':
                model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
            else:
                model_path = args.model_path
                print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            net.load_state_dict(torch.load(model_path), strict=False)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        raise Exception('Not implemented')
    if args.eval:
        test(args, net, test_loader, boardio, textio)
    else:
        train(args, net, train_loader, test_loader, boardio, textio)


    print('FINISH')
    # boardio.close()


if __name__ == '__main__':
    main()