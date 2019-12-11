import torch.nn as nn
import torch
import copy
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from util import PointNetSetAbstraction, PointNetFeaturePropogation, FlowEmbedding, PointNetSetUpConv
from lib import pointnet2_utils as pointutils
from util import query_ball_point
from scipy.spatial.transform import Rotation

def nearest_neighbor(src, dst):
    '''
    find the nearest neighbor of src in dst.
    input: 
        src: [B, C, N]
        dst: [B, C, N]
    output:
        distances: [B, N]
        indices: [B, N]
    '''
    inner = -2 * torch.matmul(src.transpose(2,1).contiguous(), dst)  # src, dst (num_dims, num_points)
    distances = -torch.sum(src ** 2, dim=1, keepdim=True).transpose(2,1).contiguous() - inner - torch.sum(dst ** 2, dim=1, keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances.squeeze_(), indices.squeeze_()

def cal_loss(src, src_corr, rotation_ab, t_ab, src_ds, tgt_ds, attn, attention_loss = True):
    B = src.shape[0]
    gt_src_corr = torch.matmul(rotation_ab, src) + t_ab.unsqueeze(2)
    epe_loss = torch.mean(torch.sum((src_corr - gt_src_corr) ** 2, dim=1) / 2.0)
    if attention_loss:
        # 计算src在ground truth的R，t下旋转的点云与tgt的对应关系
        src_ds_corr = torch.matmul(rotation_ab, src_ds) + t_ab.unsqueeze(2)
        dist, tgt_idx = nearest_neighbor(src_ds_corr, tgt_ds)  # [B, N]
        mask = -dist < 1e-3
        mask_sum = mask.sum(-1)
        
        attn_max = attn.max(-1)  # [B, N, N]
        # attn_loss = F.nll_loss(torch.log(attn + 1e-8), tgt_idx)
        attn_loss = torch.gather(torch.log(attn + 1e-8), dim=-1, index=tgt_idx.unsqueeze(-1)) * mask.unsqueeze(-1).float()
        attn_loss = -attn_loss.mean()
        loss = epe_loss + attn_loss
    return epe_loss, attn_loss

class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.emb_dims = args.emb_dims
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.ff_dims = args.ff_dims
        self.n_heads = args.n_heads
        self.sqrt_dk = math.sqrt(512)
        self.mlp = [512, 512, 256]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        Cin = 256
        for Cout in self.mlp:
            self.convs.append(nn.Conv1d(Cin, Cout, 1))
            self.bns.append(nn.BatchNorm1d(Cout))
            Cin = Cout
        
        
    def forward(self, src, tgt, attention='dist'):
        '''
        src, tgt: [B, C, N]
        '''
        if attention is 'dot':
            attn = torch.matmul(src.transpose(2, 1).contiguous(), tgt) / \
                    (torch.norm(tgt, 2, 1, keepdim=True) + 1e-6) / \
                    (torch.norm(src.transpose(2, 1).contiguous(), 2, -1, keepdim=True) + 1e-6)
            # attn = attn.clamp(-1.,1.)
            # x1 = torch.acos(attn)
            attn = attn.transpose(2,1).contiguous()
            for i, conv in enumerate(self.convs):
                if i == len(self.mlp) - 1:
                    attn = conv(attn)
                else:
                    attn = F.relu(self.bns[i](conv(attn)))
            attn = attn.transpose(2, 1).contiguous()
        elif attention is 'dist':
            inner = -2 * torch.matmul(src.transpose(2, 1).contiguous(), tgt)    # [B, N, N]
            src_square = torch.sum(src ** 2, dim=1, keepdim=True)   # [B, 1, N]
            tgt_square = torch.sum(tgt ** 2, dim=1, keepdim=True)   # [B, 1, N]
            attn = -tgt_square - inner - src_square.transpose(2, 1).contiguous()   # [B, N, N]
        x = F.softmax(attn, dim = -1)
        return [], x


def get_corr_points(feature2_attn, k=64):
    '''
    feature2_attn: the target attention of source, [B, N, C]
    '''
    device = feature2_attn.device
    B = feature2_attn.shape[0]
    # target 点云中在src点云对应点的下标
    # tgt_corr, tgt_corr_idx = feature2_attn.max(-1)
    # src 点云中在 tgt 点云对应点的下标
    src_corr, src_corr_idx = feature2_attn.max(-1)   # [B, N]
    
    topk_src_corr, topk_idx = src_corr.topk(k, dim=-1) # [B, k]
    
    batch_indices = torch.arange(B, dtype=torch.int64).to(device).view(B,1).repeat(1,k)
    src_corr_idx = src_corr_idx[batch_indices, topk_idx]

    return topk_src_corr, src_corr_idx.int(), topk_idx.int()

class PosRefine(nn.Module):
    def __init__(self, radius, nsample, in_channel, mlp, mlp2, pooling='max', corr_func='concat', knn = True):
        super(PosRefine, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.pooling = pooling
        self.corr_func = corr_func
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        if corr_func is 'concat':
            last_channel = in_channel*2+3+1
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        for out_channel in mlp2:
            self.mlp2_convs.append(nn.Sequential(nn.Conv1d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm1d(out_channel)))
            last_channel = out_channel

    def forward(self, pos1, pos2, feature1, feature2):
        """
        Input:
            xyz1: (batch_size, 3, npoint) less points
            xyz2: (batch_size, 3, npoint)
            feat1: (batch_size, channel, npoint)
            feat2: (batch_size, channel, npoint)
        Output:
            xyz1: (batch_size, 3, npoint)
            feat1_new: (batch_size, mlp[-1], npoint)
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B, N, C = pos1_t.shape
        if self.knn:
            _, idx = pointutils.knn(self.nsample, pos1_t, pos2_t)
        else:
            # If the ball neighborhood points are less than nsample,
            # than use the knn neighborhood points
            idx, cnt = query_ball_point(self.radius, self.nsample, pos2_t, pos1_t)
            # 利用knn取最近的那些点
            _, idx_knn = pointutils.knn(self.nsample, pos1_t, pos2_t)
            cnt = cnt.view(B, -1, 1).repeat(1, 1, self.nsample)
            idx = idx_knn[cnt > (self.nsample-1)]
        
        pos2_grouped = pointutils.grouping_operation(pos2, idx) # [B, 3, N, S]
        pos_diff = pos2_grouped - pos1.view(B, -1, N, 1)    # [B, 3, N, S]
        
        feat2_grouped = pointutils.grouping_operation(feature2, idx)    # [B, C, N, S]
        if self.corr_func=='concat':
            feat_diff = torch.cat([feat2_grouped, feature1.view(B, -1, N, 1).repeat(1, 1, 1, self.nsample)], dim = 1)
        
        feat1_new = torch.cat([pos_diff, feat_diff], dim = 1)  # [B, 2*C+3,N,S]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            feat1_new = F.relu(bn(conv(feat1_new)))

        feat1_new = torch.max(feat1_new, -1)[0]  # [B, mlp[-1], npoint]

        for i, conv in enumerate(self.mlp2_convs):
            feat1_new = F.relu(conv(feat1_new))

        return feat1_new

class FlowNet3D(nn.Module):
    def __init__(self,args):
        super(FlowNet3D,self).__init__()

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=0, mlp=[64,64,128], mlp2=[128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256], mlp2=[], group_all=False)

        self.pointer = Attention(args = args)

        self.fe_layer = PosRefine(radius=0.2, nsample=16, in_channel = 256, mlp=[512, 256], mlp2=[256, 128, 3], pooling='max', corr_func='concat')
        # self.su1 = PointNetSetUpConv(nsample=8, radius=0.4, f1_channel = 256, f2_channel = 512, mlp=[], mlp2=[256, 256])
        # self.su2 = PointNetSetUpConv(nsample=8, radius=0.2, f1_channel = 128+128, f2_channel = 256, mlp=[128, 128, 256], mlp2=[256])
        # self.su3 = PointNetSetUpConv(nsample=8, radius=0.1, f1_channel = 64, f2_channel = 256, mlp=[128, 128, 256], mlp2=[256])
        # self.fp = PointNetFeaturePropogation(in_channel = 256, mlp = [256, 256])
        
        self.conv = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(256),
                                  nn.ReLU(),
                                  nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(256),
                                  nn.ReLU(),
                                  nn.Conv1d(256, 512, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(512),
                                  nn.ReLU())
        
    def forward(self, pc1, pc2, feature1, feature2):
        B, C, N = pc1.shape
        device = pc1.device
        l1_pc1, l1_feature1 = self.sa1(pc1, feature1)
        l2_pc1, l2_feature1 = self.sa2(l1_pc1, l1_feature1)
        
        l1_pc2, l1_feature2 = self.sa1(pc2, feature2)
        l2_pc2, l2_feature2 = self.sa2(l1_pc2, l1_feature2)
        
        l2_feature1_new = self.conv(l2_feature1)
        l2_feature2_new = self.conv(l2_feature2)
        # l2_feature1 = self.conv(l2_feature1)
        # l2_feature2 = self.conv(l2_feature2)

        feature1_attn, feature2_attn = self.pointer(l2_feature1_new, l2_feature2_new)  # [B, 256, C]
        
        src_corr_score, src_corr_idx, src_idx = get_corr_points(feature2_attn)
        
        # attention_feature2 = torch.matmul(l2_feature2, feature2_attn.transpose(2, 1).contiguous())  # [B, C, N]
        # attention_tgt = torch.matmul(l2_pc2, feature2_attn.transpose(2, 1).contiguous())  # [B, C, N]

        src_corr_feature = pointutils.gather_operation(l2_feature1, src_idx)
        src_corr_pc = pointutils.gather_operation(l2_pc2, src_corr_idx)
        src_interest = pointutils.gather_operation(l2_pc1, src_idx)
        # TODO: 将256维score加入feature中试一下，目前只加了max的score
        pos_diff = self.fe_layer(src_corr_pc, l2_pc2, torch.cat([src_corr_feature,src_corr_score.view(B, 1, -1)],dim = 1),l2_feature2)

        # l3_fnew1 = self.su1(l3_pc1, l4_pc1, l3_feature1, l4_feature1)
        # l2_fnew1 = self.su2(l2_pc1, l3_pc1, torch.cat([l2_feature1, l2_feature1_new], dim=1), l3_fnew1)
        # l1_fnew1 = self.su3(l1_pc1, l2_pc1, l1_feature1, l2_fnew1)
        # l0_fnew1 = self.fp(pc1, l1_pc1, feature1, l1_fnew1)
        
        src_corr_pc = src_corr_pc + pos_diff

        return src_interest, src_corr_pc, l2_pc1, l2_pc2, feature2_attn, feature1_attn
        
if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8,3,2048))
    label = torch.randn(8,16)
    model = FlowNet3D()
    output = model(input,input)
    print(output.size())
