#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
#from __init__ import DEVICE


class xqdaLoss(nn.Module):
    """
    Shape:
        - Input: :math:`(N, C)` where `C = number of channels`
        - Target: :math:`(N)`
        - Output: scalar.
    """

    def __init__(self, size_average=True):
        super(xqdaLoss, self).__init__()
        self.size_average = size_average

    def __call__(self, input, target, hyper_paramter,usemean,use_exp):
        y_true = target.int().unsqueeze(-1)
        same_id = torch.eq(y_true, y_true.t()).type_as(input.data)
        #print(input.shape)
        #print(len(input[0]))
        cov_i = Variable(torch.zeros([len(input[0]),len(input[0])]).cuda(),  requires_grad=True)
        cov_j = Variable(torch.zeros([len(input[0]),len(input[0])]).cuda(),  requires_grad=True)
        mean_i = Variable(torch.zeros([len(input[0])]).cuda(),  requires_grad=True)
        mean_j = Variable(torch.zeros([len(input[0])]).cuda(),  requires_grad=True)
        #print(cov_i.shape,cov_j.shape)
        #print(cov_i.shape,cov_j.shape)
        #print(cov_i.shape,cov_j.shape)
        numberofpos=0
        numberofneg=0
        pos_mask = same_id
        neg_mask = 1 - same_id
        for i in range(target.size()[0]):
            for j in range(i):
                diff=(input[i]-input[j]).view(2048,1)
                if(same_id[i][j]==1):
                    cov_i=cov_i+torch.matmul(diff,diff.t())
                    mean_i=mean_i+diff
                    numberofpos+=1
                else:
                    cov_j=cov_j+torch.matmul(diff,diff.t())
                    mean_j=mean_j+diff
                    numberofneg+=1
        # output[i, j] = || feature[i, :]||_2 - lambda*||feature[j, :] ||_2 - 
        if(use_exp==True):
            dist=torch.exp((-1)*torch.norm(cov_i-cov_j))
            loss=dist
        else: 
            dist=torch.norm(cov_i)-hyper_paramter*torch.norm(cov_j)-usemean*torch.norm((mean_i/numberofpos)-(mean_j/numberofneg))
            loss=torch.clamp(dist,min=0)
        return loss,dist,torch.norm(cov_i),torch.norm(cov_j)
