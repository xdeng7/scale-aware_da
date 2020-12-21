# imports and stuff
import os
import os.path as osp
import numpy as np
import datetime
import sys

import random
import itertools
import parser
import math

from glob import glob
# Matplotlib
import matplotlib.pyplot as plt

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable

#dataset and options 
from options.train_options import TrainOptions
from dataset.ISPRS_dataset import ISPRS_dataset
from dataset.ISPRS_dataset_multiscale import ISPRS_dataset_multi
from distutils.version import LooseVersion

import argparse

from modeling.deeplab import *
from modeling.scalenet_deeplab import *
from modeling.discriminator import FCDiscriminator


def get_parameters(model, bias=False):

    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        FCN32s,
        FCN16s,
        FCN8s,
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))



def cross_entropy2d(input, target, weight=None):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    
    loss /= mask.data.sum()
    return loss


def accuracy(pred,gt):
    return 100*float(np.count_nonzero(pred==gt))/gt.size


def update_lr(old_lr, factor, optimizer, mylog):
    new_lr=old_lr/factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    print('Learning rate changed! lr:{:.10f}>>>> {:.10f}'.format(old_lr,new_lr))
    print('Learning rate changed! lr:{:.10f}>>>> {:.10f}'.format(old_lr,new_lr),file=mylog)

    return new_lr

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter,args):
    lr = lr_poly(args.lr, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter,args):
    lr = lr_poly(args.lr_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def main():


    here = osp.dirname(osp.abspath(__file__))

    trainOpts = TrainOptions()
    args=trainOpts.get_arguments()

    now = datetime.datetime.now()
    args.out = osp.join(here,'results', args.model+'_'+args.dataset+'_'+now.strftime('%Y%m%d__%H%M%S'))

    if not osp.isdir(args.out):
        os.makedirs(args.out) 

    log_file=osp.join(args.out,args.model+'_'+args.dataset+'.log')
    mylog=open(log_file,'w')

    checkpoint_dir=osp.join(args.out,'checkpoints')
    os.makedirs(checkpoint_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset


    # MAIN_FOLDER = args.folder + 'Vaihingen/'
    # DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
    # LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'


    MAIN_FOLDER = args.folder + 'Potsdam/'
    DATA_FOLDER = MAIN_FOLDER + '3_Ortho_IRRG/top_potsdam_{}_IRRG.tif'
    LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
    ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'   

    all_files = sorted(glob(LABEL_FOLDER.replace('{}', '*')))
    all_ids = ["_".join(f.split('_')[-3:-1]) for f in all_files]


    train_ids=['2_10','3_10','3_11','3_12','4_11','4_12','5_10','5_12',\
    '6_8','6_9','6_10','6_11','6_12','7_7','7_9','7_11','7_12']
    val_ids=[ '2_11', '2_12', '4_10', '5_11', '6_7', '7_8', '7_10']

    train_set = ISPRS_dataset(train_ids, DATA_FOLDER, LABEL_FOLDER,cache=args.cache)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size)


    MAIN_FOLDER = args.folder + 'Vaihingen_multiscale/'
    DATA_FOLDER1 = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
    LABEL_FOLDER1 = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'

    DATA_FOLDER2 = MAIN_FOLDER + 'resized_resolution_half/top_mosaic_09cm_area{}.tif'
    LABEL_FOLDER2 = MAIN_FOLDER + 'gts_for_participants_half/top_mosaic_09cm_area{}.tif'


    train_ids = ['1', '3', '5', '21','23', '26', '7',  '13',  '17', '32', '37']
    val_ids =['11','15', '28', '30', '34']

    target_set = ISPRS_dataset_multi(0.5,train_ids,DATA_FOLDER1, LABEL_FOLDER1,DATA_FOLDER2, LABEL_FOLDER2,cache=args.cache)
    target_loader = torch.utils.data.DataLoader(target_set,batch_size=args.batch_size)

    val_set = ISPRS_dataset(val_ids, DATA_FOLDER1, LABEL_FOLDER1,cache=args.cache)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=args.batch_size)

    LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
    N_CLASS = len(LABELS) # Number of classes


    # 2. model

    if args.backbone=='resnet':

        model = DeepLab(num_classes=N_CLASS,
                            backbone=args.backbone,
                            output_stride=args.out_stride,
                            sync_bn=args.sync_bn,
                            freeze_bn=args.freeze_bn)

    elif args.backbone == 'resnet_multiscale':

        model = DeepLabCA(num_classes=N_CLASS,
                            backbone=args.backbone,
                            output_stride=args.out_stride,
                            sync_bn=args.sync_bn,
                            freeze_bn=args.freeze_bn)

    train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

    start_epoch = 0
    start_iteration = 0
    

    # 3. optimizer
    lr=args.lr
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
    #     momentum=args.momentum, weight_decay=args.weight_decay)
    netD_domain = FCDiscriminator(num_classes=N_CLASS)
    netD_scale = FCDiscriminator(num_classes=N_CLASS)

    optim_netG = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
    optim_netD_domain = optim.Adam(netD_domain.parameters(), lr=args.lr_D, betas=(0.9, 0.99))
    optim_netD_scale = optim.Adam(netD_scale.parameters(), lr=args.lr_D, betas=(0.9, 0.99))

    if cuda:
        model,netD_domain,netD_scale = model.cuda(),netD_domain.cuda(),netD_scale.cuda()

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)

    bce_loss = torch.nn.BCEWithLogitsLoss()
    # 4. training
    iter_=0
    no_optim=0
    val_best_loss=float('Inf')
    factor=10


    max_iter=60000

    trainloader_iter = enumerate(train_loader)
    targetloader_iter = enumerate(target_loader)

    source_label = 0
    target_label = 1

    source_scale_label=0
    target_scale_label=1


    train_loss=[]
    train_acc=[]
    target_acc_s1=[]
    target_acc_s2=[]


    while iter_ < max_iter:

        optim_netG.zero_grad()

        adjust_learning_rate(optim_netG, iter_,args)

        optim_netD_domain.zero_grad()
        optim_netD_scale.zero_grad()
        adjust_learning_rate_D(optim_netD_domain, iter_,args)
        adjust_learning_rate_D(optim_netD_scale, iter_,args)

        if iter_%1000==0:
            train_loss=[]
            train_acc=[]
            target_acc_s1=[]
            target_acc_s2=[]
        

        for param in netD_domain.parameters():
            param.requires_grad = False

        for param in netD_scale.parameters():
            param.requires_grad = False

        _, batch = trainloader_iter.__next__()
        im_s, label_s = batch


        _, batch = targetloader_iter.__next__()

        im_t_s1, label_t_s1, im_t_s2,label_t_s2 = batch

        if cuda:
            im_s, label_s =im_s.cuda(), label_s.cuda()
            im_t_s1, label_t_s1, im_t_s2,label_t_s2 =im_t_s1.cuda(), label_t_s1.cuda(), im_t_s2.cuda(),label_t_s2.cuda()
                
        ############
        #TRAIN NETG#
        ############
        #train with source 
        #optimize segmentation network with source data


        pred_seg,_= model(im_s)
        seg_loss = cross_entropy2d(pred_seg, label_s)
        seg_loss /= len(im_s)
        loss_data = seg_loss.data.item()
        if np.isnan(loss_data):
            # continue
            raise ValueError('loss is nan while training')
        seg_loss.backward()


        pred = np.argmax(pred_seg.data.cpu().numpy()[0], axis=0)
        gt = label_s.data.cpu().numpy()[0]

        train_acc.append(accuracy(pred,gt))
        train_loss.append(loss_data)

        #train with target
        pred_s1,_=model(im_t_s1)
        pred = np.argmax(pred_s1.data.cpu().numpy()[0], axis=0)
        gt = label_t_s1.data.cpu().numpy()[0]
        target_acc_s1.append(accuracy(pred,gt))

        pred_s2,_=model(im_t_s2)
        pred = np.argmax(pred_s2.data.cpu().numpy()[0], axis=0)
        gt = label_t_s2.data.cpu().numpy()[0]
        target_acc_s2.append(accuracy(pred,gt))

        pred_d=netD_domain(F.softmax(pred_s2))
        pred_s=netD_scale(F.softmax(pred_s1))

        loss_adv_domain=bce_loss(pred_d,Variable(torch.FloatTensor(pred_d.data.size()).fill_(source_label)).cuda())
        loss_adv_scale = bce_loss(pred_s,Variable(torch.FloatTensor(pred_s.data.size()).fill_(source_scale_label)).cuda())

        loss=args.lambda_adv_domain*loss_adv_domain+args.lambda_adv_scale*loss_adv_scale
        # loss=loss_adv_domain
        loss /= len(im_t_s1)
        loss.backward()

        ############
        #TRAIN NETD#
        ############
        for param in netD_domain.parameters():
            param.requires_grad = True

        for param in netD_scale.parameters():
            param.requires_grad = True

 #train with source domain and source scale
        pred_seg,pred_s1=pred_seg.detach(),pred_s1.detach()
        pred_d=netD_domain(F.softmax(pred_seg))
        # pred_s=netD_scale(F.softmax(pred_seg))
        pred_s=netD_scale(F.softmax(pred_s1))

        loss_D_domain = bce_loss(pred_d,Variable(torch.FloatTensor(pred_d.data.size()).fill_(source_label)).cuda())
        loss_D_scale = bce_loss(pred_s,Variable(torch.FloatTensor(pred_s.data.size()).fill_(source_scale_label)).cuda())

        loss_D_domain=loss_D_domain/len(im_s)/2
        loss_D_scale=loss_D_scale/len(im_s)/2

        loss_D_domain.backward()
        loss_D_scale.backward()

        #train with target domain and target scale
        pred_s1,pred_s2=pred_s1.detach(),pred_s2.detach()
        pred_d=netD_domain(F.softmax(pred_s1))
        pred_s=netD_scale(F.softmax(pred_s2))

        loss_D_domain = bce_loss(pred_d,Variable(torch.FloatTensor(pred_d.data.size()).fill_(target_label)).cuda())
        loss_D_scale = bce_loss(pred_s,Variable(torch.FloatTensor(pred_s.data.size()).fill_(target_scale_label)).cuda())

        loss_D_domain=loss_D_domain/len(im_s)/2
        loss_D_scale=loss_D_scale/len(im_s)/2

        loss_D_domain.backward()
        loss_D_scale.backward()

        optim_netG.step()
        optim_netD_domain.step()
        optim_netD_scale.step()

        # #train with source domain and source scale
        # pred_seg,pred_s2=pred_seg.detach(),pred_s2.detach()
        # pred_d=netD_domain(F.softmax(pred_seg))
        # # pred_s=netD_scale(F.softmax(pred_seg))
        # pred_s=netD_scale(F.softmax(pred_s1))

        # loss_D_domain = bce_loss(pred_d,Variable(torch.FloatTensor(pred_d.data.size()).fill_(source_label)).cuda())
        # loss_D_scale = bce_loss(pred_s,Variable(torch.FloatTensor(pred_s.data.size()).fill_(source_scale_label)).cuda())

        # loss_D_domain=loss_D_domain/len(im_s)/2
        # loss_D_scale=loss_D_scale/len(im_s)/2

        # loss_D_domain.backward()
        # loss_D_scale.backward()

        # #train with target domain and target scale
        # pred_s1,pred_s2=pred_s1.detach(),pred_s2.detach()
        # pred_d=netD_domain(F.softmax(pred_s2))
        # pred_s=netD_scale(F.softmax(pred_s1))

        # loss_D_domain = bce_loss(pred_d,Variable(torch.FloatTensor(pred_d.data.size()).fill_(target_label)).cuda())
        # loss_D_scale = bce_loss(pred_s,Variable(torch.FloatTensor(pred_s.data.size()).fill_(target_scale_label)).cuda())

        # loss_D_domain=loss_D_domain/len(im_s)/2
        # loss_D_scale=loss_D_scale/len(im_s)/2

        # loss_D_domain.backward()
        # loss_D_scale.backward()

        # optim_netG.step()
        # optim_netD_domain.step()
        # optim_netD_scale.step()

        if iter_%args.print_freq==0:
            print('Train [{}/{} Source loss:{:.6f} acc:{:.4f} % Target s1 acc:{:4f}% Target s2 acc:{:4f}%]'.format(
                iter_, max_iter, sum(train_loss)/len(train_loss),sum(train_acc)/len(train_acc),
                sum(target_acc_s1)/len(target_acc_s1),sum(target_acc_s2)/len(target_acc_s2)))
            print('Train  [{}/{} Source loss:{:.6f} acc:{:.4f} % Target s1 acc:{:4f}% Target s2 acc:{:4f}%]'.format(
                iter_, max_iter, sum(train_loss)/len(train_loss),sum(train_acc)/len(train_acc),
                sum(target_acc_s1)/len(target_acc_s1),sum(target_acc_s2)/len(target_acc_s2)),file=mylog)
        

        if iter_%1000==0:
            print('saving checkpoint.....')
            torch.save(model.state_dict(), osp.join(checkpoint_dir,'iter{}.pth'.format(iter_)))

        iter_ +=1

if __name__ == '__main__':
    main()