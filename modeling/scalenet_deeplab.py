import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone
from modeling.attention import RCAB,CAM_Module

class DeepLabCA(nn.Module):
    def __init__(self, backbone='resnet_multiscale', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLabCA, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d


        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.avg_pool= nn.AdaptiveAvgPool2d(32)
        # self.se=RCAB(2048+1024+512+256+256,1,16)
        self.se=CAM_Module()
        in_channels=2048+1024+512+256+256
        inter_channels=in_channels//4
        # self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                            BatchNorm(inter_channels),
        #                            nn.ReLU())
        # self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #                            BatchNorm(inter_channels),
        #                            nn.ReLU())

        # self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, num_classes, 1))

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat,l1,l2,l3,l4= self.backbone(input)
        # import pdb
        # pdb.set_trace()

        x1=self.avg_pool(l1)
        x2=self.avg_pool(l2)
        x3=self.avg_pool(l3)
        x4=self.avg_pool(l4)
        x = self.aspp(x)
        # import pdb
        # pdb.set_trace()
        x=torch.cat((x4,x3,x2,x1,x),dim=1)
        cout1=x

        # x=self.conv5c(x)
        
        # import pdb
        # pdb.set_trace()
        # feamap=x
        x,_=self.se(x)
        # cout

        # fea=x
        # x=self.conv5(x)
        # cout2=self.conv6(x)
        # cout=x
        # cout1=F.interpolate(cout1, size=input.size()[2:], mode='bilinear', align_corners=True)
        # cout2=F.interpolate(cout2, size=input.size()[2:], mode='bilinear', align_corners=True)

        
        
        # import pdb
        # pdb.set_trace()
        # attn_map=x
        # attn=F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        # import pdb
        # pdb.set_trace()

        x = self.decoder(x, low_level_feat)
        # feature=x

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x, cout1

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


