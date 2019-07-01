import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b3')

class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=2, dilation=2)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3) 
        )       
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
#        self.branch5 = nn.Sequential(
#            nn.Conv2d(in_channel, out_channel, 1),
#            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
#            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
#            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7))
        self.conv_cat = nn.Conv2d(5*out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x0 = self.branch0(x)
#        print(x0.shape)
        x1 = self.branch1(x)
#        print(x1.shape)
        x2 = self.branch2(x)
#        print(x2.shape)
        x3 = self.branch3(x)
#        print(x3.shape)
        x4 = self.branch4(x)
#        print(x4.shape)

        x_cat = torch.cat((x0, x1, x2, x3, x4), 1)
        x_cat = self.conv_cat(x_cat)

        x = self.relu(x_cat + self.conv_res(x))
        return x

class aggregation(nn.Module):
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = nn.Conv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = nn.Conv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = nn.Conv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = nn.Conv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x1, x2, x3, x4):
        # x1: 1/16 x2: 1/8 x3: 1/4
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

       
        
class SAM(nn.Module):
    def _init_(self, in_channel, out_channel):
        super(SAM,self)._init_()
        self.branch = nn.Sequential(nn.Conv2d(in_channel, out_channel,1),
            nn.BatchNorm2d(out_channel),nn.Sigmoid())
    def forward(self, x):
        x0 = self.branch(x)
        x = torch.mul(x0,x)
        return x
    
class FusionLayer(nn.Module):
    def __init__(self, nums=4):
        super(FusionLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(nums))
        self.nums = nums
        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.weights, 1 / self.nums)

    def forward(self, x):
        for i in range(self.nums):
            out = self.weights[i] * x[i] if i == 0 else out + self.weights[i] * x[i]
        return out     
class QNet(nn.Module):
    def __init__(self, channel=32):
        super(QNet, self).__init__()
#        resnext = ResNeXt101()
#        self.layer0 = resnext.layer0
#        self.layer1 = resnext.layer1
#        self.layer2 = resnext.layer2
#        self.layer3 = resnext.layer3
#        self.layer4 = resnext.layer4
        net = model
        self.layer0 = net._conv_stem
        self.layer1 = net._bn0
        self.layer2 = nn.Sequential(*list(net._blocks[:2]))
        self.layer3 = nn.Sequential(*list(net._blocks[2:5]))
        self.layer4 = nn.Sequential(*list(net._blocks[5:8]))
        self.layer5 = nn.Sequential(*list(net._blocks[8:13]))
        self.layer6 = nn.Sequential(*list(net._blocks[13:18]))
        self.layer7 = nn.Sequential(*list(net._blocks[18:24]))
        self.layer8 = nn.Sequential(*list(net._blocks[24:26]))
        self.rfb1 = RFB(24, channel)
        self.rfb2 = RFB(32, channel)
        self.rfb3 = RFB(48, channel)
        self.rfb4 = RFB(136, channel)
        self.rfb5 = RFB(384, channel)
        self.agg1 = aggregation(channel)
#        self.sa = SAM(32,32)
        self.fuse = FusionLayer()

        self.predict1 = nn.Conv2d(32, 1, kernel_size=1)
        self.predict2 = nn.Conv2d(32, 1, kernel_size=1)
        self.predict3 = nn.Conv2d(32, 1, kernel_size=1)
        self.predict4 = nn.Conv2d(32, 1, kernel_size=1)
        self.predict5 = nn.Conv2d(32, 1, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x):
#        layer0 = self.layer0(x)
#        layer1 = self.layer1(layer0)
#        layer2 = self.layer2(layer1)
#        layer3 = self.layer3(layer2)
#        layer4 = self.layer4(layer3)

        x1 = self.layer0(x)
        x2 = self.layer1(x1)
#        print(x2.shape)
#        print(self.layer2)
        layer0 = self.layer2(x2)

#        layer0 = nn.Sequential(*list(self.layer2))(x2)
        layer1 = self.layer3(layer0)

        layer2 = self.layer4(layer1)

        layer3 = self.layer5(layer2)

        layer4 = self.layer6(layer3)

        layer5 = self.layer7(layer4)

        layer6 = self.layer8(layer5)

        
        p0 = self.rfb1(layer0)
        predict0 = self.predict1(p0)
        p1 = self.rfb2(layer1)
        predict1 = self.predict2(p1)
        p2 = self.rfb3(layer2)
        predict2 = self.predict3(p2)
        p3 = self.rfb4(layer4)
        predict3 = self.predict4(p3)
        p4 = self.rfb5(layer6)
        predict4 = self.predict5(p4)

        predict0 = F.upsample(predict0, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.upsample(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.upsample(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.upsample(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict4 = F.upsample(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        fuse = [predict0, predict1, predict2, predict3, predict4]
        predict5 = self.fuse(fuse)
        if self.training:
            return predict0, predict1, predict2, predict3, predict4, predict5
        return F.sigmoid(predict5)
