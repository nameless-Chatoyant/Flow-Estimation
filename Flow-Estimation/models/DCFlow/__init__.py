from ..base import Learnable
import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, mode = 'train'):
        super(Net, self).__init__()

        self.mode = mode
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 4, out_channels = 32, kernel_size = 7, stride = 1, padding = 2, dilation = 1, groups = 1, bias = True),
            nn.ELU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 7, stride = 1, padding = 2, dilation = 1, groups = 1, bias = True),
            nn.ELU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 7, stride = 1, padding = 2, dilation = 1, groups = 1, bias = True),
            nn.ELU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 2, dilation = 1, groups = 1, bias = True),
            nn.ELU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 2, dilation = 1, groups = 1, bias = True),
            nn.ELU(inplace=True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 2, dilation = 1, groups = 1, bias = True),
            nn.ELU(inplace=True))
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 7, stride = 1, padding = 2, dilation = 1, groups = 1, bias = True),
            nn.ELU(inplace=True))
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 7, stride = 1, padding = 2, dilation = 1, groups = 1, bias = True),
            nn.ELU(inplace=True))
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 7, stride = 1, padding = 2, dilation = 1, groups = 1, bias = True),
            nn.ELU(inplace=True))
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 7, stride = 1, padding = 2, dilation = 1, groups = 1, bias = True),
            nn.ELU(inplace=True))
        if self.mode == 'train':
            self.detour1 = nn.Conv2d(in_channels = 32, out_channels = 2, kernel_size = 7, stride = 1, padding = 2, dilation = 1, groups = 1, bias = True)
            self.detour2 = nn.Conv2d(in_channels = 64, out_channels = 2, kernel_size = 7, stride = 1, padding = 2, dilation = 1, groups = 1, bias = True)
            self.detour3 = nn.Conv2d(in_channels = 128, out_channels = 2, kernel_size = 7, stride = 1, padding = 2, dilation = 1, groups = 1, bias = True)
            self.detour4 = nn.Conv2d(in_channels = 128, out_channels = 2, kernel_size = 7, stride = 1, padding = 2, dilation = 1, groups = 1, bias = True)
            self.detour5 = nn.Conv2d(in_channels = 128, out_channels = 2, kernel_size = 7, stride = 1, padding = 2, dilation = 1, groups = 1, bias = True)
            self.detour6 = nn.Conv2d(in_channels = 128, out_channels = 2, kernel_size = 7, stride = 1, padding = 2, dilation = 1, groups = 1, bias = True)
            self.detour7 = nn.Conv2d(in_channels = 256, out_channels = 2, kernel_size = 7, stride = 1, padding = 2, dilation = 1, groups = 1, bias = True)
            self.detour8 = nn.Conv2d(in_channels = 256, out_channels = 2, kernel_size = 7, stride = 1, padding = 2, dilation = 1, groups = 1, bias = True)
            self.detour9 = nn.Conv2d(in_channels = 256, out_channels = 2, kernel_size = 7, stride = 1, padding = 2, dilation = 1, groups = 1, bias = True)
        self.detour10 = nn.Conv2d(in_channels = 256, out_channels = 2, kernel_size = 7, stride = 1, padding = 2, dilation = 1, groups = 1, bias = True)


    def forward(self, sparse_flow_map, mask, edge_map):
        """
        sparse_flow_map: (b, 2, h, w)
        mask: (b, 1, h, w)
        edge_map: (b, 1, h, w)
        """
        x = torch.cat([sparse_flow_map, mask, edge_map], dim = 1)
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)
        out_conv8 = self.conv8(out_conv7)
        out_conv9 = self.conv9(out_conv8)
        out_conv10 = self.conv10(out_conv9)

        if self.mode == 'train':
            out_detour1 = self.detour1(out_conv1)
            out_detour2 = self.detour2(out_conv2)
            out_detour3 = self.detour3(out_conv3)
            out_detour4 = self.detour4(out_conv4)
            out_detour5 = self.detour5(out_conv5)
            out_detour6 = self.detour6(out_conv6)
            out_detour7 = self.detour7(out_conv7)
            out_detour8 = self.detour8(out_conv8)
            out_detour9 = self.detour9(out_conv9)
        out_detour10 = self.detour10(out_conv10)

        return out_detour1, out_detour2, out_detour3, out_detour4, out_detour5, out_detour6, out_detour7, out_detour8, out_detour9, out_detour10 if self.mode == 'train' else out_detour10


def compute_loss(outputs_detour):

    def epe(flow1, flow2): return F.pairwise_distance(flow1, flow2)
    def epe_loss(inferenced, target): return epe(inferenced, target) / target.view(-1).size()[0]
    
    def lateral_dependency_loss(inferenced, target):
        shift_inferenced1 = inferenced
        shift_inferenced2 = inferenced
        shift_target1 = target
        shift_target2 = target

        return ((epe(inferenced, shift_inferenced1) - epe(target, shift_target1)).abs().sum() + \
                (epe(inferenced, shift_inferenced2) - epe(target, shift_target2)).abs().sum()) / target.view(-1).size()[0]

    total_loss = 0
    for out in outputs[:-1]:
        total_loss += 0.5 * (epe_loss(out, target) + lateral_dependency_loss(out, target))
    
    out = outputs_detour[-1]
    total_loss += (epe_loss(out, target) + lateral_dependency_loss(out, target))

    return total_loss

class DCFlow(Learnable):
    def __init__(self):
        self.model = Net()

        self.criterion = 0
        pass
    
    
    def train(self, train_loader, eval_loader):
        pass

        # Downscale


        # Upscale
    
    
    def eval(self):
        pass