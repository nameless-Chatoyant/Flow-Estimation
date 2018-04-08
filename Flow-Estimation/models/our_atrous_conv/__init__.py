from ..base import Learnable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 2, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.ReLU(inplace = True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 2, dilation = 2, groups = 1, bias = True),
            nn.ReLU(inplace = True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 4, dilation = 4, groups = 1, bias = True),
            nn.ReLU(inplace = True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 8, dilation = 8, groups = 1, bias = True),
            nn.ReLU(inplace = True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 16, dilation = 16, groups = 1, bias = True),
            nn.ReLU(inplace = True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.ReLU(inplace = True))
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 2, dilation = 2, groups = 1, bias = True),
            nn.ReLU(inplace = True))
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 4, dilation = 4, groups = 1, bias = True),
            nn.ReLU(inplace = True))
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 8, dilation = 8, groups = 1, bias = True),
            nn.ReLU(inplace = True))
        self.conv10 = nn.Conv2d(in_channels = 64, out_channels = 2, kernel_size = 3, stride = 1, padding = 16, dilation = 16, groups = 1, bias = True)
        

    def forward(self, img1, img2):
        x = torch.cat([img1, img2], dim = 1)
        print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)

        return x


class Config:
    max_epoches = 80

class OurAtrousConv(Learnable):
    def __init__(self):
        self.model = Net().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), 4e-5)
    
    

    def train(self, train_loader, eval_loader):
        self.model.train()
        
        def one_epoch(epoch):
            for batch_idx, (data, target) in enumerate(train_loader):

                # ===============================================
                # Input
                # ===============================================
                img1, img2 = data
                flow = target
                flow = flow.float()
                img1, img2, flow = Variable(img1).cuda(), Variable(img2).cuda(), Variable(flow).cuda()
                img1 /= 255.0
                img2 /= 255.0


                # ===============================================
                # Forward
                # ===============================================
                predicted_flow = self.model(img1, img2)


                # ===============================================
                # Loss Function
                # ===============================================
                # Compute Losses
                loss = F.mse_loss(predicted_flow, flow)

                # compute gradient and do Adam step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                # ===============================================
                # Summary
                # ===============================================
                print('one_epoch')
        for i in range(Config.max_epoches):
            one_epoch(i)
    

    def eval(self):
        pass


if __name__ == '__main__':
    from torch.autograd import Variable
    from torch import Tensor
    import torch

    import numpy as np


    net = Net()
    x = Variable(Tensor(np.ones((8,3,224,224))))
    net.forward(x, x)
