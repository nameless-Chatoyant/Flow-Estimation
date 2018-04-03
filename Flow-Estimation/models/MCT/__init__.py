import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class CoarseFlowNet(nn.Module):
    
    def __init__(self):
        super(CoarseFlowNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 2, out_channels = 24, kernel_size = 5, stride = 2, padding = 2, dilation = 1, groups = 1, bias = True),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 24, out_channels = 24, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 24, out_channels = 24, kernel_size = 5, stride = 2, padding = 2, dilation = 1, groups = 1, bias = True),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 24, out_channels = 24, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 24, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.Tanh())
        self.upscale = nn.PixelShuffle(4)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.upscale(x)

        return x

class FineFlowNet(nn.Module):
    
    def __init__(self):
        super(FineFlowNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 4, out_channels = 24, kernel_size = 5, stride = 2, padding = 2, dilation = 1, groups = 1, bias = True),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 24, out_channels = 24, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 24, out_channels = 24, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 24, out_channels = 24, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 24, out_channels = 8, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.Tanh())
        self.upscale = nn.PixelShuffle(2)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.upscale(x)

        return x


# def motion_estimation(reference, img):
#     """compute optical flow from img to reference
#     """
#     l = tf.concat((reference, img), axis = -1) # (b, h, w, 2)
#     print('l',l)
#     coarse_flow = coarse_flow_estimation(l) # (b, h, w, 2)
#     coords = get_coords(h, w) # (b, h, w, 2)
#     # coarse_flow is (-1, 1)
#     mapping = coords - coarse_flow * h / 2
#     sampled = BackwardWarping('warp.1', [reference, mapping], borderMode='constant') # (b, h, w, 1)
#     l = tf.concat((reference, img, coarse_flow, sampled), axis = -1) # (b, h, w, 5)
#     fine_flow = fine_flow_estimation(l) # (b, h, w, 2)

#     # coarse_flow & fine_flow are both (-1, 1) (after tanh)
#     return coarse_flow + fine_flow # shape: (b, h, w, 2) range: ()

class Net(nn.Module):
    

    def __init__(self):
        super(Net, self).__init__()
        self.coarse_flow_net = CoarseFlowNet()
        self.fine_flow_net = FineFlowNet()

    
    def forward(self, img1, img2):
        x = torch.cat([img1, img2], dim = 1)
        coarse_flow = self.coarse_flow_net(x)
        coords = []
        mapping = []
        sampled = []
        x = torch.cat([img1, img2, coarse_flow], dim = 1)
        fine_flow = self.fine_flow_net(x)
        return coarse_flow + fine_flow


class Config:
    max_epoches = 80


class MCT:
    

    def __init__(self):
        self.model = Net()
        self.optimizer = torch.optim.Adam(self.model.parameters(), 4e-5)
    

    def train(self, train_loader, eval_loader):
        self.model.train()
        
        def one_epoch(epoch):
            for batch_idx, (img1, img2, flow) in enumerate(train_loader):

                # ===============================================
                # Input
                # ===============================================
                img1 = img1.float()
                img2 = img2.float()
                img1, img2, flow = Variable(img1), Variable(img2), Variable(flow)
                img1 /= 255.0
                img2 /= 255.0



                # ===============================================
                # Forward
                # ===============================================
                self.model(img1, img2)


                # ===============================================
                # Loss Function
                # ===============================================
                # Compute Losses
                loss = Variable(torch.Tensor(np.array(0)))

                # compute gradient and do Adam step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                # ===============================================
                # Summary
                # ===============================================
        
        for i in range(Config.max_epoches):
            one_epoch(i)


    def eval(self):
        pass
