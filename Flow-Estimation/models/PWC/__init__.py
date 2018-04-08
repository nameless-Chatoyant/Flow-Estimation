import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import L2

def cost_volume(x1, x2):
    return x1, x2


class FeaturePyramidExtractor(nn.Module):

    def __init__(self):
        super(FeaturePyramidExtractor, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 2, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 8, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 8, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 8, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 8, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 8, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 96, kernel_size = 3, stride = 2, padding = 8, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, stride = 1, padding = 8, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels = 96, out_channels = 128, kernel_size = 3, stride = 2, padding = 8, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 8, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 192, kernel_size = 3, stride = 2, padding = 8, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels = 192, out_channels = 192, kernel_size = 3, stride = 1, padding = 8, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
    

    def forward(self, x):
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
        out_conv11 = self.conv11(out_conv10)
        out_conv12 = self.conv12(out_conv11)

        return out_conv2, out_conv4, out_conv6, out_conv8, out_conv10, out_conv12
        

class OpticalFlowEstimator(nn.Module):

    def __init__(self):
        super(OpticalFlowEstimator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 115, out_channels = 128, kernel_size = 3, stride = 1, padding = 8, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 8, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 96, kernel_size = 3, stride = 1, padding = 8, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 96, out_channels = 64, kernel_size = 3, stride = 1, padding = 8, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 8, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv6 = nn.Conv2d(in_channels = 32, out_channels = 2, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)

    def forward(self, feature1, feature2, flow):
        warped = F.grid_sample(feature2, flow)
        
        out_conv1 = self.conv1(flow)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        
        return out_conv6

class ContextNet(nn.Module):

    def __init__(self):
        super(ContextNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 8, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 8, dilation = 2, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 8, dilation = 4, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 8, dilation = 8, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 8, dilation = 16, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 8, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv7 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 8, dilation = 1, groups = 1, bias = True)
        pass
    
    def forward(self, x):
        pass


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.feature_pyramid_extractor = FeaturePyramidExtractor()
        self.optical_flow_estimator = OpticalFlowEstimator()
        self.context_net = ContextNet()
        pass
    

    def forward(self, x):
        """
        """
        pass


class Config:
    loss_weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1]
    pass


def loss_func(flow, flow_gt):
    loss = 0
    for level, (flow, flow_gt) in enumerate():
        flow

class PWC:
    def __init__(self):
        self.model = Net()
        self.optimizer = 0
        self.criterion = 0
        pass
    

    def train(self, train_loader, eval_loader):
        self.model.train()
        
        def one_epoch(epoch):
            for batch_idx, (data, target) in enumerate(train_loader):

                # ===============================================
                # Input
                # ===============================================
                img1, img2 = data
                flow = target
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
                loss = self.criterion()

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