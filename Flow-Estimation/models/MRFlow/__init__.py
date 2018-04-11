import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import L2


class Config:
    loss_weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1]
    pass


def loss_func(flow, flow_gt):
    loss = 0
    for level, (flow, flow_gt) in enumerate():
        flow

class MRFlow:
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


    def predict(self):
        pass
        