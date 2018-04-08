import torch.nn as nn
from .convlstm import ConvLSTM


class ContrastDetector(nn.Module):
    """最小的receptive field
    
    """
    def __init__(self):
        super(ContrastDetector, self).__init__()

class NetConvexityDetector(nn.Module):
    """第二小的receptive field
    
    """
    def __init__(self):
        super(NetConvexityDetector, self).__init__()

class MovingEdgeDetector(nn.Module):
    """第二大的receptive field
    
    """
    def __init__(self):
        super(MovingEdgeDetector, self).__init__()

class NetDimmingDetector(nn.Module):
    """最大的receptive field
    
    """
    def __init__(self):
        super(NetDimmingDetector, self).__init__()


class Net(nn.Module):
    

    def __init__(self):
        super(Net, self).__init__()
        self.conv_lstm = ConvLSTM(input_size = None, input_dim = None, hidden_dim = None, kernel_size = None, num_layers = None, batch_first = False, bias = True, return_all_layers = False)


    def forward(self, img1, img2):

        con
        img_h, img_w = img1.size()[2:]
        flow = self.conv_lstm(img1)
        
        return flow




class Config:
    max_epoches = 80


class OurFrogNet:
    

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
