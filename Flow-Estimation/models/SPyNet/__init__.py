from ..base import Learnable
import torch.nn as nn

"""
https://github.com/sniklaus/pytorch-spynet
"""
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		class Preprocess(nn.Module):
			def __init__(self):
				super(Preprocess, self).__init__()
			# end

			def forward(self, variableInput):
				variableBlue = variableInput[:, 0:1, :, :] - 0.406
				variableGreen = variableInput[:, 1:2, :, :] - 0.456
				variableRed = variableInput[:, 2:3, :, :] - 0.485

				variableBlue = variableBlue / 0.225
				variableGreen = variableGreen / 0.224
				variableRed = variableRed / 0.229

				return torch.cat([variableRed, variableGreen, variableBlue], 1)
			# end
		# end

		class Basic(nn.Module):
			def __init__(self, intLevel):
				super(Basic, self).__init__()

				self.moduleBasic = nn.Sequential(
					nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
					nn.ReLU(inplace=False),
					nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
					nn.ReLU(inplace=False),
					nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
					nn.ReLU(inplace=False),
					nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
					nn.ReLU(inplace=False),
					nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
				)

				if intLevel == 5:
					if arguments_strModel == '3' or arguments_strModel == '4':
						intLevel = 4 # the models trained on the flying chairs dataset do not come with weights for the sixth layer
					# end
				# end

				for intConv in range(5):
					self.moduleBasic[intConv * 2].weight.data.copy_(torch.utils.serialization.load_lua('./models/modelL' + str(intLevel + 1) + '_' + arguments_strModel  + '-' + str(intConv + 1) + '-weight.t7'))
					self.moduleBasic[intConv * 2].bias.data.copy_(torch.utils.serialization.load_lua('./models/modelL' + str(intLevel + 1) + '_' + arguments_strModel  + '-' + str(intConv + 1) + '-bias.t7'))
				# end
			# end

			def forward(self, variableInput):
				return self.moduleBasic(variableInput)
			# end
		# end

		class Backward(nn.Module):
			def __init__(self):
				super(Backward, self).__init__()
			# end

			def forward(self, variableInput, variableFlow):
				if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != variableInput.size(0) or self.tensorGrid.size(2) != variableInput.size(2) or self.tensorGrid.size(3) != variableInput.size(3):
					torchHorizontal = torch.linspace(-1.0, 1.0, variableInput.size(3)).view(1, 1, 1, variableInput.size(3)).expand(variableInput.size(0), 1, variableInput.size(2), variableInput.size(3))
					torchVertical = torch.linspace(-1.0, 1.0, variableInput.size(2)).view(1, 1, variableInput.size(2), 1).expand(variableInput.size(0), 1, variableInput.size(2), variableInput.size(3))

					self.tensorGrid = torch.cat([ torchHorizontal, torchVertical ], 1).cuda()
				# end

				variableGrid = torch.autograd.Variable(data=self.tensorGrid, volatile=not self.training)

				variableFlow = torch.cat([ variableFlow[:, 0:1, :, :] / ((variableInput.size(3) - 1.0) / 2.0), variableFlow[:, 1:2, :, :] / ((variableInput.size(2) - 1.0) / 2.0) ], 1)

				return nn.functional.grid_sample(input=variableInput, grid=(variableGrid + variableFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
			# end
		# end

		self.modulePreprocess = Preprocess()

		self.moduleBasic = nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])

		self.moduleBackward = Backward()
	# end

	def forward(self, variableFirst, variableSecond):
		variableFlow = []

		variableFirst = [ self.modulePreprocess(variableFirst) ]
		variableSecond = [ self.modulePreprocess(variableSecond) ]

		for intLevel in range(5):
			if variableFirst[0].size(2) > 32 or variableFirst[0].size(3) > 32:
				variableFirst.insert(0, nn.functional.avg_pool2d(input=variableFirst[0], kernel_size=2, stride=2))
				variableSecond.insert(0, nn.functional.avg_pool2d(input=variableSecond[0], kernel_size=2, stride=2))
			# end
		# end

		variableFlow = torch.autograd.Variable(data=torch.zeros(variableFirst[0].size(0), 2, int(math.floor(variableFirst[0].size(2) / 2.0)), int(math.floor(variableFirst[0].size(3) / 2.0))).cuda(), volatile=not self.training)

		for intLevel in range(len(variableFirst)):
			variableUpsampled = nn.functional.upsample(input=variableFlow, scale_factor=2, mode='bilinear') * 2.0

			if variableUpsampled.size(2) != variableFirst[intLevel].size(2): variableUpsampled = nn.functional.pad(input=variableUpsampled, pad=[0, 0, 0, 1], mode='replicate')
			if variableUpsampled.size(3) != variableFirst[intLevel].size(3): variableUpsampled = nn.functional.pad(input=variableUpsampled, pad=[0, 1, 0, 0], mode='replicate')

			variableFlow = self.moduleBasic[intLevel](torch.cat([ variableFirst[intLevel], self.moduleBackward(variableSecond[intLevel], variableUpsampled), variableUpsampled ], 1)) + variableUpsampled
		# end

		return variableFlow
	# end
# end

class SPyNet(Learnable):

    def __init__(self):
        self.model = Net()
		self.optimizer = torch.optim.SGD()
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
    

    def predict(self, img1, img2):
        """
        """
		assert(tensorInputFirst.size(1) == tensorInputSecond.size(1))
		assert(tensorInputFirst.size(2) == tensorInputSecond.size(2))

        self.model(img1, img2)
        intWidth = tensorInputFirst.size(2)
		intHeight = tensorInputFirst.size(1)

		assert(intWidth == 640) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
		assert(intHeight == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

		if True:
			tensorInputFirst = tensorInputFirst.cuda()
			tensorInputSecond = tensorInputSecond.cuda()
			tensorOutput = tensorOutput.cuda()
		# end

		if True:
			variableInputFirst = torch.autograd.Variable(data=tensorInputFirst.view(1, 3, intHeight, intWidth), volatile=True)
			variableInputSecond = torch.autograd.Variable(data=tensorInputSecond.view(1, 3, intHeight, intWidth), volatile=True)

			tensorOutput.resize_(2, intHeight, intWidth).copy_(moduleNetwork(variableInputFirst, variableInputSecond).data[0])
		# end

		if True:
			tensorInputFirst = tensorInputFirst.cpu()
			tensorInputSecond = tensorInputSecond.cpu()
			tensorOutput = tensorOutput.cpu()
		# end

		return tensorOutput