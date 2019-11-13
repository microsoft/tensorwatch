import torch
import torchvision.models
import tensorwatch as tw

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
writer = SummaryWriter()
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(1, 2),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(1, 1),
            nn.Dropout(0.5),
            nn.Linear(110,30)
            )

    def forward(self, x):
        return self.layer(x)


net = Net()
args = torch.ones([1, 3, 224, 224])
# writer.add_graph(net, args)
# writer.close()

#vgg16_model = torchvision.models.vgg16()

drawing = tw.draw_model(net, [1, 3, 224, 224])
drawing.save('abc2.png')

input("Press any key")