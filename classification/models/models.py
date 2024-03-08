import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

# class AdaptedResNet18():
#     def __init__(self, num_classes):
#         self.model = torchvision.models.resnet18(num_classes=num_classes)
#         self.conv1 = torch.nn.Conv2d(1, 3, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)