import torch.nn as nn
import torch.nn.functional as F

'''该类为训练的神经网络'''


class trainNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入图片为224*224*3
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )  # 输出图片为112*112*256

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )  # 输出图片为56*56*128

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )  # 输出图片为28*28*64

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, 1, 1, 0),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Conv2d(32, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )  # 输出图片为14*14*64

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 32, 1, 1, 0),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Conv2d(32, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )  # 输出图片为7*7*64

        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )  # 输出图片为5*5*32

        # 用于置信度
        self.linear1 = nn.Sequential(
            nn.Linear(5 * 5 * 32, 1),
            nn.BatchNorm1d(1),
        )

        # 用于坐标
        self.linear2 = nn.Sequential(
            nn.Linear(5 * 5 * 32, 4),
            nn.BatchNorm1d(4),
            nn.ReLU()
        )

    def forward(self, data):
        y1 = self.conv1(data)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        y5 = self.conv5(y4)
        y6 = self.conv6(y5)
        y6 = y6.reshape(y6.size(0), -1)
        confidence = F.sigmoid(self.linear1(y6))
        coordinate = self.linear2(y6)
        return coordinate, confidence
