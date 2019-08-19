import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
from net import trainNet
from dataset import datasets
import torch

'''该类用于训练模型'''

data = datasets(r"./datasets/train_img")
if os.path.exists("models/net.pth"):
    net = torch.load("models/net.pth")
else:
    net = trainNet().cuda()

train_data = DataLoader(data, batch_size=10, shuffle=True, drop_last=True)

loss_mse = nn.MSELoss()
loss_bce = nn.BCELoss()
optim = torch.optim.Adam(net.parameters())

for epoch in range(10):
    for i, (x, y) in enumerate(train_data):
        x = x.cuda()
        y = y.cuda()
        # 切片获取坐标和置信度
        coordinate = y[:, 0:4] / 224
        confidence = y[:, 4:5]
        coordinate_output, confidence_output = net(x)
        coordinate_loss = loss_mse(coordinate_output, coordinate)
        confidence_loss = loss_bce(confidence_output, confidence)
        loss_total = coordinate_loss + confidence_loss
        optim.zero_grad()
        loss_total.backward()
        optim.step()
        if i % 100 == 0:
            print("第{0}轮第{1}批的损失为:{2}".format(epoch + 1, i + 1, loss_total.item()))
            # 将置信度与坐标转为numpy便于算精度
            confidence_output = confidence_output.cpu().detach().numpy()
            confidence = confidence.cpu().detach().numpy()
            coordinate_output = coordinate_output.cpu().detach().numpy()
            coordinate = coordinate.cpu().detach().numpy()
            # 置信度转one-hot
            confidence_output = np.where(confidence_output <= 0.5, 0, 1)
            acc_confidence = np.mean(np.array(confidence_output == confidence), dtype=np.float32)
            # 用于判断坐标点再一定范围内的精确度以及追踪成功的坐标数,0.02*224=4.48所以范围控制再5个像素点以内
            sum = 0
            for coordinate1, coordinate2 in zip(coordinate, coordinate_output):
                for j in range(4):
                    if (coordinate1[j] - coordinate2[j] <= 0.02 and coordinate1[j] - coordinate2[j] >= 0) or (
                            coordinate1[j] - coordinate2[j] >= -0.02 and coordinate1[j] - coordinate2[j] <= 0):
                        sum += 1
            acc_coordinate = sum / 4 / train_data.batch_size
            acc_total = (acc_confidence + acc_coordinate) / 2
            print("第{0}轮第{1}批的精度为:{2}%".format(epoch + 1, i + 1, acc_total * 100))
            torch.save(net, "models/net.pth")
print("训练结束")
