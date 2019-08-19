import PIL.Image as pimg
import PIL.ImageDraw as draw
import PIL.ImageFont as font
from dataset import datasets
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt

data = datasets(r"./datasets/test_img")
test_data = DataLoader(data, 1, shuffle=True)
net = torch.load("models/net.pth")
net.eval()
# 定义字体
font_path = r"font/simkai.ttf"
font = font.truetype(font_path, size=50)
# 图片保存路径
save_path = r"./save"
loss_mse = nn.MSELoss()
loss_bce = nn.BCELoss()
# 用于统计测试损失和精度
loss_total = 0
acc_confidence_total = 0
acc_coordinate_total = 0
index = 1
for data in test_data:
    img_data, label = data
    # 加载图片
    img_new = np.array((img_data.numpy().transpose(0, 2, 3, 1) + 0.5) * 255, dtype=np.uint8)
    # 降维
    img = pimg.fromarray(img_new[0])
    img_draw = draw.ImageDraw(img)
    img_data = img_data.cuda()
    label = label.cuda()
    # 切片获取置信度和坐标点
    coordinate = label[:, 0:4]
    confidence = label[:, 4:5]
    coordinate_output, confidence_output = net(img_data)
    coordinate_output = coordinate_output * 224
    # 用于判断坐标点再一定范围内的精确度,0.02*224=4.48所以范围控制再5个像素点以内
    for coordinate1, coordinate2 in zip(coordinate, coordinate_output):
        for i in range(4):
            if (coordinate1[i] - coordinate2[i] <= 5 and coordinate1[i] - coordinate2[i] >= 0) or (
                    coordinate1[i] - coordinate2[i] >= -5 and coordinate1[i] - coordinate2[i] <= 0):
                acc_coordinate_total += 1
    plt.ion()
    # 根据置信度判断是否要画框
    if confidence_output.item() >= 0.5:
        # 获取矩形的坐标点
        x1 = coordinate_output[0][0].item()
        y1 = coordinate_output[0][1].item()
        x2 = coordinate_output[0][2].item()
        y2 = coordinate_output[0][3].item()
        img_draw.rectangle(xy=(x1, y1, x2, y2), outline="red", width=3)
        # 置信度保留两位小数
        img_draw.text(xy=(10, 10), text="{:.2f}".format(confidence_output.item()), fill="white", font=font)
    else:
        img_draw.text(xy=(10, 10), text="{:.2f}".format(confidence_output.item()), fill="black", font=font)
    plt.imshow(img)
    plt.pause(1)
    plt.ioff()
    # 保存图片
    img.save("{0}/{1}.png".format(save_path, str(index)))
    index += 1
    # 计算损失
    coordinate_loss = loss_mse(coordinate_output, coordinate)
    confidence_loss = loss_bce(confidence_output, confidence)
    loss = coordinate_loss + confidence_loss
    loss_total += loss.item()
    # GPU转CPU
    confidence_output = confidence_output.cpu().detach().numpy()
    confidence = confidence.cpu().detach().numpy()
    # 置信度转one-hot
    confidence_output = np.where(confidence_output <= 0.5, 0, 1)
    acc_confidence = (confidence == confidence_output).sum()
    acc_confidence_total += acc_confidence

average_loss = loss_total / len(test_data)
average_confidence_acc = acc_confidence_total / len(test_data)
average_coordinate_acc = acc_coordinate_total / 4 / len(test_data)
print("精度为:{}".format(average_loss))
print("置信度的精度为:{}%".format(average_confidence_acc * 100))
print("坐标的精度为:{}%".format(average_coordinate_acc * 100))
