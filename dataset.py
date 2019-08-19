from torch.utils import data
import os
import PIL.Image as pimg
import numpy as np
import torch

'''该类用于创建获取数据集类'''


class datasets(data.Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = []
        self.dataset.extend(os.listdir(path))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        label = torch.Tensor(np.array(self.dataset[index].split(".")[1:-1], dtype=np.int))
        img = pimg.open(os.path.join(self.path, self.dataset[index]))
        img_data = torch.Tensor((np.array(img) / 255 - 0.5).transpose(2, 0, 1))
        return img_data, label


if __name__ == '__main__':
    train_img = r"./datasets/train_img"
    test_img = r"./datasets/test_img"
    data = datasets(train_img)
    img = data[1][0]
    label = data[1][1]
    print(label[0:4])
    print(img.shape)
    i = img.numpy().transpose(1, 2, 0)
    i = np.array((i + 0.5) * 255, dtype=np.uint8)
    print(i)
    print(i.shape)
    i = pimg.fromarray(i)
    i.show()
