import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class WaterBodyDataSet(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):  # 定义txt_path参数
        fh = open(txt_path, 'r')  # 读取txt文件
        imgs = []  # 定义imgs的列表
        for line in fh:
            line = line.rstrip()  # 默认删除的是空白符（'\n', '\r', '\t', ' '）
            words = line.split()  # 默认以空格、换行(\n)、制表符(\t)进行分割，大多是"\"
            imgs.append((words[0], int(words[1])))  # 存放进imgs列表中

        self.imgs = imgs  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]  # fn代表图片的路径，label代表标签
        img = cv2.imread(fn,0)
        img=cv2.resize(img,(1024,1024))
        _, img = cv2.threshold(img, 240, 1, cv2.THRESH_BINARY_INV)
        img=np.reshape(img,(1,1024,1024))
        if self.transform is not None:
            img = self.transform(img)  # 在这里做transform，转为tensor等等
        img = torch.from_numpy(img).float()
        return img, label

    def __len__(self):
        return len(self.imgs)  # 返回图片的长度
