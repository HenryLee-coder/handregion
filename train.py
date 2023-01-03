# 导包
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter

# 数据集的引用
train_dataset = datasets.MNIST('./data/train',
                           train=True,
                           download=True,
                           transform=transforms.ToTensor())
test_dataset = datasets.MNIST('./data/test',
                          train=False,
                          download=True,
                          transform=transforms.ToTensor())
# dataloader使用
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

'''print(train_dataset.data)
print(train_dataset.targets)'''
# 超参数
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 训练类
class Train_v1:
    def __init__(self):
        self.summary = SummaryWriter('logs')




    def __call__(self):
        pass




if __name__ == '__main__':
    pass