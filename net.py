'''
数据集 MNIST (N, 1*28*28)  类别10， 对输出的结果进行激活数据控制在1的softmax
导入数据集
构建网络
train
test
predict
'''

import torch
from torch import nn

class Net_v1(nn.Module):
    def __init__(self):
        super(Net_v1, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(1*28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 68),
            nn.ReLU(),
            nn.Linear(68, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc_layer(x)


if __name__ == '__main__':
    net = Net_v1()
    x = torch.randn(1, 1*28*28)
    print(net(x).shape)
