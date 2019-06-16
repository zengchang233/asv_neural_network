from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import math

class AngleLinear(nn.Module):#定义最后一层
    def __init__(self, in_features, out_features, m=3, phiflag=True):#输入特征维度，输出特征维度，margin超参数
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))#本层权重
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)#初始化权重，在第一维度上做normalize
        self.m = m
        self.phiflag = phiflag
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]#匿名函数,用于得到cos_m_theta

    @staticmethod
    def myphi(x, m):
        x = x * m
        return 1 - x ** 2 / math.factorial(2) + x ** 4 / math.factorial(4) - x ** 6 / math.factorial(6) +\
               x ** 8 / math.factorial(8) - x ** 9 / math.factorial(9)

    def forward(self, x):#前向过程，输入x
        w = self.weight

        ww = w.renorm(2, 1, 1e-5).mul(1e5)#方向0上做normalize
        x_len = x.pow(2).sum(1).pow(0.5)
        w_len = ww.pow(2).sum(0).pow(0.5)

        cos_theta = x.mm(ww)
        cos_theta = cos_theta / x_len.view(-1, 1) / w_len.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)#由m和/cos(/theta)得到cos_m_theta
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k#得到/phi(/theta)
        else:
            theta = cos_theta.acos()#acos得到/theta
            phi_theta = self.myphi(theta, self.m)#得到/phi(/theta)
            phi_theta = phi_theta.clamp(-1*self.m, 1)#控制在-m和1之间

        cos_theta = cos_theta * x_len.view(-1, 1)
        phi_theta = phi_theta * x_len.view(-1, 1)
        output = [cos_theta, phi_theta]#返回/cos(/theta)和/phi(/theta)
        return output


class AngleLoss(nn.Module):#设置loss，超参数gamma，最小比例，和最大比例
    def __init__(self, gamma=0, lambda_min=5, lambda_max=1500):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def forward(self, x, y): #分别是output和target
        self.it += 1
        cos_theta, phi_theta = x #output包括上面的[cos_theta, phi_theta]
        y = y.view(-1, 1)

        index = cos_theta.data * 0.0
        index.scatter_(1, y.data.view(-1, 1), 1)#将label存成稀疏矩阵
        index = index.byte()
        index = Variable(index)

        lamb = max(self.lambda_min, self.lambda_max / (1 + 0.1 * self.it))#动态调整lambda，来调整cos(\theta)和\phi(\theta)的比例
        output = cos_theta * 1.0
        output[index] -= cos_theta[index]*(1.0+0)/(1 + lamb)#减去目标\cos(\theta)的部分
        output[index] += phi_theta[index]*(1.0+0)/(1 + lamb)#加上目标\phi(\theta)的部分

        logpt = F.log_softmax(output, dim = 1)
        logpt = logpt.gather(1, y)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss


class ReLU20(nn.Hardtanh):#relu
    def __init__(self, inplace=False):
        super(ReLU20, self).__init__(0, 20, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str


def conv3x3(in_planes, out_planes, stride=1):#3x3卷积，输入通道，输出通道，stride
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):#定义block

    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None):#输入通道，输出通道，stride，下采样
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = ReLU20(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out#block输出


class ResNet(nn.Module):#定义resnet
    def __init__(self, layers, block=BasicBlock, embedding_size=None, n_classes=1000, m=3):#block类型，embedding大小，分类数，maigin大小
        super(ResNet, self).__init__()
        if embedding_size is None:
            embedding_size = n_classes

        self.relu = ReLU20(inplace=True)

        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.in_planes = 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer2 = self._make_layer(block, 128, layers[1])

        self.in_planes = 256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.layer3 = self._make_layer(block, 256, layers[2])

        self.avg_pool = nn.AdaptiveAvgPool2d([4, 1])

        self.fc = nn.Sequential(
            nn.Linear(self.in_planes * 4, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

        self.angle_linear = AngleLinear(in_features=embedding_size, out_features=n_classes, m=m)

        for m in self.modules():#对于各层参数的初始化
            if isinstance(m, nn.Conv2d):#以2/n的开方为标准差，做均值为0的正态分布
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):#weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):#weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, target=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        logit = self.angle_linear(x)
        return logit, x #返回最后一层和倒数第二层的表示
