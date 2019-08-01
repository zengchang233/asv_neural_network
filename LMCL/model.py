from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import math

class OnlineTripletLoss(nn.Module):
    def __init__(self, centers, margin, selector = 'hardest', cpu=False):
        super(OnlineTripletLoss, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.centers = F.normalize(centers)
        self.centers.requires_grad = False
        self.selector = selector

    def forward(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        cos_matrix = F.linear(embeddings, self.centers)# cos_matrix batch_size * 1211
        rows = torch.arange(embeddings.size(0))
        positive_cos = cos_matrix[rows, labels].view(-1,1) # 32 * 1
        idx = torch.ones((embeddings.size(0), self.centers.size(0)), dtype = rows.dtype) # 32 * 1211
        idx[rows, labels] = 0
        negative_cos_matrix = cos_matrix[idx > 0].view(embeddings.size(0), -1) # 32 * 1210
        loss_values = negative_cos_matrix + self.margin - positive_cos # 求出所有的loss 32 * 1210
        if self.selector == 'hardest': # 挑选出最大的loss
            loss_value, _ = torch.max(loss_values, dim = 1)
        if self.selector == 'hard':
            pass
        if self.selector == 'semihard':
            pass
        losses = F.relu(loss_value.view(-1,1))
        return losses.mean(), (loss_value > 0).sum().item() 

class LMCL(nn.Module):
    def __init__(self, embedding_size, num_classes, s, m):
        super(LMCL, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weights = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.kaiming_normal_(self.weights)

    def forward(self, embedding, label):
        assert embedding.size(1) == self.embedding_size, 'embedding size wrong'
        logits = F.linear(F.normalize(embedding), F.normalize(self.weights))
        margin = torch.zeros_like(logits)
        margin.scatter_(1, label.view(-1, 1), self.m)
        m_logits = self.s * (logits - margin)
        return logits, m_logits, self.s * F.normalize(embedding), F.normalize(self.weights)

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
    def __init__(self, layers, block=BasicBlock, embedding_size=None, n_classes=1000, s = 8, m=0.2):#block类型，embedding大小，分类数，maigin大小
        super(ResNet, self).__init__()
        if embedding_size is None:
            embedding_size = n_classes

        self.relu = ReLU20(inplace=True)

        self.in_planes = 128
        self.conv1 = nn.Conv2d(1, self.in_planes, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.in_planes, layers[0])

        self.in_planes = 256
        self.conv2 = nn.Conv2d(128, self.in_planes, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(self.in_planes)
        self.layer2 = self._make_layer(block, self.in_planes, layers[1])

        self.in_planes = 512
        self.conv3 = nn.Conv2d(256, self.in_planes, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(self.in_planes)
        self.layer3 = self._make_layer(block, self.in_planes, layers[2])

        self.avg_pool = nn.AdaptiveAvgPool2d([4, 1])

        self.fc = nn.Sequential(
            nn.Linear(self.in_planes * 4, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

        #self.angle_linear = AngleLinear(in_features=embedding_size, out_features=n_classes, m=m)
        self.lmcl = LMCL(embedding_size, n_classes, s, m)

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
        for _ in range(1, blocks):
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

        #logit = self.angle_linear(x)
        if target is None:
            return x, F.normalize(x)
        else:
            _, m_logit, __, ___ = self.lmcl(x, target)
            return m_logit, x #返回最后一层和倒数第二层的表示
