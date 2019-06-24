from torch import nn
import torch.nn.functional as F
import torch
import math

# Large Margin Cosine Loss, a modified softmax loss for speaker verification
# More detail information please read this paper: CosFace: Large Margin Cosine Loss for Deep Face Recognition
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
        margin.scatter_(1, label.view(-1,1), self.m)
        m_logits = self.s * (logits - margin)
        return logits, m_logits

class ReLU20(nn.Hardtanh): # relu
    def __init__(self, inplace=False):
        super(ReLU20, self).__init__(0, 20, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None):
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
        return out


class EmbeddingNet(nn.Module): # define the original resnet whose output is normalized
    def __init__(self, layers, block=BasicBlock, embedding_size=None, n_classes=1000):
        '''
        Params:
            layers: residual block number
            block: residual block
            embedding_size: output vector size
            n_classes: the number of speakers
        '''
        super(EmbeddingNet, self).__init__()
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

        for m in self.modules(): # initialize the parameters of model
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d): # weight is set as 1，bias is set as 0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d): # weight is set as 1，bias is set as 0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
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

        return x, F.normalize(x)


# ResNet-like model, the output of the last hidden layer is normalized to 1 and no activation such as relu.
class ResNet(nn.Module):
    def __init__(self, layers, block=BasicBlock, embedding_size=None, n_classes=1211, s = 20, m = 0.2):
        '''
        Params:
            s: hyperparameter in lmcl, it is used for fast convergence, it has lower limitation 
               according to the number of speakers. Please read the cosface paper for more detail.
            m: hyperparameter in lmcl, it is used for more compact embedding, larger m, more compact.
               the upper limitation is about 0.35
        '''
        super(ResNet, self).__init__()
        self.embedding_net = EmbeddingNet(layers, block, embedding_size, n_classes)
        self.lmcl = LMCL(embedding_size, n_classes, s, m)

    def forward(self, x, target = None):
        out, embedding = self.embedding_net(x)
        if target is None:
            return out, embedding
        else:
            logits, m_logits = self.lmcl(out, target)
            return m_logits, logits
