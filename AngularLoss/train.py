from torchvision import transforms
from dataset import TruncatedInput, ToTensor
import torch
import os
from tqdm import tqdm
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from dataset import VAD
from dataset import SpeakerTrainDataset, SpeakerTestDataset
from model import ResNet, AngleLoss
from torch.utils.data import DataLoader
from preprocess import cal_eer
import pandas as pd
import torchsummary

class Args:
    def __init__(self):
        self.embedding_size = 512
        self.m = 3
        self.lambda_min = 5 #lambda下限
        self.lambda_max = 1000 #lambda 上限
        self.epochs = 20
        self.batch_size = 128
        self.optimizer = 'sgd'
        self.momentum = 0.9
        self.dampening = 0
        self.lr = 1e-1
        self.lr_decay = 0
        self.wd = 0.00001
        self.model_dir = './model/sgd/'
        self.final_dir = './final_model/sgd/'
        self.start = None
        self.resume = self.final_dir + 'net.pth'
        self.load_it = True
        self.it = None
        self.load_optimizer = True
        self.seed = 123456

        self.use_out = True
        self.use_embedding = True

        self.test_batch_size = 24
        self.transform = transforms.Compose([
            TruncatedInput(input_per_file=1),
            ToTensor(),
        ])

args = Args()
device = torch.device('cuda')
os.makedirs(args.model_dir, exist_ok=True)
os.makedirs(args.final_dir, exist_ok=True)

'''
def adjust_learning_rate(optimizer, epoch):#调整学习率策略，优化器，目前轮数
    if epoch <= 15:
        lr = args.lr
    elif epoch <= 20:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
'''

def train(epoch, model, criterion, optimizer, train_loader):#训练轮数，模型，loss，优化器，数据集读取
    model.train()#初始化模型为训练模式
    adjust_learning_rate(optimizer, epoch)#调整学习率

    sum_loss, sum_samples = 0, 0
    progress_bar = tqdm(enumerate(train_loader))
    for batch_idx, (data, label) in progress_bar:
        sum_samples += len(data)
        data = data.to(device)
        label = label.to(device)  # 数据和标签

        out, _ = model(data, label)#通过模型，输出最后一层和倒数第二层

        loss = criterion(out, label)#loss
        optimizer.zero_grad()
        loss.backward()#bp训练
        optimizer.step()

        sum_loss += loss.item() * len(data)
        progress_bar.set_description(
            'Train Epoch: {:3d} [{:4d}/{:4d} ({:3.3f}%)] Loss: {:.4f}'.format(
                epoch, batch_idx + 1, len(train_loader),
                100. * (batch_idx + 1) / len(train_loader),
                sum_loss / sum_samples))

    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'it': criterion.it,
                'optimizer': optimizer.state_dict()},
               '{}/net_{}.pth'.format(args.model_dir, epoch))#保存当轮的模型到net_{}.pth
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'it': criterion.it,
                'optimizer': optimizer.state_dict()},
               '{}/net.pth'.format(args.final_dir))#保存当轮的模型到net.pth


def test(model, test_loader):#测试，模型，测试集读取
    model.eval()#设置为测试模式

    pairs, similarities_out, similarities_embedding = [], [], []
    progress_bar = tqdm(enumerate(test_loader))
    for batch_idx, (pair, data1, data2) in progress_bar:#按batch读取数据
        pairs.append(pair)
        with torch.no_grad():
            data1, data2 = data1.to(device), data2.to(device)

            out1, embedding1 = model(data1)
            out2, embedding2 = model(data2)
            if args.use_out:#使用最后一层计算余弦相似度
                sim_out = F.cosine_similarity(out1, out2).cpu().data.numpy()
                similarities_out.append(sim_out)
            if args.use_embedding:#使用倒数第二层计算余弦相似度
                sim_embedding = F.cosine_similarity(embedding1, embedding2).cpu().data.numpy()
                similarities_embedding.append(sim_embedding)

            progress_bar.set_description('Test: [{}/{} ({:3.3f}%)]'.format(
                batch_idx + 1, len(test_loader), 100. * (batch_idx + 1) / len(test_loader)))

    pairs = np.concatenate(pairs)
    if args.use_out:
        similarities_out = np.array([sub_sim for sim in similarities_out for sub_sim in sim])
        if VAD:
            csv_file = 'pred_out_vad.csv'
        else:
            csv_file = 'pred_out.csv'
        with open(args.final_dir + csv_file, mode='w') as f:
            f.write('pairID,pred\n')
            for i in range(len(similarities_out)):
                f.write('{},{}\n'.format(pairs[i], similarities_out[i]))
    if args.use_embedding:
        similarities_embedding = np.array([sub_sim for sim in similarities_embedding for sub_sim in sim])
        if VAD:
            csv_file = 'pred_vad.csv'
        else:
            csv_file = 'pred.csv'
        with open(args.final_dir + csv_file, mode='w') as f:
            f.write('pairID,pred\n')
            for i in range(len(similarities_embedding)):
                f.write('{},{}\n'.format(pairs[i], similarities_embedding[i]))

def main():
    torch.manual_seed(args.seed)#设置随机种子

    train_dataset = SpeakerTrainDataset(samples_per_speaker = args.samples_per_speaker)#设置训练集读取
    n_classes = train_dataset.n_classes#说话人数
    print('Num of classes: {}'.format(n_classes))

    model = ResNet(layers=[1, 1, 1, 1], embedding_size=args.embedding_size, n_classes=n_classes, m=args.m).to(device)
    torchsummary.summary(model, (1,161,300))
    if args.optimizer == 'sgd':#优化器使用sgd
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, dampening=args.dampening, weight_decay=args.wd)
    elif args.optimizer == 'adagrad':#优化器使用adagrad
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, lr_decay=args.lr_decay, weight_decay=args.wd)
    else:#优化器使用adam
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = AngleLoss(lambda_min=args.lambda_min, lambda_max=args.lambda_max).to(device)#loss设置

    start = 1
    if args.resume:#是否从之前保存的模型开始
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.start is not None:
                start = start
            else:
                start = checkpoint['epoch'] + 1
            if args.load_it:
                criterion.it = checkpoint['it']
            elif args.it is not None:
                criterion.it = args.it
            if args.load_optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=1, pin_memory=True)

    test_dataset = SpeakerTestDataset(transform=args.transform)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                             num_workers=1, pin_memory=True)

    for epoch in range(start, args.epochs + 1):
        train(epoch, model, criterion, optimizer, train_loader)

        if epoch % 5 == 0:
            test(model, test_loader)#测试
            task = pd.read_csv('task/task.csv', header=None, delimiter = '[ ]', engine='python')
            pred = pd.read_csv(args.final_dir + 'pred.csv', engine='python')
            y_true = np.array(task.iloc[:, 0])
            y_pred = np.array(pred.iloc[:, -1])
            eer, thresh = cal_eer(y_true, y_pred)
            print('EER      : {:.3%}'.format(eer))
            print('Threshold: {:.5f}'.format(thresh))

args.epochs = 50
args.m = 3
args.lambda_min = 5
args.lambda_max = 1500
args.lr = 1e-1
args.model_dir = './model/sgd/vox1/'
args.final_dir = './final_model/sgd/vox1/'
args.start = None
args.use_out = False
args.resume = args.final_dir + 'net.pth'
args.load_it = True
args.load_optimizer = True
args.test_batch_size = 1
args.transform = None

os.makedirs(args.model_dir, exist_ok=True)
os.makedirs(args.final_dir, exist_ok=True)

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 15:
        lr = args.lr
    elif epoch <= 30:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
       
main()    