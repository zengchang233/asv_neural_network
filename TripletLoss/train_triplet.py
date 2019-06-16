import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import numpy as np
import os
from logger import Logger
from model import OnlineTripletLoss
from utils import AllTripletSelector
from utils import HardestNegativeTripletSelector
from utils import RandomNegativeTripletSelector
from utils import SemihardNegativeTripletSelector
from model import EmbeddingNet, DeepSpeaker, OnlineTripletLoss
from enrollment_test import evaluate, predict
from preprocess import extractFeature
from DeepSpeakerDataset import DeepSpkDataset, BalancedBatchSampler
from eval_metrics import AverageNonzeroTripletsMetric
import argparse
import warnings
warnings.filterwarnings('ignore')
EVAL_DEFINITION_DIR = '/home/zeng/zeng/aishell/af2019-sr-devset-20190312'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',
                        type = str,
                        default = None,
                        help = 'name for log (default : None)')
    parser.add_argument('--resume',
                        type = str,
                        default = None,
                        help = 'checkpoint path (default : None)')
    parser.add_argument('--train',
                        action = 'store_true',
                        help = 'if train, train the model, else, just test')
    parser.add_argument('--maxlen',
                        type = int,
                        default = 300,
                        help = 'max frames of speech (default : 300)')
    parser.add_argument('--selection',
                        type = str,
                        default = 'randomhard',
                        choices = ['randomhard','hardest','semihard', 'all'],
                        help = 'the strategy of triplet chosen (default : randomhard')
    parser.add_argument('--embedding_size',
                        type = int,
                        default = 512,
                        help = 'the size of speaker embedding (default : 512)')
    parser.add_argument('--layers',
                        type = int,
                        default = 4,
                        help = 'the number of network layer (default : 4)') 
    parser.add_argument('--resblk',
                        type = int,
                        default = 3,
                        help = 'the number of resnet block (default : 3)')                       
    parser.add_argument('--wd',
                        type = float,
                        default = 1e-5,
                        help = 'weights decay (default : 1e-5)')                        
    parser.add_argument('--lr',
                        type = float,
                        default = 0.001,
                        help = 'learning rate (default : 0.001)')
    parser.add_argument('--lr_adjust_step',
                        type = int,
                        default = 8,
                        help = 'every step epoches adjust lr (default : 8)')
    parser.add_argument('--lr_decay',
                        type = int,
                        default = 0.5,
                        help = 'lr decay weights (default : 0.5)')
    parser.add_argument('--n_classes',
                        type = int,
                        default = 20,
                        help = 'class number of one batch (default : 20)')
    parser.add_argument('--n_samples',
                        type = int,
                        default = 10,
                        help = 'sample number of one class (default : 10)')
    parser.add_argument('--margin',
                        type = float,
                        default = 1,
                        help = 'triplet loss margin (default : 1)')                        
    parser.add_argument('--eval',
                        action = 'store_true',
                        help = 'print the score of test')
    parser.add_argument('--pretrain_epoch',
                        type = int,
                        default = 5,
                        help = 'softmax pretrain epoch number (default : 5)')                        
    parser.add_argument('--n_epochs',
                        type = int,
                        default = 10,
                        help = 'train epoch number (default : 10)')                           
    parser.add_argument('--trainning-dataset',
                        type = str,
                        default = '/home/zeng/zeng/aishell/wav',
                        help = 'training wav path (default)')
    parser.add_argument('--trainfeature',
                        type = str,
                        default = '/home/zeng/zeng/aishell/mfcc_train_feature',
                        help = 'train feature path (default)')
    parser.add_argument('--model',
                        type = str,
                        default = None,
                        help = 'trained model path (default : None)')
    parser.add_argument('--test-dataset',
                        type = str,
                        default = '/home/zeng/zeng/aishell/af2019-sr-devset-20190312/data',
                        help = 'test wav path (default : None)') 
    parser.add_argument('--testfeature',
                        type = str,
                        default = '/home/zeng/zeng/aishell/pretraindeepspeaker/mfcc_test_feature',
                        help = 'test feature path (default : None)')
    parser.add_argument('--eval-dataset',
                        type = str,
                        default = '/home/zeng/zeng/aishell/af2019-sr-devset-20190312/data',
                        help = 'eval wav path (default)') 
    parser.add_argument('--evalfeature',
                        type = str,
                        default = '/home/zeng/zeng/aishell/pretraindeepspeaker/mfcc_test_feature',
                        help = 'eval feature path (default)')                                                                                                              
    return parser.parse_args()

def fit(train_loader,
        pre_loader,
        model, 
        loss_fn, 
        optimizer, 
        scheduler, 
        pretrain_epoch,
        n_epochs, 
        cuda,
        device, 
        log_interval, 
        log_dir,
        eval_path,
        logger,
        metrics = [],
        evaluatee = True,
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    acc = []
    thres = []
    if pretrain_epoch > start_epoch:
        criterion = CrossEntropyLoss()
        pre_optimizer = optim.SGD(model.parameters(), lr = optimizer.param_groups[0]['lr'], momentum = 0.99)
        pre_scheduler = optim.lr_scheduler.StepLR(pre_optimizer, scheduler.step_size, gamma = scheduler.gamma, last_epoch = -1)
        for epoch in range(start_epoch, pretrain_epoch):
            pre_scheduler.step()
            scheduler.step()
            total_loss, accuracy = pretrain(pre_loader, model, criterion, pre_optimizer, 100, device)
            message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, total_loss) 
            message += '\tAccuracy : {:.4f}'.format(accuracy)
            print(message)

            if evaluatee:
                model.cpu()
                accuracy, threshold = eval(model, log_dir, EVAL_DEFINITION_DIR, eval_path, logger)
                model.to(device)
                acc.append(accuracy)
                thres.append(threshold)
            np.savetxt(log_dir + '/acc.txt',np.array(acc))
            np.savetxt(log_dir + '/thres.txt', np.array(thres))
        start_epoch = pretrain_epoch
        torch.save({'epoch': pretrain_epoch, 'state_dict': model.embedding_net.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    '{}/checkpoint_{}.pth'.format(log_dir, pretrain_epoch))

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, 
                                          model.embedding_net, 
                                          loss_fn, 
                                          optimizer, 
                                          cuda, 
                                          log_interval, 
                                          metrics,
                                          logger,
                                          device)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        print(message)

        # do checkpointing
        torch.save({'epoch': epoch + 1, 'state_dict': model.embedding_net.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    '{}/checkpoint_{}.pth'.format(log_dir, epoch))

        if evaluatee:
            model.cpu()
            accuracy, threshold = eval(model, log_dir, EVAL_DEFINITION_DIR, eval_path, logger)
            model.to(device)
            acc.append(accuracy)
            thres.append(threshold)
        np.savetxt(log_dir + '/acc.txt',np.array(acc))
        np.savetxt(log_dir + '/thres.txt', np.array(thres))

def train_epoch(train_loader, 
                model, 
                loss_fn, 
                optimizer, 
                cuda, 
                log_interval, 
                metrics,
                logger,
                device):
    for metric in metrics:
        metric.reset() # reset the metrics before every epoch start

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.to(device) for d in data)
            if target is not None:
                target = target.to(device)

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,) # size = (batch_size, 2)

        loss_inputs = outputs # 这个表示用来计算loss的inputs
        if target is not None:
            target = (target,)
            loss_inputs += target # 这个应该是把数据添加到tuple里面

        loss_outputs = loss_fn(*loss_inputs) # loss and number of triplets
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)
        
        logger.log_value('Avg not zero triplets', metrics[0].value()).step()
        logger.log_value('loss', loss.item()).step()

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics

def pretrain(train_loader, 
             model,
             loss_fn, 
             optimizer, 
             log_interval,
             device):

    model.train()
    losses = []
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        _, idx = output.max(dim = 1)
        correct += (idx == target).sum().item()
        total += len(target)

        losses.append(loss.item())
        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            message = '\rTrain: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * train_loader.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            message += '\tAccuracy : {:.6f}'.format(correct / total)
            print(message, end = '')
            losses = []

    total_loss /= (batch_idx + 1)
    print()
    return total_loss, correct / total

def eval(model, log_dir, eval_definition_dir, eval_feature_dir, logger = None):
    with torch.no_grad():
        annotation = eval_definition_dir + '/' + 'annotation.csv'
        evaluation = eval_definition_dir + '/' + 'test.csv'
        enroll = eval_definition_dir + '/' + 'enrollment.csv'
        accuracy, threshold = evaluate(model, eval_feature_dir, enroll, evaluation, annotation)
        if logger != None:
            logger.log_value('eval accuracy', accuracy)
            logger.log_value('eval threshold', threshold)
    return accuracy, threshold

def main():
    args = get_args()
    logdir = 'log/{}-emb{}-{}layers-{}resblk-lr{}-wd{}-maxlen{}-alpha10-margin{}'\
             '{}class-{}sample-{}selector'\
             .format(args.name, 
                     args.embedding_size,
                     args.layers,
                     args.resblk,
                     args.lr, 
                     args.wd, 
                     args.maxlen,
                     args.margin,
                     args.n_classes,
                     args.n_samples,
                     args.selection)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    resblock = []
    for i in range(args.layers):
        resblock.append(args.resblk)

    if args.train:
        logger = Logger(logdir)
        if not os.path.exists(args.trainfeature):
            os.mkdir(args.trainfeature)
            extractFeature(args.training-dataset, args.trainfeature)
        trainset = DeepSpkDataset(args.trainfeature, args.maxlen)
        pre_loader = DataLoader(trainset, batch_size = 128, shuffle = True, num_workers = 8)
        train_batch_sampler = BalancedBatchSampler(trainset.train_labels, 
                                                   n_classes = args.n_classes, 
                                                   n_samples = args.n_samples)
        kwargs = {'num_workers' : 1, 'pin_memory' : True}
        online_train_loader = torch.utils.data.DataLoader(trainset, 
                                                          batch_sampler=train_batch_sampler,
                                                          **kwargs) 
        margin = args.margin
        
        embedding_net = EmbeddingNet(resblock,  
                                     embedding_size = args.embedding_size,
                                     layers = args.layers)
        model = DeepSpeaker(embedding_net, trainset.get_num_class())
        device = torch.device('cuda:0')
        model.to(device) # 要在初始化optimizer之前把model转换到GPU上，这样初始化optimizer的时候也是在GPU上
        optimizer = optim.SGD(model.embedding_net.parameters(), 
                              lr = args.lr, 
                              momentum = 0.99,
                              weight_decay = args.wd)
        start_epoch = 0
        if args.resume:
            if os.path.isfile(args.resume):
                print('=> loading checkpoint {}'.format(args.resume))
                checkpoint = torch.load(args.resume)
                start_epoch = checkpoint['epoch']
                model.embedding_net.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('=> no checkpoint found at {}'.format(args.resume))

        pretrain_epoch = args.pretrain_epoch
        
        if args.selection == 'randomhard':
            selector = RandomNegativeTripletSelector(margin)
        if args.selection == 'hardest':
            selector = HardestNegativeTripletSelector(margin)
        if args.selection == 'semihard':
            selector = SemihardNegativeTripletSelector(margin)
        if args.selection == 'all':
            print('warning : select all triplet may take very long time')
            selector = AllTripletSelector()

        loss_fn = OnlineTripletLoss(margin, selector)   
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size = args.lr_adjust_step,
                                        gamma = args.lr_decay,
                                        last_epoch = -1) 
        n_epochs = args.n_epochs
        log_interval = 50
        fit(online_train_loader,
            pre_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            pretrain_epoch,
            n_epochs,
            True,
            device,
            log_interval,
            log_dir = logdir,
            eval_path = args.evalfeature,
            logger = logger,
            metrics = [AverageNonzeroTripletsMetric()],
            evaluatee = args.eval,
            start_epoch = start_epoch)
    else:
        if not os.path.exists(args.testfeature):
            os.mkdir(args.testfeature)
            extractFeature(args.test-dataset, args.testfeature)
        model = EmbeddingNet(resblock,  
                             embedding_size = args.embedding_size,
                             layers = args.layers)
        model.cpu()
        if args.model:
            if os.path.isfile(args.model):
                print('=> loading checkpoint {}'.format(args.model))
                checkpoint = torch.load(args.model)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                print('=> no checkpoint found at {}'.format(args.model))
        thres = np.loadtxt(logdir + '/thres.txt')
        acc = np.loadtxt(logdir + '/acc.txt')
        idx = np.argmax(acc)
        best_thres = thres[idx]
        predict(model, args.testfeature, best_thres)

if __name__ == "__main__":
    main()
