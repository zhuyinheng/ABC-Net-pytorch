import argparse
import os
import shutil
import time
import sys
import gc
import platform
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
cwd = os.getcwd()
sys.path.append(cwd + '/../')
sys.path.append(cwd + '/networks/')
sys.path.append(cwd)

import model_list as model_list
import util as util


# set the seed
torch.manual_seed(1)

# torch.cuda.manual_seed(1)

import datasets as datasets
# from datasets import transforms
# import datasets.transforms as transforms
print(cwd)
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                    help='model architecture (default: alexnet)')
parser.add_argument('--data', metavar='DATA_PATH', default='./data/',
                    help='path to imagenet data (default: ./data/)')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--base_number', default=2, type=int,
                    metavar='N', help='base_number (default: 2)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.90, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    default=False, help='use pre-trained model')
parser.add_argument('--cuda', dest='cuda', action='store_true',
                    help='running on no cuda')
best_prec1 = 0

# define global bin_op
bin_op = None

# define optimizer
optimizer=None

def main():
    global args, best_prec1
    args = parser.parse_args()

    if platform.system() =="Windows":
        args.nocuda=True
    else:
        args.nocuda = False

    # create model
    if args.arch == 'alexnet':
        model = model_list.alexnet(pretrained=args.pretrained,base_number=args.base_number)
        input_size = 227
    else:
        raise Exception('Model not supported yet')

    model.features = torch.nn.DataParallel(model.features)
    if args.nocuda == False:
        model.cuda()

    # define loss function (criterion) and optimizer
    if args.nocuda == False:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    global optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                 weight_decay=args.weight_decay)
    # random initialization
    if not args.pretrained:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                c = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, 1.0 / c)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data = m.weight.data.zero_().add(1.0)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # original saved file with DataParallel
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            print(checkpoint)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.nocuda==False:
        cudnn.benchmark = True

    # # Data loading code

    # if you want to use pre-prosecess in used in caffe:
    # transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.RandomResizedCrop(input_size),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x * 255),
    #     transforms.Lambda(lambda x: torch.cat(reversed(torch.split(x, 1, 0)))),
    #     transforms.Lambda(lambda x: x - torch.Tensor([103.939, 116.779, 123.68]).view(3, 1, 1).expand(3, 227, 227))
    # ])
    # transform_val = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.CenterCrop(input_size),
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x * 255),
    #     transforms.Lambda(lambda x: torch.cat(reversed(torch.split(x, 1, 0)))),
    #     transforms.Lambda(lambda x: x - torch.Tensor([103.939, 116.779, 123.68]).view(3, 1, 1).expand(3, 227, 227))
    # ])

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
        					 std = [0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
        					 std = [0.229, 0.224, 0.225]),
    ])


    traindir = os.path.join(args.data, 'ILSVRC2012_img_train')
    valdir = os.path.join(args.data, 'ILSVRC2012_img_val')
    train_dataset = datasets.ImageFolder(traindir, transform, mapfile=os.path.join(args.data, "ImageNet12_train.txt"))
    val_dataset = datasets.ImageFolder(valdir, transform_val, mapfile=os.path.join(args.data, "ImageNet12_val.txt"))

    if args.nocuda==False:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers)

    print(model)

    # define the binarization operator
    global bin_op
    bin_op = util.BinOp(model)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.nocuda==False:
            target = target.cuda(async=True)
        if args.nocuda == False:
            input_var = torch.autograd.Variable(input).cuda()
        else:
            input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # process the weights including binarization
        bin_op.binarization()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

        # because the training process is too slow
        if i % 100 == 99:
            save_checkpoint({
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, False,filename="checkpoint_every_100_batches.pth.tar")
        gc.collect()


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bin_op.binarization()
    for i, (input, target) in enumerate(val_loader):

        if args.nocuda == False:
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True).cuda()
            target_var = torch.autograd.Variable(target, volatile=True)
        else:
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))
    bin_op.restore()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 25 epochs"""
    lr = args.lr * (0.1 ** (epoch // 25))
    print('Learning rate:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print(pred)
    # print(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
