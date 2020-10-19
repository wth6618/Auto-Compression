import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torch.nn.utils.prune as prune

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import *
from prune import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--device_name', '-d', default=9 ,type=int, help='cuda device number')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if device == 'cuda':
    torch.cuda.set_device(0)

# Data
print('==> Preparing data..')



transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False)

print('Prepare data done!')

#net = ResNet18()
net = MobileNet()

#checkpoint = torch.load('./baselines/resNet18.pth', map_location='cuda:0')
checkpoint = torch.load('./baselines/mobileNet.pth', map_location='cuda:0')

net.load_state_dict(checkpoint['state_dict'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
print("loaded model with accuracy {}".format( best_acc))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
net = net.to(device)

# pruner = structured_L1(net, pruning_rate=[0.1,0.2,0.5], skip=[0,7,12,17], depth=18)
pruner = structured_L1(net, pruning_rate=0.6)
#pruner = Unstructured(net, pruning_rate=0.8, global_wise=True)
pruner.step()


pruner.zero_params()





def train(epoch, net):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print("accuracy: {}".format(acc))
    if acc > best_acc:
        print('Saving..')
        state = {
            'state_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('pruned_model'):
            os.mkdir('pruned_model')
        torch.save(state, './pruned_model/mobileNet.pth')
        best_acc = acc
    return (acc, epoch)

net.to(device)
train(1,net)
acc , epoch = test(1,net)
print('Saving..')

if not os.path.isdir('pruned_model'):
    os.mkdir('pruned_model')
torch.save(net, './pruned_model/try.pth')

model = torch.load('./pruned_model/try.pth')

test(1,model)