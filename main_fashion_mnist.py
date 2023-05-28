
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from binarized_modules_approx import  BinarizeLinear,BinarizeConv2d
from binarized_modules_approx import  Binarize,HingeLoss
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpus', default=0,
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

test_model = True
# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.test_batch_size, shuffle=True, **kwargs)
train_set = datasets.FashionMNIST("./datasets", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
# test_set = datasets.FashionMNIST("../data", download=True, train=False, transform=
                                            #    transforms.Compose([transforms.ToTensor()]))  
train_loader = torch.utils.data.DataLoader(train_set, 
                                           batch_size=100)
# test_loader = torch.utils.data.DataLoader(test_set,
#                                           batch_size=100)
n_samples_seen = 0.
mean = 0
std = 0
for train_batch, train_target in train_loader:
    batch_size = train_batch.shape[0]
    train_batch = train_batch.view(batch_size, -1)
    this_mean = torch.mean(train_batch, dim=1)
    this_std = torch.sqrt(
        torch.mean((train_batch - this_mean[:, None]) ** 2, dim=1))
    mean += torch.sum(this_mean, dim=0)
    std += torch.sum(this_std, dim=0)
    n_samples_seen += batch_size

mean /= n_samples_seen
std /= n_samples_seen
#print(f'mean: {mean}, std: {std}')

train_data = datasets.FashionMNIST('./datasets', train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=mean.view(1),
                                                            std=std.view(1))]))

test_data = datasets.FashionMNIST('./datasets', train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean.view(1),
                                                           std=std.view(1))]))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=100,
                                          shuffle=False)
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.infl_ratio=3
#         self.fc1 = BinarizeLinear(784, 2048*self.infl_ratio)
#         self.htanh1 = nn.Hardtanh()
#         self.bn1 = nn.BatchNorm1d(2048*self.infl_ratio)
#         self.fc2 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
#         self.htanh2 = nn.Hardtanh()
#         self.bn2 = nn.BatchNorm1d(2048*self.infl_ratio)
#         self.fc3 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
#         self.htanh3 = nn.Hardtanh()
#         self.bn3 = nn.BatchNorm1d(2048*self.infl_ratio)
#         self.fc4 = nn.Linear(2048*self.infl_ratio, 10)
#         self.logsoftmax=nn.LogSoftmax()
#         self.drop=nn.Dropout(0.5)

#     def forward(self, x):
#         x = x.view(-1, 28*28)
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.htanh1(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.htanh2(x)
#         x = self.fc3(x)
#         x = self.drop(x)
#         x = self.bn3(x)
#         x = self.htanh3(x)
#         x = self.fc4(x)
#         return self.logsoftmax(x)

#Simplized VGG
# class Net(nn.Module):

#     def __init__(self, num_classes=10):
#         super(Net, self).__init__()
#         self.infl_ratio=3;
#         self.features = nn.Sequential(
#             BinarizeConv2d(1, 128*self.infl_ratio, kernel_size=3, stride=1, padding=1,
#                       bias=True),
#             nn.BatchNorm2d(128*self.infl_ratio),
#             nn.Hardtanh(inplace=True),

#             BinarizeConv2d(128*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, padding=1, bias=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.BatchNorm2d(128*self.infl_ratio),
#             nn.Hardtanh(inplace=True),


#             BinarizeConv2d(128*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True),
#             nn.BatchNorm2d(256*self.infl_ratio),
#             nn.Hardtanh(inplace=True),


#             BinarizeConv2d(256*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.BatchNorm2d(256*self.infl_ratio),
#             nn.Hardtanh(inplace=True),


#             BinarizeConv2d(256*self.infl_ratio, 512*self.infl_ratio, kernel_size=3, padding=1, bias=True),
#             nn.BatchNorm2d(512*self.infl_ratio),
#             nn.Hardtanh(inplace=True),


#             BinarizeConv2d(512*self.infl_ratio, 512, kernel_size=3, padding=1, bias=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.BatchNorm2d(512),
#             nn.Hardtanh(inplace=True)

#         )
#         self.classifier = nn.Sequential(
#             BinarizeLinear(512 * 3 * 3, 1024, bias=True),
#             nn.BatchNorm1d(1024),
#             nn.Hardtanh(inplace=True),
#             #nn.Dropout(0.5),
#             BinarizeLinear(1024, 1024, bias=True),
#             nn.BatchNorm1d(1024),
#             nn.Hardtanh(inplace=True),
#             #nn.Dropout(0.5),
#             BinarizeLinear(1024, num_classes, bias=True),
#             nn.BatchNorm1d(num_classes, affine=False),
#             nn.LogSoftmax()
#         )

#         self.regime = {
#             0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
#             40: {'lr': 1e-3},
#             80: {'lr': 5e-4},
#             100: {'lr': 1e-4},
#             120: {'lr': 5e-5},
#             140: {'lr': 1e-5}
#         }

#     def forward(self, x):
#         x = self.features(x)
#         print(x.shape)
#         x = x.view(-1, 512 * 3 * 3)
#         x = self.classifier(x)
#         return x

#True VGG
# class Net(nn.Module):

#     def __init__(self, num_classes=1000):
#         super(Net, self).__init__()
#         # self.infl_ratio=3;
#         self.features = nn.Sequential(
#             BinarizeConv2d(1, 64, kernel_size=3, stride=1, padding=1,
#                       bias=True),
#             nn.BatchNorm2d(64),
#             nn.Hardtanh(inplace=True),

#             BinarizeConv2d(64, 64, kernel_size=3, padding=1, bias=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.BatchNorm2d(64),
#             nn.Hardtanh(inplace=True),


#             BinarizeConv2d(64, 128, kernel_size=3, padding=1, bias=True),
#             nn.BatchNorm2d(128),
#             nn.Hardtanh(inplace=True),


#             BinarizeConv2d(128, 128, kernel_size=3, padding=1, bias=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.BatchNorm2d(128),
#             nn.Hardtanh(inplace=True),


#             BinarizeConv2d(128, 256, kernel_size=3, padding=1, bias=True),
#             nn.BatchNorm2d(256),
#             nn.Hardtanh(inplace=True),

#             BinarizeConv2d(256, 256, kernel_size=3, padding=1, bias=True),
#             nn.BatchNorm2d(256),
#             nn.Hardtanh(inplace=True),

#             BinarizeConv2d(256, 256, kernel_size=3, padding=1, bias=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.BatchNorm2d(256),
#             nn.Hardtanh(inplace=True),

#             BinarizeConv2d(256, 512, kernel_size=3, padding=1, bias=True),
#             nn.BatchNorm2d(512),
#             nn.Hardtanh(inplace=True),

#             BinarizeConv2d(512, 512, kernel_size=3, padding=1, bias=True),
#             nn.BatchNorm2d(512),
#             nn.Hardtanh(inplace=True),

#             BinarizeConv2d(512, 512, kernel_size=3, padding=1, bias=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.BatchNorm2d(512),
#             nn.Hardtanh(inplace=True),

#             # BinarizeConv2d(512, 512, kernel_size=3, padding=1, bias=True),
#             # nn.BatchNorm2d(512),
#             # nn.Hardtanh(inplace=True),

#             # BinarizeConv2d(512, 512, kernel_size=3, padding=1, bias=True),
#             # nn.BatchNorm2d(512),
#             # nn.Hardtanh(inplace=True),

#             # BinarizeConv2d(512, 512, kernel_size=3, padding=1, bias=True),
#             # nn.MaxPool2d(kernel_size=2, stride=2),
#             # nn.BatchNorm2d(512),
#             # nn.Hardtanh(inplace=True)

#         )
#         self.classifier = nn.Sequential(
#             BinarizeLinear(512 * 1 * 1, 1024, bias=True),
#             nn.BatchNorm1d(1024),
#             nn.Hardtanh(inplace=True),
#             #nn.Dropout(0.5),
#             BinarizeLinear(1024, 1024, bias=True),
#             nn.BatchNorm1d(1024),
#             nn.Hardtanh(inplace=True),
#             #nn.Dropout(0.5),
#             BinarizeLinear(1024, num_classes, bias=True),
#             nn.BatchNorm1d(num_classes, affine=False),
#             nn.LogSoftmax()
#         )

#         self.regime = {
#             0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
#             40: {'lr': 1e-3},
#             80: {'lr': 5e-4},
#             100: {'lr': 1e-4},
#             120: {'lr': 5e-5},
#             140: {'lr': 1e-5}
#         }

#     def forward(self, x):
#         x = self.features(x)
#         # print(x.shape)
#         x = x.view(-1, 512 * 1 * 1)
#         x = self.classifier(x)
#         return x

#Paper VGG
class Net(nn.Module):

    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        # self.infl_ratio=3;
        self.features = nn.Sequential(
            BinarizeConv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.Hardtanh(inplace=True)

        )
        self.classifier = nn.Sequential(
            BinarizeLinear(64 * 7 * 7, 2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            BinarizeLinear(2048, num_classes, bias=True),
            nn.BatchNorm1d(num_classes, affine=False),
            nn.LogSoftmax(dim=1)
        )

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
            40: {'lr': 1e-3},
            80: {'lr': 5e-4},
            100: {'lr': 1e-4},
            120: {'lr': 5e-5},
            140: {'lr': 1e-5}
        }

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(-1, 64 * 7 * 7)
        x = self.classifier(x)
        return x

model = Net()
if args.cuda:
    torch.cuda.set_device(0)
    model.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        if epoch%40==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
patience = 25
best_acc = 0
stale = 0
def test():
    global best_acc, break_signal, stale, patience
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            #test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    #test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset), test_acc))
    
    if test_acc > best_acc:
        #torch.save(model.state_dict(), "Fashion_MNIST_best_paper_retrain_from_no_approx.ckpt") # only save best to prevent output memory exceed error
        best_acc = test_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
			#print(f"No improvment {patience} consecutive epochs, early stopping")
            break_signal = True

break_signal = False

# if args.evaluate:
#         if not os.path.isfile(args.evaluate):
#             parser.error('invalid checkpoint: {}'.format(args.evaluate))
#         checkpoint = torch.load(args.evaluate)
#         model.load_state_dict(checkpoint['state_dict'])
#         logging.info("loaded checkpoint '%s' (epoch %s)",
#                      args.evaluate, checkpoint['epoch'])
#====================================

num_parameters = sum([l.nelement() for l in model.parameters()])
print("number of parameters:", num_parameters)

checkpoint = torch.load("Fashion_MNIST_best_paper_no_approx.ckpt")
model.load_state_dict(checkpoint)

import gc
gc.collect()
if test_model:
    test()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
else:
    args.epochs = 50
    test()
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()
        if epoch%20 == 0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.5
        if break_signal:
            break
        '''
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        '''
