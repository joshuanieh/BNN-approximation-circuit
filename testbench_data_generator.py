'''
Goal: Generate inputs and outputs for PE_array_64_by_64_simulation, which has
    parameter WIDTH = 11;
    parameter ROW_LENGTH = 64;
    parameter O_CH = 64;
    input  [9*ROW_LENGTH:1]      activation_columns_in;
    input  [9*ROW_LENGTH*O_CH:1] weights_in;
    output [WIDTH*O_CH:1]        psum_rows_out;//64*9=576, singed psum 11 bits
'''

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
from binarized_modules import  BinarizeLinear,BinarizeConv2d
import time

time_s = time.time()

torch.manual_seed(1)

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
print(f'mean: {mean}, std: {std}')

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

print("length of test loader:", len(test_loader))

#Paper VGG
class Net(nn.Module):
    def __init__(self, num_classes=10, correction_factor=0):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            BinarizeConv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(64, 64, kernel_size=3, padding=1, bias=True, correction_factor=correction_factor), #correction_factor meaningful in 0 to 64*3*2 for one level approx
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

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(-1, 64 * 7 * 7)
        x = self.classifier(x)
        return x
    
checkpoint = torch.load("Fashion_MNIST_best_paper.ckpt", map_location=torch.device('cpu'))

criterion = nn.CrossEntropyLoss()

find_correction_factor = False
run_my_idea = True

if run_my_idea:
    print("Running my idea starts.")
    #To check the correction factor exists, run on a range from 0~384

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = Variable(data), Variable(target)
                output = model(data)
                test_loss += criterion(output, target).item() # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        test_acc = 100. * correct / len(test_loader.dataset)
        print(f'acc = {test_acc}')

    model = Net() #Do not pass correction factor into it
    for param in model.features.parameters():
        param.requires_grad = False


    # num_parameters = sum([l.nelement() for l in model.parameters()])
    # print("number of parameters:", num_parameters)

    model.load_state_dict(checkpoint)
    test()
    
    print("Running my idea ends.")

elif find_correction_factor:
    print("Finding correction factor starts.")
    #To check the correction factor exists, run on a range from 0~384

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = Variable(data), Variable(target)
                output = model(data)
                test_loss += criterion(output, target).item() # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        test_acc = 100. * correct / len(test_loader.dataset)
        print(f'M = {M}, acc = {test_acc}')

    for M in range(0, 381, 10):
        model = Net(correction_factor=M)
        for param in model.features.parameters():
            param.requires_grad = False


        # num_parameters = sum([l.nelement() for l in model.parameters()])
        # print("number of parameters:", num_parameters)

        model.load_state_dict(checkpoint)
        test()
    
    print("Finding correction factor ends.")

    '''
    one level approximation results:
        M = 0, acc = 10.0
        M = 10, acc = 10.0
        M = 20, acc = 10.0
        M = 30, acc = 10.0
        M = 40, acc = 10.0
        M = 50, acc = 9.989999771118164
        M = 60, acc = 9.960000038146973
        M = 70, acc = 9.949999809265137
        M = 80, acc = 10.010000228881836
        M = 90, acc = 9.989999771118164
        M = 100, acc = 9.989999771118164
        M = 110, acc = 9.970000267028809
        M = 120, acc = 9.989999771118164
        M = 130, acc = 10.109999656677246
        M = 140, acc = 10.329999923706055
        M = 150, acc = 10.720000267028809
        M = 160, acc = 11.020000457763672
        M = 170, acc = 12.180000305175781
        M = 180, acc = 17.540000915527344
        M = 190, acc = 18.559999465942383
        M = 200, acc = 22.469999313354492
        M = 210, acc = 24.479999542236328
        M = 220, acc = 26.25
        M = 230, acc = 25.940000534057617
        M = 240, acc = 27.0
        M = 250, acc = 25.360000610351562
        M = 260, acc = 24.68000030517578
        M = 270, acc = 24.0
        M = 280, acc = 24.440000534057617
        M = 290, acc = 23.170000076293945
        M = 300, acc = 21.149999618530273
        M = 310, acc = 19.5
        M = 320, acc = 16.809999465942383
        M = 330, acc = 15.020000457763672
        M = 340, acc = 14.859999656677246
        M = 350, acc = 14.09000015258789
        M = 360, acc = 13.5
        M = 370, acc = 13.170000076293945
        M = 380, acc = 12.489999771118164
    '''

#Todo: Retraining by manipulating GPU (faster) (no nvcc, but can try with numba manipulating gpu)


time_e = time.time()
print("Run time", time_e-time_s)























# #According to the BNN approximation aware training paper, there is a Conv2d(64,64,3)
# def generate_PE_array_data():
#     #######################################################################
#     class Net(nn.Module):
#         def __init__(self, num_classes=10):
#             super(Net, self).__init__()
#             # self.infl_ratio=3;
#             self.features = nn.Sequential(
#                 BinarizeConv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
#                 nn.MaxPool2d(kernel_size=2, stride=2),
#                 nn.BatchNorm2d(64),
#                 nn.Hardtanh(inplace=True),
#                 #Generate PE array 64*64*3*3 weights
#                 BinarizeConv2d(64, 64, kernel_size=3, padding=1, bias=True),
#                 nn.MaxPool2d(kernel_size=2, stride=2),
#                 nn.BatchNorm2d(64),
#                 nn.Hardtanh(inplace=True)

#             )
#             self.classifier = nn.Sequential(
#                 BinarizeLinear(64 * 7 * 7, 2048, bias=True),
#                 nn.BatchNorm1d(2048),
#                 nn.Hardtanh(inplace=True),
#                 #nn.Dropout(0.5),
#                 BinarizeLinear(2048, num_classes, bias=True),
#                 nn.BatchNorm1d(num_classes, affine=False),
#                 nn.LogSoftmax(dim=1)
#             )

#             self.regime = {
#                 0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
#                 40: {'lr': 1e-3},
#                 80: {'lr': 5e-4},
#                 100: {'lr': 1e-4},
#                 120: {'lr': 5e-5},
#                 140: {'lr': 1e-5}
#             }

#         def forward(self, x):
#             x = self.features(x)
#             # print(x.shape)
#             x = x.view(-1, 64 * 7 * 7)
#             x = self.classifier(x)
#             return x

#     model = Net()
#     checkpoint = torch.load("Fashion_MNIST_best_paper.ckpt")
#     model.load_state_dict(checkpoint)
#     #######################################################################
    
#     # weight_list = []
#     k=10#3*row_length*k=i_ch, a PE has 3 channels, k runs
#     row_length=7
#     o_ch=9
#     i_ch=3*row_length*k
#     data_list = []
#     data = torch.randn(1,i_ch,3,3).sign()                  #Binarized input
#     conv2d = nn.Conv2d(i_ch,o_ch,3)
#     conv2d.weight.data = conv2d.weight.data.sign()      #Binarizing weight
#     conv2d.bias.data = torch.tensor([0 for z in range(o_ch)])
#     output = conv2d(data)
#     # print(data[0])
#     # print(conv2d.weight.data)

#     for m in range(k):
#         for i in conv2d.weight.data: #o_ch iterations
#             count = 0
#             for j in i[m*row_length*3:(m+1)*row_length*3]: #take 3*row_length channels, a PE has 3 channels
#                 if count == 0:
#                     data_list += [j.flatten().tolist()]#weight, 3 should be batched together
#                 else:
#                     data_list[-1] += j.flatten().tolist()
#                 count += 1
#                 count %= 3
        
#         count = 0
#         for i in data[0][m*row_length*3:(m+1)*row_length*3]: #i_ch iterations
#             if count == 0:
#                 data_list += [i.flatten().tolist()]
#             else:
#                 data_list[-1] += i.flatten().tolist()
#             count += 1
#             count %= 3

#         # with open('weight.dat', 'w') as f:
#         #     for i in weight_list:
#         #         weight_string = ""
#         #         for j in i:
#         #             if j == -1:
#         #                 weight_string += '0'
#         #             else:
#         #                 weight_string += '1'

#         #         f.write("".join(weight_string) + '\n')
        
#         with open('data.dat', 'w') as f:
#             for i in data_list:
#                 data_string = ""
#                 for j in i:
#                     if j == -1:
#                         data_string += '0'
#                     else:
#                         data_string += '1'

#                 f.write(data_string + '\n')

#     with open('golden.dat', 'w') as f:
#         for i in output.data.flatten().tolist():
#             i = int(i)
#             output_string = ""
#             # print(i, end=": ")
#             if i >= 0:
#                 output_string = '{:014b}'.format(i)
#             else:
#                 i = (abs(i)-1)
#                 complement_string = '{:014b}'.format(i)
#                 output_list = ['0' if i == '1' else '1' for i in complement_string]
#                 output_string = "".join(output_list)
#             # print(output_string)
#             f.write(output_string + '\n')
# generate_PE_array_data()