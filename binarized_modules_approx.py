import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import random
import numpy as np

def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    elif quant_mode=='circuit': #circuit view, LOW means -1, HIGH means 1
        tensor[tensor>0]=1
        tensor[tensor<=0]=0
        return tensor.byte()
    elif quant_mode=='logical':
        return tensor>0
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)




class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       import pdb; pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output

def Quantize(tensor,quant_mode='det',  params=None, numBits=8):
    tensor.clamp_(-2**(numBits-1),2**(numBits-1))
    if quant_mode=='det':
        tensor=tensor.mul(2**(numBits-1)).round().div(2**(numBits-1))
    else:
        tensor=tensor.mul(2**(numBits-1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2**(numBits-1))
        quant_fixed(tensor, params)
    return tensor

# import torch.nn._functions as tnnf


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            input.data=Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.scale = torch.nn.Parameter(torch.randn(self.weight.data.size(0)))
        self.scale.requires_grad = True
    
    def forward(self, input):
        input.data = Binarize(input.data)
        #print("in", input.shape)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.data)
        out = conv_function(input.data, self.weight.data) #remember to change the mode of weight and input to circuit
        # out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                    #    self.padding, self.dilation, self.groups)
        
        # for och in range(out.data.size(0)):
        #     out[och] -= self.weight.data.size(1) * self.weight.data.size(2) * (self.weight.data.size(3) / 3) * random.random() #one level approx, ich * kernel
        
        # out -= torch.rand(out.shape).cuda() * self.weight.data.size(1) * self.weight.data.size(2) * (self.weight.data.size(3) / 3)
        
        if not self.scale is None:
            self.scale.org=self.scale.data.clone()
            out *= self.bias.view(1, -1, 1, 1).expand_as(out)
            
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)


        return out
'''
#Now no approximation
def approx_conv_function(input, weight):
    # print(weight)
    weight = weight.reshape(weight.size(0), weight.size(1), 9) #och, ich, 9
    out = torch.zeros(input.size(0), weight.size(0), input.size(2), input.size(3)).float().cuda()
    for b in range(input.size(0)):
        print("Enter batch", b)
        for i_ch in range(input.size(1)):
			#print("Enter ich", i_ch)
            input_padding = torch.zeros(input.size(2)+2, input.size(3)+2).cuda()
            input_padding[1:input.size(2)+1, 1:input.size(3)+1] = input[b][i_ch]
            input_padding = input_padding.unfold(0,3,1).unfold(1,3,1) #h_o, w_o, 3, 3
            input_padding = input_padding.reshape(input_padding.size(0), input_padding.size(1), 9)
            for o_ch in range(weight.size(0)):
                for row_index in range(input.size(2)): #Note that the input size is the same as output size
                    for column_index in range(input.size(3)):
                        for u in range(3):     #row wise
                            tmp = 0
                            for v in range(3): #column wise
                                # out_np[b][_o_ch][row_index][column_index] += input_padding[row_index+u][column_index+v] * weight[_o_ch][0][u][v]
                                tmp += input_padding[row_index][column_index][3*u+v] * weight[o_ch][i_ch][3*u+v]
                            out[b][o_ch][row_index][column_index] = out[b][o_ch][row_index][column_index] + (3 * tmp.sign()).clip(-3,1)

    return out
'''
#No approx
def conv_function(input, weight):
    weight = weight.reshape(weight.size(0),1,1, weight.size(1) * 3, 3)
    #out = torch.zeros(input.size(0), weight.size(0), input.size(2), input.size(3)).float().cuda()
    input_padding = torch.zeros(input.size(0), input.size(1), input.size(2)+2, input.size(3)+2).cuda()
    input_padding[:,:,1:input.size(2)+1, 1:input.size(3)+1] = input #padding
    tmp = torch.nn.functional.unfold(input_padding, (3,3)).transpose(-1,-2)
    input_padding = tmp.reshape(tmp.size(0), tmp.size(1), -1, 3)
    #print(weight.shape)
    #print(input_padding.shape)
    out = (weight.mul(input_padding).sum(-1).sign()*5).clip(-3,1).sum(-1)
    return torch.nn.functional.fold(out, (input.size(2), input.size(3)), (1,1)).transpose(0,1)
'''
#No approx
def conv_function(input, weight):
    weight = weight.reshape(weight.size(0), weight.size(1) * 9)
    #out = torch.zeros(input.size(0), weight.size(0), input.size(2), input.size(3)).float().cuda()
    input_padding = torch.zeros(input.size(0), input.size(1), input.size(2)+2, input.size(3)+2).cuda()
    input_padding[:,:,1:input.size(2)+1, 1:input.size(3)+1] = input #padding
    input_padding = torch.nn.functional.unfold(input_padding, (3,3))
    out = weight.matmul(input_padding)
    return torch.nn.functional.fold(out, (input.size(2), input.size(3)), (1,1))
''' 
"""
input: torch.Size([100, 1, 28, 28])
weight: torch.Size([64, 1, 3, 3])
padding: (1, 1)
out: torch.Size([100, 64, 28, 28])
bias: torch.Size([64])

maxpool by 2

input: torch.Size([100, 64, 14, 14])
weight: torch.Size([64, 64, 3, 3])
padding: (1, 1)
out: torch.Size([100, 64, 14, 14])
bias: torch.Size([64])

maxpool by 2
"""
