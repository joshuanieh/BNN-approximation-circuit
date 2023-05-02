import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import random
import numpy as np
from numba import njit, prange


def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    elif quant_mode=='circuit': #circuit view, LOW means -1, HIGH means 1
        tensor[tensor>0]=1
        tensor[tensor<=0]=0
        return tensor
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

    def __init__(self, *kargs, correction_factor=0, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.display_size=False
        self.communicate_with_verilog = True
        self.load_weight_finish = False
        self.simulation_by_cpu = True
        self.approximation = True
        self.correction_factor = correction_factor
    
    
    def forward(self, input):
        # if self.simulation_by_cpu:
        #     input.data = Binarize(input.data, quant_mode='cir')
        # else:
        #     input.data = Binarize(input.data)
        input.data = Binarize(input.data, quant_mode='logical') #remember ti change the mode of weight as well
        # if not hasattr(self.weight,'org'):
        #     self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.data, quant_mode='logical')


##############################################################################

        # if self.communicate_with_verilog:
        #     '''
        #     Pass weights and activations to verilog
        #     '''
        #     #write weights and activatios into files and communicates with verilog
        #     if input.size(1) == 1:
        #         """
        #         input: torch.Size([100, 1, 28, 28])
        #         weight: torch.Size([64, 1, 3, 3])
        #         padding: (1, 1)
        #         out: torch.Size([100, 64, 28, 28])
        #         bias: torch.Size([64])
        #         """
        #         out = torch.zeros([100, 64, 28, 28])
        #         with open("operation.txt", 'w') as f: #operation to switch between first convolution and second
        #             f.write("0")
        #         if self.load_weight_finish == False:
        #             with open("weight.txt", 'w') as f:
        #                 for i in self.weight.data:
        #                     weight_string = ""
        #                     for j in i[0].flatten().tolist():
        #                         if j == -1:
        #                             weight_string += '0'
        #                         elif j == 1:
        #                             weight_string += '1'

        #                     f.write(weight_string)
        #             self.load_weight_finish = True

        #         with open("output_valid.txt", 'w') as f:
        #             f.write("0") #Reset to zero
        #         with open("input_valid.txt", 'w') as f:
        #             f.write("0") #Reset to zero
        #         for b, i in enumerate(input): #data in a batch
        #             input_size = 28
        #             input_padding = torch.zeros([input_size+2,input_size+2])
        #             input_padding[1:input_size+1, 1:input_size+1] = i[0]
        #             for row_index in range(input_size): #Note that the input size is the same as output size
        #                 for column_index in range(input_size):
        #                     with open("input.txt", 'w') as f:
        #                         data_string = ""
        #                         for u in range(3):     #row wise
        #                             for v in range(3): #column wise
        #                                 if input_padding[row_index+u][column_index+v] == -1:
        #                                     data_string += '0'
        #                                 elif input_padding[row_index+u][column_index+v] == 1:
        #                                     data_string += '1'
        #                                 else: #zero padding
        #                                     data_string += str(random.randint(0,1))
        #                         f.write(data_string + '\n') #Pass data to verilog
        #                     with open("output_valid.txt", 'w') as f:
        #                         f.write("0") #Reset to zero
        #                     with open("input_valid.txt", 'w') as f:
        #                         f.write("1") #Tell verilog the data is valid
        #                     while True:
        #                         with open("output_valid.txt", 'r') as f:
        #                             if f.read(1) == '1': #If verilog has calculated out the output
        #                                 break
        #                     with open("output.txt", 'r') as f:
        #                         _o_ch = 0
        #                         for line in f.readlines(): #Read the output
        #                             out[b][_o_ch][row_index][column_index] = int(line)
        #                             _o_ch += 1
                                    
        #         print("Hi!!")
        #     elif input.size(1) == 64:
        #         """
        #         input: torch.Size([100, 64, 14, 14])
        #         weight: torch.Size([64, 64, 3, 3])
        #         padding: (1, 1)
        #         out: torch.Size([100, 64, 14, 14])
        #         bias: torch.Size([64])
        #         """
        #         out = torch.zeros([100, 64, 14, 14])
        #         input_size = 14
        #         with open("operation.txt", 'w') as f: #operation to switch between first convolution and second
        #             f.write("1")
        #         if self.load_weight_finish == False:
        #             with open("weight.txt", 'w') as f:
        #                 for i in self.weight.data:
        #                     weight_string = ""
        #                     for j in i.flatten().tolist():
        #                         if j == -1:
        #                             weight_string += '0'
        #                         elif j == 1:
        #                             weight_string += '1'

        #                     f.write(weight_string)
        #             self.load_weight_finish = True

        #         with open("output_valid.txt", 'w') as f:
        #             f.write("0") #Reset to zero
        #         with open("input_valid.txt", 'w') as f:
        #             f.write("0") #Reset to zero
        #         for b, i in enumerate(input): #data in a batch
        #                 for row_index in range(input_size): #Note that the input size is the same as output size #output pixel wise
        #                     for column_index in range(input_size):
        #                         data_string = ""
        #                         with open("input.txt", 'w') as f:
        #                             for _i_ch in range(64):
        #                                 input_padding = torch.zeros([input_size+2,input_size+2])
        #                                 input_padding[1:input_size+1, 1:input_size+1] = i[_i_ch]
        #                                 for u in range(3):     #row wise
        #                                     for v in range(3): #column wise
        #                                         if input_padding[row_index+u][column_index+v] == -1:
        #                                             data_string += '0'
        #                                         elif input_padding[row_index+u][column_index+v] == 1:
        #                                             data_string += '1'
        #                                         else: #zero padding
        #                                             data_string += str(random.randint(0,1))
        #                             f.write(data_string) #Pass data to verilog
        #                         with open("output_valid.txt", 'w') as f:
        #                             f.write("0") #Reset to zero
        #                         with open("input_valid.txt", 'w') as f:
        #                             f.write("1") #Tell verilog the data is valid
        #                         while True:
        #                             with open("output_valid.txt", 'r') as f:
        #                                 if f.read(1) == '1': #If verilog has calculated out the output
        #                                     break
        #                         with open("output.txt", 'r') as f:
        #                             _o_ch = 0
        #                             for line in f.readlines(): #Read the output
        #                                 out[b][_o_ch][row_index][column_index] = int(line)
        #                                 _o_ch += 1
        #         print("Josh.")
        # else:        
        #     out = nn.functional.conv2d(input, self.weight, None, self.stride,
        #                             self.padding, self.dilation, self.groups)




##############################################################################
#Simulation approximation aware training by hand -> Too long
        if self.simulation_by_cpu:
            input_np = input.data.numpy()
            weight_np = self.weight.data.numpy()
            if self.approximation:
                out_double = torch.from_numpy(approx_conv_function(input_np, weight_np, self.correction_factor))
            else:
                out_double = torch.from_numpy(conv_function(input_np, weight_np))
            out = out_double.float()
            # print(nn.functional.conv2d(input, self.weight, None, self.stride,
            #                            self.padding, self.dilation, self.groups))
            # print(out)
        else:
            out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                       self.padding, self.dilation, self.groups)
################################################################################
        
        if not self.bias is None:
            # self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        if self.display_size:
            print("input:", input.size())
            print("weight:", self.weight.size())
            print("padding:", self.padding) #(1,1)
            if not self.bias is None:
                if self.display_size:
                    print("out:",out.size())
                    print("bias:",self.bias.size())

            self.display_size=False

        return out

#The acc is 97.97, a little lower because I take input as logical and 0 is seen as false. It suits more in the circuit level
@njit(parallel=True)
def conv_function(input_np, weight_np):
    # print(weight_np)
    if input_np.shape[1] == 1:
        out_np = np.zeros((100, 64, 28, 28))
        for b in prange(100):
            i = input_np[b] #data in a batch
            input_size = 28
            input_padding = np.zeros((input_size+2,input_size+2))
            input_padding[1:input_size+1, 1:input_size+1] = i[0]
            for _o_ch in prange(64):
                for row_index in prange(input_size): #Note that the input size is the same as output size
                    for column_index in prange(input_size):
                        for u in prange(3):     #row wise
                            for v in prange(3): #column wise
                                # out_np[b][_o_ch][row_index][column_index] += input_padding[row_index+u][column_index+v] * weight_np[_o_ch][0][u][v]
                                out_np[b][_o_ch][row_index][column_index] += 0 if np.logical_xor(input_padding[row_index+u][column_index+v], weight_np[_o_ch][0][u][v]) else 1
        # print(out_np)
        out_np *= 2
        out_np -= 1 * 3 * 3
        # out_np = 2 * out_np - np.full(out_np.shape, 1 * 3 * 3)
        print("Hi!")
    elif input_np.shape[1] == 64:
        out_np = np.zeros((100, 64, 14, 14))
        for b in prange(100):
            i = input_np[b] #data in a batch
            input_size = 14
            for _i_ch in prange(64):
                input_padding = np.zeros((input_size+2,input_size+2))
                input_padding[1:input_size+1, 1:input_size+1] = i[_i_ch]
                for _o_ch in prange(64):
                    for row_index in prange(input_size): #Note that the input size is the same as output size
                        for column_index in prange(input_size):
                            for u in prange(3):     #row wise
                                for v in prange(3): #column wise
                                    out_np[b][_o_ch][row_index][column_index] += 0 if np.logical_xor(input_padding[row_index+u][column_index+v], weight_np[_o_ch][_i_ch][u][v]) else 1
        # print(out_np.shape)
        # out_np = 2 * out_np - np.full(out_np.shape, 64 * 3 * 3)
        # print(out_np)
        out_np *= 2
        out_np -= 64 * 3 * 3
        # out_np -= np.full(out_np.shape, 64 * 3 * 3)
        print("Josh!")
    return out_np

#The acc is 97.97, a little lower because I take input as logical and 0 is seen as false. It suits more in the circuit level
@njit(parallel=True)
def approx_conv_function(input_np, weight_np, correction_factor=0):
    # print(weight_np)
    if input_np.shape[1] == 1:
        out_np = np.zeros((100, 64, 28, 28))
        for b in prange(100):
            i = input_np[b] #data in a batch
            input_size = 28
            input_padding = np.zeros((input_size+2,input_size+2))
            input_padding[1:input_size+1, 1:input_size+1] = i[0]
            for _o_ch in prange(64):
                for row_index in prange(input_size): #Note that the input size is the same as output size
                    for column_index in prange(input_size):
                        for u in prange(3):     #row wise
                            for v in prange(3):
                                out_np[b][_o_ch][row_index][column_index] += 0 if np.logical_xor(input_padding[row_index+u][column_index+v], weight_np[_o_ch][0][u][v]) else 1
                            # o_0 = np.logical_not(np.logical_xor(input_padding[row_index+u][column_index+0], weight_np[_o_ch][0][u][0]))
                            # o_1 = np.logical_not(np.logical_xor(input_padding[row_index+u][column_index+1], weight_np[_o_ch][0][u][1]))
                            # o_2 = np.logical_not(np.logical_xor(input_padding[row_index+u][column_index+2], weight_np[_o_ch][0][u][2]))

                            # o_0_and_1 = np.logical_and(o_0, o_1)
                            # o_0_xor_1 = np.logical_xor(o_0, o_1)
                            # o_0_xor_1_and_2 = np.logical_and(o_0_xor_1, o_2)

                            # out_np[b][_o_ch][row_index][column_index] += 2 if np.logical_or(o_0_and_1, o_0_xor_1_and_2) else 0
        # print(out_np)
        out_np *= 2
        out_np -= 1 * 3 * 3
        # out_np = 2 * out_np - np.full(out_np.shape, 1 * 3 * 3)
        # print("Hi!")
    elif input_np.shape[1] == 64:
        out_np = np.zeros((100, 64, 14, 14))
        for b in prange(100):
            i = input_np[b] #data in a batch
            input_size = 14
            for _i_ch in prange(64):
                input_padding = np.zeros((input_size+2,input_size+2))
                input_padding[1:input_size+1, 1:input_size+1] = i[_i_ch]
                for _o_ch in prange(64):
                    for row_index in prange(input_size): #Note that the input size is the same as output size
                        for column_index in prange(input_size):
                            for u in prange(3):     #row wise
                                approx_in_row = False
                                if approx_in_row:
                                    o_0 = np.logical_not(np.logical_xor(input_padding[row_index+u][column_index+0], weight_np[_o_ch][0][u][0]))
                                    o_1 = np.logical_not(np.logical_xor(input_padding[row_index+u][column_index+1], weight_np[_o_ch][0][u][1]))
                                    o_2 = np.logical_not(np.logical_xor(input_padding[row_index+u][column_index+2], weight_np[_o_ch][0][u][2]))
                                else:
                                    o_0 = np.logical_not(np.logical_xor(input_padding[row_index+0][column_index+u], weight_np[_o_ch][0][0][u]))
                                    o_1 = np.logical_not(np.logical_xor(input_padding[row_index+1][column_index+u], weight_np[_o_ch][0][1][u]))
                                    o_2 = np.logical_not(np.logical_xor(input_padding[row_index+2][column_index+u], weight_np[_o_ch][0][2][u]))

                                o_0_and_1 = np.logical_and(o_0, o_1)
                                o_0_xor_1 = np.logical_xor(o_0, o_1)
                                o_0_xor_1_and_2 = np.logical_and(o_0_xor_1, o_2)

                                out_np[b][_o_ch][row_index][column_index] += 2 if np.logical_or(o_0_and_1, o_0_xor_1_and_2) else 0

        # out_np = 2 * out_np - np.full(out_np.shape, 64 * 3 * 3)
        # print(out_np)
        out_np *= 2
        out_np -= 64 * 3 * 3

        # M = 400
        out_np += correction_factor
        # out_np -= np.full(out_np.shape, 64 * 3 * 3)
        # print("Josh!")
    return out_np

#Use numpy method to calculate conv, such as input*weight, but found slow, discard
# @njit(parallel=True)
# def conv_function(input_np, weight_np):
#     if input_np.shape[1] == 1:
#         out_np = np.zeros((100, 64, 28, 28))
#         for b in prange(100):
#             i = input_np[b] #data in a batch
#             input_size = 28
#             input_padding = np.zeros((input_size+2,input_size+2))
#             input_padding[1:input_size+1, 1:input_size+1] = i[0]
#             for _o_ch in prange(64):
#                 for row_index in prange(input_size): #Note that the input size is the same as output size
#                     for column_index in prange(input_size):
#                         # out_np[b][_o_ch][row_index][column_index] = (input_padding[row_index:row_index+3, column_index:column_index+3] * weight_np[_o_ch][0]).sum()
#                         logic = np.logical_not(np.logical_xor(input_padding[row_index:row_index+3, column_index:column_index+3], weight_np[_o_ch][0]))
#                         value = np.zeros(logic.shape)
#                         for u in prange(3):
#                             for v in prange(3):
#                                 value[u,v] = 1 if logic[u,v] else 0
#                         out_np[b][_o_ch][row_index][column_index] = value.sum()
#         print("Hi!")
#     elif input_np.shape[1] == 64:
#         out_np = np.zeros((100, 64, 14, 14))
#         for b in prange(100):
#             input_size = 14
#             input_padding = np.zeros((64, input_size+2,input_size+2))
#             input_padding[:, 1:input_size+1, 1:input_size+1] = input_np[b]
#             for _o_ch in prange(64):
#                 for row_index in prange(input_size): #Note that the input size is the same as output size
#                     for column_index in prange(input_size):
#                         out_np[b][_o_ch][row_index][column_index] = (input_padding[:, row_index:row_index+3, column_index:column_index+3] * weight_np[_o_ch]).sum()
#         print("Josh!")
#     return out_np


# @njit(parallel=True)
# def approx_conv_function(input_np, weight_np):
#     if input_np.shape[1] == 1:
#         out_np = np.zeros((100, 64, 28, 28))
#         for b in prange(100):
#             i = input_np[b] #data in a batch
#             input_size = 28
#             input_padding = np.zeros((input_size+2,input_size+2))
#             input_padding[1:input_size+1, 1:input_size+1] = i[0]
#             for _o_ch in prange(64):
#                 for row_index in prange(input_size): #Note that the input size is the same as output size
#                     for column_index in prange(input_size):
#                         for u in prange(3):     #row wise
#                             tmp = 0
#                             for v in prange(3): #column wise
#                                 tmp += input_padding[row_index+u][column_index+v] * weight_np[_o_ch][0][u][v]
                            
#                             out_np[b][_o_ch][row_index][column_index] += (tmp // 2) * 2 #single layer approximation
#         out_np = 2*out_np-1*3*3
#         print("Approx: Hi!", out_np.shape)
#     elif input_np.shape[1] == 64:
#         out_np = np.zeros((100, 64, 14, 14))
#         for b in prange(100):
#             i = input_np[b] #data in a batch
#             input_size = 14
#             for _i_ch in prange(64):
#                 input_padding = np.zeros((input_size+2,input_size+2))
#                 input_padding[1:input_size+1, 1:input_size+1] = i[_i_ch]
#                 for _o_ch in prange(64):
#                     for row_index in prange(input_size): #Note that the input size is the same as output size
#                         for column_index in prange(input_size):
#                             for u in prange(3):     #row wise
#                                 tmp = 0
#                                 for v in prange(3): #column wise
#                                     tmp += input_padding[row_index+u][column_index+v] * weight_np[_o_ch][_i_ch][u][v]

#                                 out_np[b][_o_ch][row_index][column_index] += (tmp // 2) * 2
#         out_np = 2*out_np-64*3*3
#         print("Approx: Josh!", out_np.shape)
#     return out_np

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

# ##############################################################################
# #Simulation approximation aware training by hand -> Too long
#         if self.simulation_by_cpu:
#             input_np = input.data.numpy()
#             weight_np = self.weight.data.numpy()
            
#             if input_np.shape[1] == 1:
#                 out_np = np.zeros([100, 64, 28, 28])
#                 out = torch.from_numpy(conv_function_1(input_np, weight_np, out_np))
#             elif input_np.shape[1] == 64:
#                 out_np = np.zeros([100, 64, 14, 14])
#                 out = torch.from_numpy(conv_function_2(input_np, weight_np, out_np))
#         else:
#             out = nn.functional.conv2d(input, self.weight, None, self.stride,
#                                        self.padding, self.dilation, self.groups)
# ################################################################################
        
#         if not self.bias is None:
#             self.bias.org=self.bias.data.clone()
#             out += self.bias.view(1, -1, 1, 1).expand_as(out)

#         if self.display_size:
#             print("input:", input.size())
#             print("weight:", self.weight.size())
#             print("padding:", self.padding) #(1,1)
#             if not self.bias is None:
#                 if self.display_size:
#                     print("out:",out.size())
#                     print("bias:",self.bias.size())

#             self.display_size=False

#         return out
    
# @njit
# def conv_function_1(input_np, weight_np, out_np):
#     for b, i in enumerate(input_np): #data in a batch
#         input_size = 28
#         input_padding = np.zeros([input_size+2,input_size+2])
#         input_padding[1:input_size+1, 1:input_size+1] = input_np[0]
#         for _o_ch in range(64):
#             for row_index in range(input_size): #Note that the input size is the same as output size
#                 for column_index in range(input_size):
#                     for u in range(3):     #row wise
#                         for v in range(3): #column wise
#                             # print(out.shape)
#                             # print(b)
#                             # print(_o_ch)
#                             # print(row_index)
#                             # print(column_index)
#                             out_np[b][_o_ch][row_index][column_index] += input_padding[row_index+u][column_index+v] * weight_np[_o_ch][0][u][v]
#     print("Hi!")
#     return out_np


# @njit
# def conv_function_2(input_np, weight_np, out_np):
#     for b, i in enumerate(input_np): #data in a batch
#         input_size = 14
#         for _i_ch in range(64):
#             input_padding = np.zeros([input_size+2,input_size+2])
#             input_padding[1:input_size+1, 1:input_size+1] = input_np[_i_ch]
#             for _o_ch in range(64):
#                 for row_index in range(input_size): #Note that the input size is the same as output size
#                     for column_index in range(input_size):
#                         for u in range(3):     #row wise
#                             for v in range(3): #column wise
#                                 # print(out.shape)
#                                 # print(b)
#                                 # print(_o_ch)
#                                 # print(row_index)
#                                 # print(column_index)
#                                 out_np[b][_o_ch][row_index][column_index] += input_padding[row_index+u][column_index+v] * weight_np[_o_ch][0][u][v]
#     print("Josh!")
#     return out_np