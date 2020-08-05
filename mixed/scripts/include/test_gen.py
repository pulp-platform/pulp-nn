#
# test_gen.py
# Nazareno Bruschi <nazareno.bruschi@unibo.it>
#
# Copyright (C) 2019-2020 University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import errno
import os
import imp, sys
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
import random
from mako.template import Template
from models.linear_quantized_modules import ClippedLinearQuantization, LearnedClippedLinearQuantization, ScaledClippedLinearQuantization, ScaledThresholdsQuantization4d
from include.struct_test import PULPNNSrcDirs, PULPNNSrcDirs32bit, PULPNNSrcDirs64bit, PULPNNInstallPath32bit, PULPNNInstallPath64bit, PULPNNInstallPath
from include import comp_gen, utils

# Define string returning function for input-output-weights-thresholds
def str_tensor(x, tensor_name):
    input_image_txt = '#define '+tensor_name+' {'
    in_ch,H,W = x.size(1), x.size(2), x.size(3)
    for i in range(H):
        for j in range(W):
            for c in range(in_ch):
                input_image_txt += str(int(x[0][c][i][j].item())) + ', '
        input_image_txt+='\\\n'
    input_image_txt = input_image_txt[:-4]+'}\n'
    return input_image_txt

def str_weight(weight, tensor_name):
    out_ch, in_ch, k_w  = weight.size(0), weight.size(1), weight.size(2)
    str_v = '#define '+tensor_name+' {'
    for v in range(out_ch):
        for i in range(k_w):
            for j in range(k_w):
                for k in range(in_ch):
                    str_v += str(int(weight[v][k][i][j].item())) + ', '
        str_v += '\\\n'
    str_v = str_v[:-4]+'}\n'
    return str_v

def str_thr(thr, tensor_name):
    out_ch, thr_dim  = thr.size(0), thr.size(1)
    str_v = '#define '+tensor_name+' {'
    for v in range(out_ch):
        for i in range(thr_dim+1):
            if i == thr_dim:
                str_v += str(int(0)) + ', '
            else:
                str_v += str(int(thr[v][i].item())) + ', '

        str_v += '\\\n'
    str_v = str_v[:-4]+'}\n'
    return str_v

# To convert in HWC format the weights
def HWC_weights(w, nof, fs, nif):
    weights=np.zeros((nof, fs, fs, nif), 'int8')
    for i in range(nof):
        for j in range(fs):
            for k in range(fs):
                for t in range(nif):
                    weights[i,j,k,t]= (w[i,t,j,k])
    return weights

def str_weight_8(weight, tensor_name):
    out_ch, k_w, k_w2, in_ch  = weight.shape
    str_v = '#define '+tensor_name+' {'
    for v in range(out_ch):
        for i in range(k_w):
            for j in range(k_w):
                for k in range(in_ch):
                    str_v += str(int(weight[v][i][j][k].item())) + ', '
        str_v += '\\\n'
    str_v = str_v[:-4]+'}\n'
    return str_v

def str_tensor_8(x, tensor_name):
    input_image_txt = '#define '+tensor_name+' {'
    H,W, in_ch = x.shape
    for i in range(H):
        for j in range(W):
            for c in range(in_ch):
                input_image_txt += str(int(x[i][j][c].item())) + ', '
        input_image_txt+='\\\n'
    input_image_txt = input_image_txt[:-4]+'}\n'
    return input_image_txt

def str_tensor_linear(x, tensor_name):
    input_image_txt = '#define '+tensor_name+' {'
    in_ch = x.size(1)
    for c in range(in_ch):
        input_image_txt += str(int(x[0][c].item())) + ', '
    input_image_txt+='\\\n'
    input_image_txt = input_image_txt[:-4]+'}\n'
    return input_image_txt

def str_weight_linear(weight, tensor_name):
    out_ch, in_ch  = weight.size(0), weight.size(1)
    str_v = '#define '+tensor_name+' {'
    for v in range(out_ch):
        for k in range(in_ch):
            str_v += str(int(weight[v][k].item())) + ', '
        str_v += '\\\n'
    str_v = str_v[:-4]+'}'
    return str_v

def clip8(conv, bits):
    conv[conv >= +(2**(bits) -1)] = +(2**(bits) -1)
    conv[conv <= 0] = 0
    out = np.uint8(conv)
    return out

class BatchNorm(nn.Module):
    def __init__(self, Cin = 8, Kh = 3, Kw = 3, BitA = 8, BitW = 8, BitO=8, inplace=True):
        super(BatchNorm, self).__init__()
        self.BitO = BitO
        self.k = torch.Tensor(1).random_(0,(2**(8) - 1))
        self.l = torch.Tensor(1).random_(int(-(2**(BitA + BitW + np.log2(Cin * Kh * Kw) + 8 - 2 - 1))),int((2**(BitA + BitW + np.log2(Cin * Kh * Kw) + 8 - 2 - 1) - 1)))
        self.d = torch.Tensor(1).fill_(int(BitA + BitW + np.log2(Cin * Kh * Kw) + 5 - BitO))

    def forward(self, input):
        output = input*self.k+self.l
        x = output >> self.d
        out = clip8(x, self.BitO)
        return out

class BatchNorm_DORY(nn.Module):

    def __init__(self, Cin=8, Kh=3, Kw=3, BitA=8, BitW=8, BitO=8, groups=1, inplace=True):
        super(BatchNorm_DORY, self).__init__()
        self.BitO = BitO
        self.k = torch.Tensor(1, Cin, 1, 1).uniform_(0, (2**(8)))
        self.k = torch.round(self.k)
        th = int(
            (2**(BitA + BitW + np.log2(int(Cin / groups) * Kh * Kw) + 8 - 2 - 1)))
        if th > 2**30:
            th = 2**30
        self.l = torch.Tensor(1, Cin, 1, 1).random_(-th, th)
        self.d = torch.Tensor(1).fill_(
            int(BitA + BitW + np.log2(int(Cin / groups) * Kh * Kw) + 3 - BitO))

    def forward(self, input):
        output = input * self.k + self.l
        x = output >> self.d
        out = clip8(x, self.BitO)
        return out

class BatchNorm_DORY_linear(nn.Module):

    def __init__(self, Cin=8, BitA=8, BitW=8, BitO=8, groups=1, inplace=True):
        super(BatchNorm_DORY_linear, self).__init__()
        self.BitO = BitO
        self.k = torch.Tensor(1, Cin).uniform_(0, (2**(8)))
        self.k = torch.round(self.k)
        th = int(
            (2**(BitA + BitW + np.log2(int(Cin / groups)) + 8 - 2 - 1)))
        if th > 2**30:
            th = 2**30
        self.l = torch.Tensor(1, Cin).random_(-th, th)
        self.d = torch.Tensor(1).fill_(
            int(BitA + BitW + np.log2(int(Cin / groups)) + 3 - BitO))

    def forward(self, input):
        output = input * self.k + self.l
        x = output >> self.d
        out = clip8(x, self.BitO)
        return out

# Define a convolution with subbyte quantization of output activations
def pointwise_mixed_tests_generator_thr(IN_CH, IN_DIM_X, IN_DIM_Y, OUT_CH, OUT_DIM_X, OUT_DIM_Y, DIM_KER_X, DIM_KER_Y, PAD, STRIDE, NUM_BITS_A, NUM_BITS_W, NUM_BITS_O):
    x = torch.Tensor(1,IN_CH,IN_DIM_Y,IN_DIM_X).random_(0,(2**(NUM_BITS_A) - 1))

    net = nn.Sequential(nn.Conv2d(IN_CH, OUT_CH, kernel_size=DIM_KER_X, stride=STRIDE, padding=PAD, bias=False), ScaledThresholdsQuantization4d(num_bits=NUM_BITS_O))

    net[0].weight.data.random_(-(2**(NUM_BITS_W-1)),(2**(NUM_BITS_W-1) -1))
    net[1].thresholds = torch.Tensor(OUT_CH,2**NUM_BITS_O-1)
    net[1].signs = torch.Tensor(OUT_CH).fill_(1)
    for r in range(net[1].thresholds.size(0)):
        base = torch.Tensor(1).random_(0,DIM_KER_X*DIM_KER_Y*IN_CH*(2**(NUM_BITS_A-1)-1 ))
        for s in range(net[1].thresholds.size(1) ):
            if net[1].signs[r]==1:
                net[1].thresholds[r][s] = int(torch.clamp(- base*(2**(NUM_BITS_O-1)) + base*s, -32768, 32767).item())
            else:
                net[1].thresholds[r][s] = int(torch.clamp(base*(2**(NUM_BITS_O-1)) - base*s, -32768, 32767).item())
    y = net(x)

    bias = np.zeros(OUT_CH)

    str_bias = '#define BIAS {'
    for c in range(OUT_CH):
        str_bias += str(int(bias[c].item())) + ', '
    str_bias = str_bias[:-2]+'}\n'

    str_out = str_bias
    str_out += str_thr(net[1].thresholds,'THR_INT' + str(NUM_BITS_O))
    str_out += str_tensor(x, 'IN_INT'+ str(NUM_BITS_A))
    str_out += str_tensor(y, 'OUT_INT' + str(NUM_BITS_O))
    str_out += str_weight(net[0].weight.data, 'WEIGHT_INT' + str(NUM_BITS_W))

    return str_out

def pointwise_mixed_tests_generator_bn(Cin, h, w, Cout, Kh, Kw, p, s, BitA, BitW, BitO):
    # activations
    x = torch.Tensor(1,Cin,h,w).random_(0,(2**(BitA) - 1))
    #network
    net = nn.Sequential(nn.Conv2d(Cin, Cout, kernel_size=Kh, stride=s, padding=p, bias=False), BatchNorm_DORY(Cin = Cout, Kh = Kh, Kw = Kw, BitA = BitA, BitW = BitW, BitO=BitO))
    #weights
    net[0].weight.data.random_(-(2**(BitW-1)),(2**(BitW-1) -1))

    y = net(x)

    bias = np.zeros(Cout)

    str_bias = '#define BIAS {'
    for c in range(Cout):
        str_bias += str(int(bias[c].item())) + ', '
    str_bias = str_bias[:-2]+'}\n'

    str_out = str_bias
    str_out += str_tensor(net[1].k, 'KAPPA')
    str_out += str_tensor(net[1].l, 'LAMBDA')
    str_out += '#define OUT_SHIFT '+ str(int(net[1].d.item()))+'\n'
    str_out += str_tensor(x, 'IN_INT'+ str(BitA))
    str_out += str_tensor(torch.Tensor(y), 'OUT_INT' + str(BitO))
    str_out += str_weight(net[0].weight.data, 'WEIGHT_INT' + str(BitW))

    return str_out

def depthwise_mixed_tests_generator_bn(Cin, h, w, Cout, Kh, Kw, p, s, BitA, BitW, BitO):
    # activations
    x = torch.Tensor(1,Cin,h,w).random_(0,(2**(BitA) - 1))
    #network
    net = nn.Sequential(nn.Conv2d(Cin, Cout, kernel_size=Kh, stride=s, padding=p, groups=Cin, bias=False), BatchNorm_DORY(Cin = Cout, Kh = Kh, Kw = Kw, BitA = BitA, BitW = BitW, BitO=BitO, groups=Cin))
    #weights
    net[0].weight.data.random_(-(2**(BitW-1)),(2**(BitW-1) -1))

    y = net(x)

    bias = np.zeros(Cout)

    str_bias = '#define BIAS {'
    for c in range(Cout):
        str_bias += str(int(bias[c].item())) + ', '
    str_bias = str_bias[:-2]+'}\n'

    str_out = str_bias
    str_out += str_tensor(net[1].k, 'KAPPA')
    str_out += str_tensor(net[1].l, 'LAMBDA')
    str_out += '#define OUT_SHIFT '+ str(int(net[1].d.item()))+'\n'
    str_out += str_tensor(x, 'IN_INT'+ str(BitA))
    str_out += str_tensor(torch.Tensor(y), 'OUT_INT' + str(BitO))
    str_out += str_weight(net[0].weight.data, 'WEIGHT_INT' + str(BitW))

    return str_out

def linear_mixed_tests_generator(Cin, h, w, Cout, BitA, BitW, BitO):
    # activations
    x = torch.Tensor(1,Cin*h*w).random_(0,(2**(BitA) - 1))
    #network
    net = nn.Sequential(nn.Linear(Cin*h*w, Cout, bias=False))
    #weights
    net[0].weight.data.random_(-(2**(BitW-1)),(2**(BitW-1) -1))

    y = net(x)

    kap = np.zeros(Cout)
    lmb = np.zeros(Cout)

    str_kap = '#define KAPPA {'
    str_lmb = '#define LAMBDA {'
    for c in range(Cout):
        str_kap += str(int(kap[c].item())) + ', '
        str_lmb += str(int(lmb[c].item())) + ', '
    str_kap = str_kap[:-2]+'}\n'
    str_lmb = str_lmb[:-2]+'}\n'

    str_out = str_kap
    str_out += str_lmb
    str_out += str_tensor_linear(x, 'IN_INT'+ str(BitA))
    str_out += str_tensor_linear(y, 'OUT_INT' + str(BitO))
    str_out += str_weight_linear(net[0].weight.data, 'WEIGHT_INT' + str(BitW))

    return str_out

def linear_mixed_tests_generator_bn(Cin, h, w, Cout, Kh, Kw, p, s, BitA, BitW, BitO):
    # activations
    x = torch.Tensor(1,Cin*h*w).random_(0,(2**(BitA) - 1))
    #network
    net = nn.Sequential(nn.Linear(Cin*h*w, Cout, bias=False), BatchNorm_DORY_linear(Cin = Cout, BitA = BitA, BitW = BitW, BitO=BitO))
    #weights
    net[0].weight.data.random_(-(2**(BitW-1)),(2**(BitW-1) -1))

    y = net(x)

    bias = np.zeros(Cout)

    str_bias = '#define BIAS {'
    for c in range(Cout):
        str_bias += str(int(bias[c].item())) + ', '
    str_bias = str_bias[:-2]+'}\n'

    str_out = str_bias
    str_out += str_tensor_linear(net[1].k, 'KAPPA')
    str_out += str_tensor_linear(net[1].l, 'LAMBDA')
    str_out += '#define OUT_SHIFT '+ str(int(net[1].d.item()))+'\n'
    str_out += str_tensor_linear(x, 'IN_INT'+ str(BitA))
    str_out += str_tensor_linear(torch.Tensor(y), 'OUT_INT' + str(BitO))
    str_out += str_weight_linear(net[0].weight.data, 'WEIGHT_INT' + str(BitW))

    return str_out

# Define a convolution with byte quantization of output activations
def pointwise_mixed_tests_generator_int8(IN_CH, IN_DIM_X, IN_DIM_Y, OUT_CH, OUT_DIM_X, OUT_DIM_Y, DIM_KER_X, DIM_KER_Y, PAD, STRIDE, NUM_BITS_A, NUM_BITS_W, NUM_BITS_O, qf=15, rnd=True, sat=True):
    np.random.seed(1)
    W = np.random.randint(-(2**(NUM_BITS_W - 1)), (2**(NUM_BITS_W - 1) - 1), (OUT_CH, IN_CH, DIM_KER_X, DIM_KER_Y), 'int8')
    x = np.random.randint(0, (2**(NUM_BITS_A) - 1), (IN_DIM_Y, IN_DIM_X, IN_CH), 'uint8')
    y = np.zeros((OUT_DIM_Y, OUT_DIM_X, OUT_CH), 'uint8')

    if PAD>0:
        xp = np.zeros((IN_DIM_Y+2*PAD, IN_DIM_X+2*PAD, IN_CH), 'uint8')
        xp[ PAD:-PAD, PAD:-PAD, :] = x[:,:,:]
    else:
        xp = x

    for l in range(0, OUT_CH):
        for i in range(0, OUT_DIM_Y):
            for j in range(0, OUT_DIM_X):
                conv = np.zeros(1, 'int32')
                for m in range(0, IN_CH):
                    Wx = np.int32(W[l,m]) * np.int32(xp[i*STRIDE:i*STRIDE+DIM_KER_X,j*STRIDE:j*STRIDE+DIM_KER_Y,m])
                    conv += Wx.sum()
                if rnd:
                    conv += 1 << (qf-1)
                conv = conv >> qf
                if sat:
                    if conv >= +255:
                        conv = +255
                    if conv <= 0:
                        conv = 0
                y[i,j,l] = np.uint8(conv & 0xff)

    weights = HWC_weights(W, OUT_CH, DIM_KER_X, IN_CH)
    bias = np.zeros(OUT_CH)

    str_bias = '#define BIAS {'
    for c in range(OUT_CH):
        str_bias += str(int(bias[c].item())) + ', '
    str_bias = str_bias[:-2]+'}\n'

    str_out = str_bias
    str_out += str_tensor_8(y, 'OUT_INT' + str(NUM_BITS_O))
    str_out += str_weight_8(weights, 'WEIGHT_INT' + str(NUM_BITS_W))
    str_out += str_tensor_8(x, 'IN_INT' + str(NUM_BITS_A))

    return str_out

def headers(act_prec='32bit'):
    if act_prec == '32bit':
        shutil.copyfile(PULPNNSrcDirs['script'] + "templates/stats.h", PULPNNSrcDirs32bit['include'] + "stats.h")
        shutil.copyfile(PULPNNSrcDirs32bit['pulp_nn_inc'] + "pulp_nn_functions.h", PULPNNSrcDirs32bit['include'] + "pulp_nn_functions.h")
        shutil.copyfile(PULPNNSrcDirs32bit['pulp_nn_inc'] + "pulp_nn_utils.h", PULPNNSrcDirs32bit['include'] + "pulp_nn_utils.h")
        shutil.copyfile(PULPNNSrcDirs32bit['pulp_nn_support_function'] + "pulp_nn_utils.c", PULPNNSrcDirs32bit['support_function'] + "pulp_nn_utils.c")
    elif act_prec == '64bit':
        shutil.copyfile(PULPNNSrcDirs['script'] + "templates/stats.h", PULPNNSrcDirs64bit['include'] + "stats.h")
        shutil.copyfile(PULPNNSrcDirs64bit['pulp_nn_inc'] + "pulp_nn_functions.h", PULPNNSrcDirs64bit['include'] + "pulp_nn_functions.h")
        shutil.copyfile(PULPNNSrcDirs64bit['pulp_nn_inc'] + "pulp_nn_utils.h", PULPNNSrcDirs64bit['include'] + "pulp_nn_utils.h")
        shutil.copyfile(PULPNNSrcDirs64bit['pulp_nn_support_function'] + "pulp_nn_utils.c", PULPNNSrcDirs64bit['support_function'] + "pulp_nn_utils.c")

def copy_file(src_tag, key, dest_tag, act_prec='32bit'):
    if act_prec == '32bit':
        shutil.copyfile(PULPNNSrcDirs32bit[src_tag] + "%s" % key.filename, PULPNNSrcDirs32bit[dest_tag] + "%s" % key.filename)
    elif act_prec == '64bit':
        shutil.copyfile(PULPNNSrcDirs64bit[src_tag] + "%s" % key.filename, PULPNNSrcDirs64bit[dest_tag] + "%s" % key.filename)

def allocation(path_tag, type_of_kernel, layer, act_prec='32bit', in_precision=8, out_precision=8, wt_precision=8, quant=True, type_of_quant='shift_clip'):
    if quant:
        c = comp_gen.PULPNNDataAllocation(in_data_t=in_precision, out_data_t=out_precision, wt_data_t=wt_precision, quantization=type_of_quant, type=type_of_kernel, layer=layer)
    else:
        c = comp_gen.PULPNNDataAllocationNoQuant(in_data_t=in_precision, wt_data_t=wt_precision, type=type_of_kernel, layer=layer)
    if act_prec == '32bit':
        new_file = open(PULPNNSrcDirs32bit[path_tag] + c.filename, 'w')
    elif act_prec == '64bit':
        new_file = open(PULPNNSrcDirs64bit[path_tag] + c.filename, 'w')
    new_file.write(c.generate_code())
    new_file.close()

def golden(path_tag, layer, golden_gen, in_precision=8, out_precision=8, wt_precision=8, quant=True, act_prec='32bit'):
    torch.manual_seed(5)
    random.seed(5)
    if quant:
        c = comp_gen.PULPNNGoldenModel(in_data_t=in_precision, out_data_t=out_precision, wt_data_t=wt_precision)
        if act_prec == '32bit':
            new_file = open(PULPNNSrcDirs32bit[path_tag] + c.filename, 'w')
        elif act_prec == '64bit':
            new_file = open(PULPNNSrcDirs64bit[path_tag] + c.filename, 'w')
        new_file.write(c.generate_code() + "\n" + golden_gen(Cin=layer.ch_im_in, h=layer.dim_im_in_y, w=layer.dim_im_in_x, Cout=layer.ch_im_out, Kh=layer.dim_ker_x, Kw=layer.dim_ker_y, p=layer.pad_y_top, s=layer.stride_x, BitA=in_precision, BitW=wt_precision, BitO=out_precision) + "\n" + "\n" + "#endif")
        new_file.close()
    else:
        c = comp_gen.PULPNNGoldenModelNoQuant(in_data_t=in_precision, wt_data_t=wt_precision)
        if act_prec == '32bit':
            new_file = open(PULPNNSrcDirs32bit[path_tag] + c.filename, 'w')
        elif act_prec == '64bit':
            new_file = open(PULPNNSrcDirs64bit[path_tag] + c.filename, 'w')
        new_file.write(c.generate_code() + "\n" + golden_gen(layer.ch_im_in, layer.dim_im_in_y, layer.dim_im_in_x, layer.ch_im_out, in_precision, wt_precision, 32) + "\n" + "\n" + "#endif")
        new_file.close()

def generation(api, call, make, include, c):
    api += c.generate_api() + "\n"
    call += c.generate_call() + "\n"
    make += c.generate_make() + "\n"
    include += c.generate_include() + "\n"

    return api,call,make,include

def makefile(path_tag, make, act_prec='32bit'):
    if act_prec == '32bit':
        new_file = open(PULPNNSrcDirs32bit[path_tag] + "/Makefile", 'w')
    elif act_prec == '64bit':
        new_file = open(PULPNNSrcDirs64bit[path_tag] + "/Makefile", 'w')
    new_file.write(Template(filename="templates/make").render(PULPNNMAKE=make))
    new_file.close()

def main(path_tag, include, call, type, act_prec='32bit'):
    if act_prec == '32bit':
        new_file = open(PULPNNSrcDirs32bit[path_tag] + "/main.c", 'w')
    elif act_prec == '64bit':
        new_file = open(PULPNNSrcDirs64bit[path_tag] + "/main.c", 'w')
    new_file.write(Template(filename="templates/main.c").render(PULPNNINCLUDE=include, PULPNNCALL=call, TYPE_OF_KERNEL=type))
    new_file.close()