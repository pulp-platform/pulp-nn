import torch
import torch.nn as nn
import numpy as np


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

def clip16(conv, bits):
    conv[conv >= +(2**(bits) -1)] = +(2**(bits) -1)
    conv[conv <= 0] = 0
    out = np.int16(conv)
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
        out = clip16(x, self.BitO)
        return out

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


new_file = open("layer_test.txt", 'w')
new_file.write(pointwise_mixed_tests_generator_bn(32, 16, 16, 64, 3, 3, 1, 1, 16, 16, 16))
new_file.close()