import torch
import torch.nn as nn

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

class shift_clip(nn.Module):
    def __init__(self, out_shift, out_bound, inplace=True):
        super(shift_clip, self).__init__()
        self.shift = out_shift
        self.clip = out_bound

    def forward(self, x):
        x = x >> self.shift
        out = torch.clamp(x, -(self.clip+1), self.clip)
        return out

class Layer(object):
    def __init__(self, dim_im_in, ch_im_in, ch_im_out, dim_im_out,
                    dim_ker_x, dim_ker_y, stride, pad, out_shift, in_prec, out_prec, weig_prec):
        self.din = dim_im_in
        self.cin = ch_im_in
        self.cout = ch_im_out
        self.dout = dim_im_out
        self.kx = dim_ker_x
        self.ky = dim_ker_y
        self.s = stride
        self.p = pad
        self.m = out_shift
        self.i = in_prec
        self.o = out_prec
        self.w = weig_prec

layer0 = Layer(28, 32, 192, 28, 1, 1, 1, 1, 0, 8, 8, 4)
layer1 = Layer(28, 192, 192, 28, 3, 3, 1, 1, 0, 8, 8, 4)
layer2 = Layer(28, 192, 192, 28, 1, 1, 1, 1, 0, 8, 8, 4)

x = torch.Tensor(1,layer0.cin,layer0.din,layer0.din).random_(0,(2**(layer0.i)-1))

net = nn.Sequential(nn.Conv2d(layer0.cin, layer0.cout, layer0.kx, layer0.s, layer0.p, groups=1, bias=False), #pointwise groups=1, depthwise groups=Cin
                    shift_clip(layer0.m, (2**(layer0.o-1)-1)),
                    nn.Conv2d(layer1.cin, layer1.cout, layer1.kx, layer1.s, layer1.p, groups=layer1.cin, bias=False),
                    shift_clip(layer1.m, (2**(layer1.o-1)-1)),
                    nn.Conv2d(layer2.cin, layer2.cout, layer2.kx, layer2.s, layer2.p, groups=1, bias=False),
                    shift_clip(layer2.m, (2**(layer2.o-1)-1)))

net[0].weight.data.random_(-(2**(layer0.w-1)),(2**(layer0.w-1)-1))
net[2].weight.data.random_(-(2**(layer1.w-1)),(2**(layer1.w-1)-1))
net[4].weight.data.random_(-(2**(layer2.w-1)),(2**(layer2.w-1)-1))

y = net(x)

str_out = str_tensor(x, 'IN_INT'+ str(layer0.i))
str_out += str_tensor(torch.Tensor(y), 'OUT_INT' + str(layer2.o))
str_out += str_weight(net[0].weight.data, 'WEIGHT0_INT' + str(layer0.w))
str_out += str_weight(net[2].weight.data, 'WEIGHT1_INT' + str(layer1.w))
str_out += str_weight(net[4].weight.data, 'WEIGHT2_INT' + str(layer2.w))

new_file = open("pulp_nn_network_test.txt", 'w')
new_file.write(str_out)
new_file.close()