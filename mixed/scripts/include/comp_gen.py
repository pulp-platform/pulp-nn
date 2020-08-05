#
# comp_gen.py
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

from mako.template import Template

class PULPNNLayer(object):
    def __init__(self, dim_im_in_x, dim_im_in_y, ch_im_in, ch_im_out, dim_im_out_x,
                    dim_im_out_y, dim_ker_x, dim_ker_y, stride_x, stride_y, pad_y_top,
                    pad_y_bot, pad_x_left, pad_x_right,
                    bias_shift, out_mult):
        self.dim_im_in_x = dim_im_in_x
        self.dim_im_in_y = dim_im_in_y
        self.ch_im_in = ch_im_in
        self.ch_im_out = ch_im_out
        self.dim_im_out_x = dim_im_out_x
        self.dim_im_out_y = dim_im_out_y
        self.dim_ker_x = dim_ker_x
        self.dim_ker_y = dim_ker_y
        self.stride_x = stride_x
        self.stride_y = stride_y
        self.pad_y_top = pad_y_top
        self.pad_y_bot = pad_y_bot
        self.pad_x_left = pad_x_left
        self.pad_x_right = pad_x_right
        self.bias_shift = bias_shift
        self.out_mult = out_mult

class PULPNNFactory(object):
    def __init__(self, in_data_t, out_data_t, wt_data_t, act_prec):
        self.in_data_t = in_data_t
        self.out_data_t = out_data_t
        self.wt_data_t = wt_data_t
        self.quantization = ''
        self.fn_name = ''
        self.filename = ''
        self.api = ''
        self.act_prec = act_prec
    def generate_api(self):
        if self.act_prec == '32bit':
            return Template(filename="templates/pulp_nn_api").render(config=self)
        elif self.act_prec == '64bit':
            return Template(filename="templates/pulp_nn_api_64bit").render(config=self)
    def generate_make(self):
        return Template(filename="templates/pulp_nn_make").render(config=self)
    def generate_call(self):
        return Template(filename="templates/pulp_nn_call").render(config=self)
    def generate_include(self):
        return Template(filename="templates/pulp_nn_include").render(config=self)

class PULPNNConvolve(PULPNNFactory):
    def __init__(self, in_data_t, out_data_t, wt_data_t, quantization, act_prec):
        super().__init__(in_data_t, out_data_t, wt_data_t, act_prec)
        self.fn_name = "pulp_nn_conv_u{0}_u{1}_i{2}{3}".format(str(in_data_t), str(out_data_t), str(wt_data_t),
                                str("_" + quantization if quantization != "shift_clip" else ""))
        self.filename = self.fn_name + ".c"
        self.im2col_fn = "pulp_nn_im2col_u{0}_to_u{1}".format(str(in_data_t), '8')
        self.mat_mul_fn = "pulp_nn_matmul_u{0}_i{1}{2}".format(str(out_data_t), str(wt_data_t),
                                str("_" + quantization if quantization != "shift_clip" else ""))
        self.unpack_fn = "pulp_nn_i{0}_to_i{1}".format(str(wt_data_t), '8')
        self.bn_fn = "pulp_nn_bn_quant_u{0}".format(str(out_data_t))
        self.relu_fn = "pulp_nn_quant_u{0}".format(str(out_data_t))
        self.quantization = quantization
        self.api = self.__class__.__name__
        self.act_prec = act_prec
    def generate_code(self):
        if self.act_prec == '32bit':
            return Template(filename="templates/pulp_nn_conv_x_y_z.c").render(config=self)
        elif self.act_prec == '64bit':
            return Template(filename="templates/pulp_nn_conv_64bit_x_y_z.c").render(config=self)

class PULPNNMatMul(PULPNNFactory):
    def __init__(self, out_data_t, wt_data_t, quantization, act_prec):
        super().__init__("", out_data_t, wt_data_t, act_prec)
        self.fn_name = "pulp_nn_matmul_u{0}_i{1}{2}".format(str(out_data_t), str(wt_data_t),
                        str("_" + quantization if quantization != "shift_clip" else ""))
        self.unpack_fn = "pulp_nn_i{0}_to_i{1}".format(str(wt_data_t), '8')
        self.bn_fn = "pulp_nn_bn_quant_u{0}".format(str(out_data_t))
        self.relu_fn = "pulp_nn_quant_u{0}".format(str(out_data_t))
        self.filename = self.fn_name + ".c"
        self.quantization = quantization
        self.api = self.__class__.__name__
        self.act_prec = act_prec
    def generate_code(self):
        if self.act_prec == '32bit':
            return Template(filename="templates/pulp_nn_matmul_x_y.c").render(config=self)
        elif self.act_prec == '64bit':
            return Template(filename="templates/pulp_nn_matmul_64bit_x_y.c").render(config=self)

class PULPNNDepthwise(PULPNNFactory):
    def __init__(self, in_data_t, out_data_t, wt_data_t, quantization, act_prec):
        super().__init__(in_data_t, out_data_t, wt_data_t, act_prec)
        self.fn_name = "pulp_nn_dw_u{0}_u{1}_i{2}{3}".format(str(in_data_t), str(out_data_t), str(wt_data_t),
                        str("_" + quantization if quantization != "shift_clip" else ""))
        self.bn_fn = "pulp_nn_bn_quant_u{0}".format(str(out_data_t))
        self.relu_fn = "pulp_nn_quant_u{0}".format(str(out_data_t))
        self.filename = self.fn_name + ".c"
        self.quantization = quantization
        self.api = self.__class__.__name__
        self.less_precision = min([in_data_t, wt_data_t, out_data_t])
        self.act_prec = act_prec
    def generate_code(self):
        if self.act_prec == '32bit':
            return Template(filename="templates/pulp_nn_dw_x_y_z.c").render(config=self)
        elif self.act_prec == '64bit':
            return Template(filename="templates/pulp_nn_dw_64bit_x_y_z.c").render(config=self)

class PULPNNLinearNoQuant(PULPNNFactory):
    def __init__(self, in_data_t, wt_data_t, act_prec):
        super().__init__(in_data_t, "", wt_data_t, act_prec)
        self.fn_name = "pulp_nn_linear_u{0}_i{1}_i{2}".format(str(in_data_t), '32', str(wt_data_t))
        self.filename = self.fn_name + ".c"
        self.unpack_wt_fn = "pulp_nn_i{0}_to_i{1}".format(str(wt_data_t), '8')
        self.unpack_in_fn = "pulp_nn_u{0}_to_u{1}".format(str(in_data_t), '8')
        self.api = self.__class__.__name__
        self.less_precision = min([in_data_t, wt_data_t])
        self.act_prec = act_prec
    def generate_code(self):
        if self.act_prec == '32bit':
            return Template(filename="templates/pulp_nn_linear_nq_x_y_z.c").render(config=self)
        elif self.act_prec == '64bit':
            return Template(filename="templates/pulp_nn_linear_nq_64bit_x_y_z.c").render(config=self)

class PULPNNLinearQuant(PULPNNFactory):
    def __init__(self, in_data_t, out_data_t, wt_data_t, act_prec):
        super().__init__(in_data_t, out_data_t, wt_data_t, act_prec)
        self.fn_name = "pulp_nn_linear_u{0}_u{1}_i{2}".format(str(in_data_t), str(out_data_t), str(wt_data_t))
        self.filename = self.fn_name + ".c"
        self.unpack_wt_fn = "pulp_nn_i{0}_to_i{1}".format(str(wt_data_t), '8')
        self.unpack_in_fn = "pulp_nn_u{0}_to_u{1}".format(str(in_data_t), '8')
        self.bn_fn = "pulp_nn_bn_quant_u{0}".format(str(out_data_t))
        self.relu_fn = "pulp_nn_quant_u{0}".format(str(out_data_t))
        self.api = self.__class__.__name__
        self.less_precision = min([in_data_t, wt_data_t])
        self.act_prec = act_prec
    def generate_code(self):
        if self.act_prec == '32bit':
            return Template(filename="templates/pulp_nn_linear_q_x_y_z.c").render(config=self)
        elif self.act_prec == '64bit':
            return Template(filename="templates/pulp_nn_linear_q_64bit_x_y_z.c").render(config=self)

class PULPNNDataAllocation(PULPNNFactory):
    def __init__(self, in_data_t, out_data_t, wt_data_t, quantization, type, layer):
        super().__init__(in_data_t, out_data_t, wt_data_t, "")
        self.fn_name = "data_allocation_{0}_{1}_{2}".format(str(in_data_t),
                                                                    str(out_data_t),
                                                                    str(wt_data_t))
        self.dim_im_in_x = layer.dim_im_in_x
        self.dim_im_in_y = layer.dim_im_in_y
        self.ch_im_in = layer.ch_im_in
        self.ch_im_out = layer.ch_im_out
        self.dim_im_out_x = layer.dim_im_out_x
        self.dim_im_out_y = layer.dim_im_out_y
        self.dim_ker_x = layer.dim_ker_x
        self.dim_ker_y = layer.dim_ker_y
        self.stride_x = layer.stride_x
        self.stride_y = layer.stride_y
        self.pad_y_top = layer.pad_y_top
        self.pad_y_bot = layer.pad_y_bot
        self.pad_x_left = layer.pad_x_left
        self.pad_x_right = layer.pad_x_right
        self.bias_shift = layer.bias_shift
        self.out_mult = layer.out_mult

        self.type = type

        self.quantization = quantization

        self.filename = self.fn_name + ".h"
        self.api = self.__class__.__name__

        self.less_precision = min([in_data_t, wt_data_t, out_data_t])
    def generate_code(self):
        return Template(filename="templates/data_allocation_x_y_z.h").render(config=self)

class PULPNNDataAllocationNoQuant(PULPNNFactory):
    def __init__(self, in_data_t, wt_data_t, type, layer):
        super().__init__(in_data_t, "", wt_data_t, "")
        self.fn_name = "data_allocation_{0}_{1}_{2}".format(str(in_data_t),
                                                                    '32',
                                                                    str(wt_data_t))
        self.dim_im_in_x = layer.dim_im_in_x
        self.dim_im_in_y = layer.dim_im_in_y
        self.ch_im_in = layer.ch_im_in
        self.ch_im_out = layer.ch_im_out
        self.dim_im_out_x = layer.dim_im_out_x
        self.dim_im_out_y = layer.dim_im_out_y
        self.dim_ker_x = layer.dim_ker_x
        self.dim_ker_y = layer.dim_ker_y
        self.stride_x = layer.stride_x
        self.stride_y = layer.stride_y
        self.pad_y_top = layer.pad_y_top
        self.pad_y_bot = layer.pad_y_bot
        self.pad_x_left = layer.pad_x_left
        self.pad_x_right = layer.pad_x_right
        self.bias_shift = layer.bias_shift
        self.out_mult = layer.out_mult

        self.type = type

        self.filename = self.fn_name + ".h"
        self.api = self.__class__.__name__

        self.less_precision = min([in_data_t, wt_data_t])
    def generate_code(self):
        return Template(filename="templates/data_allocation_x_y_z.h").render(config=self)

class PULPNNGoldenModel(PULPNNFactory):
    def __init__(self, in_data_t, out_data_t, wt_data_t):
        super().__init__(in_data_t, out_data_t, wt_data_t, "")
        self.fn_name = "golden_{0}_{1}_{2}".format(str(in_data_t),
                                                       str(out_data_t),
                                                       str(wt_data_t))
        self.filename = self.fn_name + ".h"
        self.api = self.__class__.__name__
    def generate_code(self):
        return Template(filename="templates/golden_x_y_z.h").render(config=self)

class PULPNNGoldenModelNoQuant(PULPNNFactory):
    def __init__(self, in_data_t, wt_data_t):
        super().__init__(in_data_t, "", wt_data_t, "")
        self.fn_name = "golden_{0}_{1}_{2}".format(str(in_data_t),
                                                       '32',
                                                       str(wt_data_t))
        self.filename = self.fn_name + ".h"
        self.api = self.__class__.__name__
    def generate_code(self):
        return Template(filename="templates/golden_x_y_z.h").render(config=self)
