#
# pulp_nn_examples_generator.py
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

################################# Mixed Precision Convolution examples generator script ##########################

############################################################### Version 1.0 #########################################################

from include import test_gen, struct_test_gen, struct_test, utils, comp_gen
import setup

SINGLE_KERNEL = setup.SINGLE_KERNEL
TYPE_OF_KERNEL = setup.TYPE_OF_KERNEL
in_precision = setup.in_precision
wt_precision = setup.wt_precision
out_precision = setup.out_precision
quantization_type = setup.quantization_type
BN_ACTIVATIONS = setup.BN_ACTIVATIONS
DIM_IM_IN_X = setup.DIM_IM_IN_X
DIM_IM_IN_Y = setup.DIM_IM_IN_Y
CH_IM_IN = setup.CH_IM_IN
DIM_IM_OUT_X = setup.DIM_IM_OUT_X
DIM_IM_OUT_Y = setup.DIM_IM_OUT_Y
CH_IM_OUT = setup.CH_IM_OUT
DIM_KERNEL_X = setup.DIM_KERNEL_X
DIM_KERNEL_Y = setup.DIM_KERNEL_Y
PADDING_Y_TOP = setup.PADDING_Y_TOP
PADDING_Y_BOTTOM = setup.PADDING_Y_BOTTOM
PADDING_X_LEFT = setup.PADDING_X_LEFT
PADDING_X_RIGHT = setup.PADDING_X_RIGHT
STRIDE_X = setup.STRIDE_X
STRIDE_Y = setup.STRIDE_Y
BIAS_SHIFT = setup.BIAS_SHIFT
OUT_MULT = setup.OUT_MULT

print("Single kernel" if SINGLE_KERNEL == 1 else "All kernels")
print(TYPE_OF_KERNEL)

struct_test_gen.mkdir_str(TYPE_OF_KERNEL)
test_gen.headers()

layer_to_test = comp_gen.PULPNNLayer(dim_im_in_x=DIM_IM_IN_X, dim_im_in_y=DIM_IM_IN_Y, ch_im_in=CH_IM_IN, ch_im_out=CH_IM_OUT, dim_im_out_x=DIM_IM_OUT_X,
                    dim_im_out_y=DIM_IM_OUT_Y, dim_ker_x=DIM_KERNEL_X, dim_ker_y=DIM_KERNEL_Y, stride_x=STRIDE_X, stride_y=STRIDE_Y, pad_y_top=PADDING_Y_TOP,
                    pad_y_bot=PADDING_Y_BOTTOM, pad_x_left=PADDING_X_LEFT, pad_x_right=PADDING_X_RIGHT, bias_shift=BIAS_SHIFT, out_mult=OUT_MULT)

if SINGLE_KERNEL == 1:

    if TYPE_OF_KERNEL == 'pointwise':
        test_gen.copy_file(src_tag='pulp_nn_pointwise_convolution', key=comp_gen.PULPNNConvolve(in_data_t=in_precision, out_data_t=out_precision, wt_data_t=wt_precision, quantization=quantization_type, act_prec=BN_ACTIVATIONS), dest_tag='pointwise_convolution')
        test_gen.copy_file(src_tag='pulp_nn_matmul', key=comp_gen.PULPNNMatMul(out_data_t=out_precision, wt_data_t=wt_precision, quantization=quantization_type, act_prec=BN_ACTIVATIONS), dest_tag='matmul')
        test_gen.allocation(path_tag='data_allocation_pw', layer=layer_to_test, in_precision=in_precision, out_precision=out_precision, wt_precision=wt_precision, quant=True, type_of_kernel=TYPE_OF_KERNEL, type_of_quant=quantization_type)
        test_gen.golden(path_tag='golden_model_pw', layer=layer_to_test, in_precision=in_precision, out_precision=out_precision, wt_precision=wt_precision, quant=True, golden_gen=test_gen.pointwise_mixed_tests_generator_bn)
        utils.PULPNNAPI,utils.PULPNNCALL,utils.PULPNNMAKE,utils.PULPNNINCLUDE=test_gen.generation(api=utils.PULPNNAPI, call=utils.PULPNNCALL, make=utils.PULPNNMAKE, include=utils.PULPNNINCLUDE, c=comp_gen.PULPNNConvolve(in_data_t=in_precision, out_data_t=out_precision, wt_data_t=wt_precision, quantization=quantization_type, act_prec=BN_ACTIVATIONS))
        utils.PULPNNAPI,utils.PULPNNCALL,utils.PULPNNMAKE,utils.PULPNNINCLUDE=test_gen.generation(api=utils.PULPNNAPI, call=utils.PULPNNCALL, make=utils.PULPNNMAKE, include=utils.PULPNNINCLUDE, c=comp_gen.PULPNNMatMul(out_data_t=out_precision, wt_data_t=wt_precision, quantization=quantization_type, act_prec=BN_ACTIVATIONS))

    elif TYPE_OF_KERNEL == 'depthwise':
        test_gen.copy_file(src_tag='pulp_nn_depthwise_convolution', key=comp_gen.PULPNNDepthwise(in_data_t=in_precision, out_data_t=out_precision, wt_data_t=wt_precision, quantization=quantization_type, act_prec=BN_ACTIVATIONS), dest_tag='depthwise_convolution')
        test_gen.allocation(path_tag='data_allocation_dw', layer=layer_to_test, in_precision=in_precision, out_precision=out_precision, wt_precision=wt_precision, quant=True, type_of_kernel=TYPE_OF_KERNEL, type_of_quant=quantization_type)
        test_gen.golden(path_tag='golden_model_dw', layer=layer_to_test, in_precision=in_precision, out_precision=out_precision, wt_precision=wt_precision, quant=True, golden_gen=test_gen.depthwise_mixed_tests_generator_bn)
        utils.PULPNNAPI,utils.PULPNNCALL,utils.PULPNNMAKE,utils.PULPNNINCLUDE=test_gen.generation(api=utils.PULPNNAPI, call=utils.PULPNNCALL, make=utils.PULPNNMAKE, include=utils.PULPNNINCLUDE, c=comp_gen.PULPNNDepthwise(in_data_t=in_precision, out_data_t=out_precision, wt_data_t=wt_precision, quantization=quantization_type, act_prec=BN_ACTIVATIONS))

    elif TYPE_OF_KERNEL == 'linear_no_quant':
        test_gen.copy_file(src_tag='pulp_nn_linear_convolution_nq', key=comp_gen.PULPNNLinearNoQuant(in_data_t=in_precision, wt_data_t=wt_precision, act_prec=BN_ACTIVATIONS), dest_tag='linear_convolution_nq')
        test_gen.allocation(path_tag='data_allocation_ln_nq', layer=layer_to_test, in_precision=in_precision, out_precision=32, wt_precision=wt_precision, quant=False, type_of_kernel=TYPE_OF_KERNEL)
        test_gen.golden(path_tag='golden_model_ln_nq', layer=layer_to_test, in_precision=in_precision, out_precision=32, wt_precision=wt_precision, quant=False, golden_gen=test_gen.linear_mixed_tests_generator)
        utils.PULPNNAPI,utils.PULPNNCALL,utils.PULPNNMAKE,utils.PULPNNINCLUDE=test_gen.generation(api=utils.PULPNNAPI, call=utils.PULPNNCALL, make=utils.PULPNNMAKE, include=utils.PULPNNINCLUDE, c=comp_gen.PULPNNLinearNoQuant(in_data_t=in_precision, wt_data_t=wt_precision, act_prec=BN_ACTIVATIONS))

    elif TYPE_OF_KERNEL == 'linear_quant':
        test_gen.copy_file(src_tag='pulp_nn_linear_convolution_q', key=comp_gen.PULPNNLinearQuant(in_data_t=in_precision, out_data_t=out_precision, wt_data_t=wt_precision, act_prec=BN_ACTIVATIONS), dest_tag='linear_convolution_q')
        test_gen.allocation(path_tag='data_allocation_ln_q', layer=layer_to_test, in_precision=in_precision, out_precision=out_precision, wt_precision=wt_precision, quant=True, type_of_kernel=TYPE_OF_KERNEL, type_of_quant=quantization_type)
        test_gen.golden(path_tag='golden_model_ln_q', layer=layer_to_test, in_precision=in_precision, out_precision=out_precision, wt_precision=wt_precision, quant=True, golden_gen=test_gen.linear_mixed_tests_generator_bn)
        utils.PULPNNAPI,utils.PULPNNCALL,utils.PULPNNMAKE,utils.PULPNNINCLUDE=test_gen.generation(api=utils.PULPNNAPI, call=utils.PULPNNCALL, make=utils.PULPNNMAKE, include=utils.PULPNNINCLUDE, c=comp_gen.PULPNNLinearQuant(in_data_t=in_precision, out_data_t=out_precision, wt_data_t=wt_precision, act_prec=BN_ACTIVATIONS))


else:

    if TYPE_OF_KERNEL == 'pointwise':
        for i in utils.PULPNNDataPrecisions:
            for j in utils.PULPNNDataPrecisions:
                for z in utils.PULPNNWeightsPrecisions:
                    for q in utils.PULPNNQuantizationMethods:
                        test_gen.copy_file(src_tag='pulp_nn_pointwise_convolution', key=comp_gen.PULPNNConvolve(in_data_t=i, out_data_t=j, wt_data_t=z, quantization=q, act_prec=BN_ACTIVATIONS), dest_tag='pointwise_convolution')
                        test_gen.allocation(path_tag='data_allocation_pw', layer=layer_to_test, in_precision=i, out_precision=j, wt_precision=z, quant=True, type_of_kernel=TYPE_OF_KERNEL, type_of_quant=q)
                        test_gen.golden(path_tag='golden_model_pw', layer=layer_to_test, in_precision=i, out_precision=j, wt_precision=z, quant=True, golden_gen=test_gen.pointwise_mixed_tests_generator_bn)
                        utils.PULPNNAPI,utils.PULPNNCALL,utils.PULPNNMAKE,utils.PULPNNINCLUDE=test_gen.generation(api=utils.PULPNNAPI, call=utils.PULPNNCALL, make=utils.PULPNNMAKE, include=utils.PULPNNINCLUDE, c=comp_gen.PULPNNConvolve(in_data_t=i, out_data_t=j, wt_data_t=z, quantization=q, act_prec=BN_ACTIVATIONS))
        for j in utils.PULPNNDataPrecisions:
            for z in utils.PULPNNWeightsPrecisions:
                for q in utils.PULPNNQuantizationMethods:
                    test_gen.copy_file(src_tag='pulp_nn_matmul', key=comp_gen.PULPNNMatMul(out_data_t=j, wt_data_t=z, quantization=q, act_prec=BN_ACTIVATIONS), dest_tag='matmul')
                    utils.PULPNNAPI,utils.PULPNNCALL,utils.PULPNNMAKE,utils.PULPNNINCLUDE=test_gen.generation(api=utils.PULPNNAPI, call=utils.PULPNNCALL, make=utils.PULPNNMAKE, include=utils.PULPNNINCLUDE, c=comp_gen.PULPNNMatMul(out_data_t=j, wt_data_t=z, quantization=q, act_prec=BN_ACTIVATIONS))

    elif TYPE_OF_KERNEL == 'depthwise':
        for i in utils.PULPNNDataPrecisions:
            for j in utils.PULPNNDataPrecisions:
                for z in utils.PULPNNWeightsPrecisions:
                    for q in utils.PULPNNQuantizationMethods:
                        test_gen.copy_file(src_tag='pulp_nn_depthwise_convolution', key=comp_gen.PULPNNDepthwise(in_data_t=i, out_data_t=j, wt_data_t=z, quantization=q, act_prec=BN_ACTIVATIONS), dest_tag='depthwise_convolution')
                        test_gen.allocation(path_tag='data_allocation_dw', layer=layer_to_test, in_precision=i, out_precision=j, wt_precision=z, quant=True, type_of_kernel=TYPE_OF_KERNEL, type_of_quant=q)
                        test_gen.golden(path_tag='golden_model_dw', layer=layer_to_test, in_precision=i, out_precision=j, wt_precision=z, quant=True, golden_gen=test_gen.depthwise_mixed_tests_generator_bn)
                        utils.PULPNNAPI,utils.PULPNNCALL,utils.PULPNNMAKE,utils.PULPNNINCLUDE=test_gen.generation(api=utils.PULPNNAPI, call=utils.PULPNNCALL, make=utils.PULPNNMAKE, include=utils.PULPNNINCLUDE, c=comp_gen.PULPNNDepthwise(in_data_t=i, out_data_t=j, wt_data_t=z, quantization=q, act_prec=BN_ACTIVATIONS))

    elif TYPE_OF_KERNEL == 'linear_no_quant':
        for i in utils.PULPNNDataPrecisions:
            for z in utils.PULPNNWeightsPrecisions:
                test_gen.copy_file(src_tag='pulp_nn_linear_convolution_nq', key=comp_gen.PULPNNLinearNoQuant(in_data_t=i, wt_data_t=z, act_prec=BN_ACTIVATIONS), dest_tag='linear_convolution_nq')
                test_gen.allocation(path_tag='data_allocation_ln_nq', layer=layer_to_test, in_precision=i, out_precision=32, wt_precision=z, quant=False, type_of_kernel=TYPE_OF_KERNEL)
                test_gen.golden(path_tag='golden_model_ln_nq', layer=layer_to_test, in_precision=i, out_precision=32, wt_precision=z, quant=False, golden_gen=test_gen.linear_mixed_tests_generator)
                utils.PULPNNAPI,utils.PULPNNCALL,utils.PULPNNMAKE,utils.PULPNNINCLUDE=test_gen.generation(api=utils.PULPNNAPI, call=utils.PULPNNCALL, make=utils.PULPNNMAKE, include=utils.PULPNNINCLUDE, c=comp_gen.PULPNNLinearNoQuant(in_data_t=i, wt_data_t=z, act_prec=BN_ACTIVATIONS))

    elif TYPE_OF_KERNEL == 'linear_quant':
        for i in utils.PULPNNDataPrecisions:
            for j in utils.PULPNNDataPrecisions:
                for z in utils.PULPNNWeightsPrecisions:
                    for q in utils.PULPNNQuantizationMethods:
                        test_gen.copy_file(src_tag='pulp_nn_linear_convolution_q', key=comp_gen.PULPNNLinearQuant(in_data_t=i, out_data_t=j, wt_data_t=z, act_prec=BN_ACTIVATIONS), dest_tag='linear_convolution_q')
                        test_gen.allocation(path_tag='data_allocation_ln_q', layer=layer_to_test, in_precision=i, out_precision=j, wt_precision=z, quant=True, type_of_kernel=TYPE_OF_KERNEL, type_of_quant=q)
                        test_gen.golden(path_tag='golden_model_ln_q', layer=layer_to_test, in_precision=i, out_precision=j, wt_precision=z, quant=True, golden_gen=test_gen.linear_mixed_tests_generator_bn)
                        utils.PULPNNAPI,utils.PULPNNCALL,utils.PULPNNMAKE,utils.PULPNNINCLUDE=test_gen.generation(api=utils.PULPNNAPI, call=utils.PULPNNCALL, make=utils.PULPNNMAKE, include=utils.PULPNNINCLUDE, c=comp_gen.PULPNNLinearQuant(in_data_t=i, out_data_t=j, wt_data_t=z, act_prec=BN_ACTIVATIONS))

test_gen.makefile('test', utils.PULPNNMAKE)
test_gen.main('test', utils.PULPNNINCLUDE, utils.PULPNNCALL, TYPE_OF_KERNEL)
