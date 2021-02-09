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

from include import pulp_nn_factory, pulp_nn_init, pulp_nn_struct
import pulp_nn_test_setup


def main():

    if pulp_nn_test_setup.TYPE_OF_KERNEL == 'depthwise' and pulp_nn_test_setup.CH_IM_IN != pulp_nn_test_setup.CH_IM_OUT:
        raise Exception("ERROR! ch_in must be equal to ch_out in a depthwise convolution")
    elif (pulp_nn_test_setup.TYPE_OF_KERNEL == 'pointwise') and ((pulp_nn_test_setup.DIM_KERNEL_X != 1) or (pulp_nn_test_setup.DIM_KERNEL_Y != 1)):
        raise Exception("ERROR! kernel dimension must be equal to 1 in a pointwise convolution")

    layer_to_gen = pulp_nn_factory.PULPNNLayer(dim_in_x=pulp_nn_test_setup.DIM_IM_IN_X, dim_in_y=pulp_nn_test_setup.DIM_IM_IN_Y, ch_in=pulp_nn_test_setup.CH_IM_IN, ch_out=pulp_nn_test_setup.CH_IM_OUT, dim_out_x=pulp_nn_test_setup.DIM_IM_OUT_X,
                        dim_out_y=pulp_nn_test_setup.DIM_IM_OUT_Y, ker_x=pulp_nn_test_setup.DIM_KERNEL_X, ker_y=pulp_nn_test_setup.DIM_KERNEL_Y, stride_x=pulp_nn_test_setup.STRIDE_X, stride_y=pulp_nn_test_setup.STRIDE_Y, pad_y_top=pulp_nn_test_setup.PADDING_Y_TOP,
                        pad_y_bot=pulp_nn_test_setup.PADDING_Y_BOTTOM, pad_x_left=pulp_nn_test_setup.PADDING_X_LEFT, pad_x_right=pulp_nn_test_setup.PADDING_X_RIGHT,
                        pool_kernel=pulp_nn_test_setup.POOL_KERNEL, pool_stride=pulp_nn_test_setup.POOL_STRIDE, bias=pulp_nn_test_setup.BIAS, bn=pulp_nn_test_setup.BN, relu=pulp_nn_test_setup.RELU)

    for a in pulp_nn_init.BN_ACTIVATIONS:
      
        pulp_nn_struct.mkdir_test(kernel=pulp_nn_test_setup.TYPE_OF_KERNEL, act_prec=a, ext=pulp_nn_test_setup.ISA)
        pulp_nn_factory.headers(act_prec=a, ext=pulp_nn_test_setup.ISA)

        if pulp_nn_test_setup.SINGLE_KERNEL == 1:

            if pulp_nn_test_setup.TYPE_OF_KERNEL == 'matmul':
                kernel_to_test = pulp_nn_factory.PULPNNKernel(name='matmul', inp=None, out=pulp_nn_test_setup.out_precision, wt=pulp_nn_test_setup.wt_precision, quant=pulp_nn_test_setup.quantization_type, act_prec=a, ext=pulp_nn_test_setup.ISA)
                matmul=pulp_nn_factory.PULPNNMatMul(kernel=kernel_to_test, layer=layer_to_gen)
                pulp_nn_factory.copy_file(src_tag='matmul', key=matmul, dest_tag='pulp_nn_matmul')
                pulp_nn_factory.allocation(path_tag='data_allocation_matm', comp=matmul)
                pulp_nn_factory.golden(path_tag='golden_model_matm', comp=matmul)
                pulp_nn_init.PULPNNCALL,pulp_nn_init.PULPNNMAKE,pulp_nn_init.PULPNNINCLUDE=pulp_nn_factory.generation(
                        call=pulp_nn_init.PULPNNCALL,
                        make=pulp_nn_init.PULPNNMAKE, 
                        include=pulp_nn_init.PULPNNINCLUDE, 
                        comp=matmul)

            elif pulp_nn_test_setup.TYPE_OF_KERNEL == 'convolution':
                kernel_to_test = pulp_nn_factory.PULPNNKernel(name='convolution', inp=pulp_nn_test_setup.in_precision, out=pulp_nn_test_setup.out_precision, wt=pulp_nn_test_setup.wt_precision, quant=pulp_nn_test_setup.quantization_type, act_prec=a, ext=pulp_nn_test_setup.ISA)
                conv=pulp_nn_factory.PULPNNConvolve(kernel=kernel_to_test, layer=layer_to_gen)
                kernel_matmul = pulp_nn_factory.PULPNNKernel(name='matmul', inp=None, out=pulp_nn_test_setup.out_precision, wt=pulp_nn_test_setup.wt_precision, quant=pulp_nn_test_setup.quantization_type, act_prec=a, ext=pulp_nn_test_setup.ISA)
                matmul=pulp_nn_factory.PULPNNMatMul(kernel=kernel_matmul, layer=layer_to_gen)
                pulp_nn_factory.copy_file(src_tag='convolution', key=conv, dest_tag='pulp_nn_convolution')
                pulp_nn_factory.copy_file(src_tag='matmul', key=matmul, dest_tag='pulp_nn_matmul')
                pulp_nn_factory.allocation(path_tag='data_allocation_conv', comp=conv)
                pulp_nn_factory.golden(path_tag='golden_model_conv', comp=conv)
                pulp_nn_init.PULPNNCALL,pulp_nn_init.PULPNNMAKE,pulp_nn_init.PULPNNINCLUDE=pulp_nn_factory.generation(
                        call=pulp_nn_init.PULPNNCALL,
                        make=pulp_nn_init.PULPNNMAKE,
                        include=pulp_nn_init.PULPNNINCLUDE,
                        comp=conv)
                dummy0,pulp_nn_init.PULPNNMAKE,dummy1=pulp_nn_factory.generation(
                        call=None,
                        make=pulp_nn_init.PULPNNMAKE,
                        include=None,
                        comp=matmul)

            elif pulp_nn_test_setup.TYPE_OF_KERNEL == 'pointwise':
                kernel_to_test = pulp_nn_factory.PULPNNKernel(name='pointwise', inp=pulp_nn_test_setup.in_precision, out=pulp_nn_test_setup.out_precision, wt=pulp_nn_test_setup.wt_precision, quant=pulp_nn_test_setup.quantization_type, act_prec=a, ext=pulp_nn_test_setup.ISA)
                pw=pulp_nn_factory.PULPNNConvolvePointwise(kernel=kernel_to_test, layer=layer_to_gen)
                kernel_matmul = pulp_nn_factory.PULPNNKernel(name='matmul', inp=None, out=pulp_nn_test_setup.out_precision, wt=pulp_nn_test_setup.wt_precision, quant=pulp_nn_test_setup.quantization_type, act_prec=a, ext=pulp_nn_test_setup.ISA)
                matmul=pulp_nn_factory.PULPNNMatMul(kernel=kernel_matmul, layer=layer_to_gen)
                pulp_nn_factory.copy_file(src_tag='pointwise', key=pw, dest_tag='pulp_nn_pointwise')
                pulp_nn_factory.copy_file(src_tag='matmul', key=matmul, dest_tag='pulp_nn_matmul')
                pulp_nn_factory.allocation(path_tag='data_allocation_pw', comp=pw)
                pulp_nn_factory.golden(path_tag='golden_model_pw', comp=pw)
                pulp_nn_init.PULPNNCALL,pulp_nn_init.PULPNNMAKE,pulp_nn_init.PULPNNINCLUDE=pulp_nn_factory.generation(
                        call=pulp_nn_init.PULPNNCALL,
                        make=pulp_nn_init.PULPNNMAKE,
                        include=pulp_nn_init.PULPNNINCLUDE,
                        comp=pw)
                dummy0,pulp_nn_init.PULPNNMAKE,dummy1=pulp_nn_factory.generation(
                        call=None,
                        make=pulp_nn_init.PULPNNMAKE,
                        include=None,
                        comp=matmul)

            elif pulp_nn_test_setup.TYPE_OF_KERNEL == 'depthwise':
                kernel_to_test = pulp_nn_factory.PULPNNKernel(name='depthwise', inp=pulp_nn_test_setup.in_precision, out=pulp_nn_test_setup.out_precision, wt=pulp_nn_test_setup.wt_precision, quant=pulp_nn_test_setup.quantization_type, act_prec=a, ext=pulp_nn_test_setup.ISA)
                dw=pulp_nn_factory.PULPNNConvolveDepthwise(kernel=kernel_to_test, layer=layer_to_gen)
                pulp_nn_factory.copy_file(src_tag='depthwise', key=dw, dest_tag='pulp_nn_depthwise')
                pulp_nn_factory.allocation(path_tag='data_allocation_dw', comp=dw)
                pulp_nn_factory.golden(path_tag='golden_model_dw', comp=dw)
                pulp_nn_init.PULPNNCALL,pulp_nn_init.PULPNNMAKE,pulp_nn_init.PULPNNINCLUDE=pulp_nn_factory.generation(
                        call=pulp_nn_init.PULPNNCALL,
                        make=pulp_nn_init.PULPNNMAKE,
                        include=pulp_nn_init.PULPNNINCLUDE,
                        comp=dw)

            elif pulp_nn_test_setup.TYPE_OF_KERNEL == 'linear_no_quant':
                kernel_to_test = pulp_nn_factory.PULPNNKernel(name='linear_no_quant', inp=pulp_nn_test_setup.in_precision, out=32, wt=pulp_nn_test_setup.wt_precision, quant=None, act_prec=a, ext=pulp_nn_test_setup.ISA)
                lin_nq=pulp_nn_factory.PULPNNLinearNoQuant(kernel=kernel_to_test, layer=layer_to_gen)
                pulp_nn_factory.copy_file(src_tag='linear_nq', key=lin_nq, dest_tag='pulp_nn_linear_nq')
                pulp_nn_factory.allocation(path_tag='data_allocation_ln_nq', comp=lin_nq)
                pulp_nn_factory.golden(path_tag='golden_model_ln_nq', comp=lin_nq)
                pulp_nn_init.PULPNNCALL,pulp_nn_init.PULPNNMAKE,pulp_nn_init.PULPNNINCLUDE=pulp_nn_factory.generation(
                        call=pulp_nn_init.PULPNNCALL,
                        make=pulp_nn_init.PULPNNMAKE,
                        include=pulp_nn_init.PULPNNINCLUDE,
                        comp=lin_nq)

            elif pulp_nn_test_setup.TYPE_OF_KERNEL == 'linear_quant':
                kernel_to_test = pulp_nn_factory.PULPNNKernel(name='linear_quant', inp=pulp_nn_test_setup.in_precision, out=pulp_nn_test_setup.out_precision, wt=pulp_nn_test_setup.wt_precision, quant=pulp_nn_test_setup.quantization_type, act_prec=a, ext=pulp_nn_test_setup.ISA)
                lin_q=pulp_nn_factory.PULPNNLinearQuant(kernel=kernel_to_test, layer=layer_to_gen)
                pulp_nn_factory.copy_file(src_tag='linear_q', key=lin_q, dest_tag='pulp_nn_linear_q')
                pulp_nn_factory.allocation(path_tag='data_allocation_ln_q', comp=lin_q)
                pulp_nn_factory.golden(path_tag='golden_model_ln_q', comp=lin_q)
                pulp_nn_init.PULPNNCALL,pulp_nn_init.PULPNNMAKE,pulp_nn_init.PULPNNINCLUDE=pulp_nn_factory.generation(
                        call=pulp_nn_init.PULPNNCALL,
                        make=pulp_nn_init.PULPNNMAKE,
                        include=pulp_nn_init.PULPNNINCLUDE,
                        comp=lin_q)

            elif pulp_nn_test_setup.TYPE_OF_KERNEL == 'maxpool':
                kernel_to_test = pulp_nn_factory.PULPNNKernel(name='maxpool', inp=pulp_nn_test_setup.in_precision, out=None, wt=None, quant=None, act_prec=a, ext=pulp_nn_test_setup.ISA)
                maxp=pulp_nn_factory.PULPNNMaxPool(kernel=kernel_to_test, layer=layer_to_gen)
                pulp_nn_factory.copy_file(src_tag='maxpool', key=maxp, dest_tag='pulp_nn_maxpool')
                pulp_nn_factory.allocation(path_tag='data_allocation_maxp', comp=maxp)
                pulp_nn_factory.golden(path_tag='golden_model_maxp', comp=maxp)
                pulp_nn_init.PULPNNCALL,pulp_nn_init.PULPNNMAKE,pulp_nn_init.PULPNNINCLUDE=pulp_nn_factory.generation(
                        call=pulp_nn_init.PULPNNCALL,
                        make=pulp_nn_init.PULPNNMAKE,
                        include=pulp_nn_init.PULPNNINCLUDE,
                        comp=maxp)

            elif pulp_nn_test_setup.TYPE_OF_KERNEL == 'avgpool':
                kernel_to_test = pulp_nn_factory.PULPNNKernel(name='avgpool', inp=pulp_nn_test_setup.in_precision, out=None, wt=None, quant=None, act_prec=a, ext=pulp_nn_test_setup.ISA)
                avgp=pulp_nn_factory.PULPNNAvgPool(kernel=kernel_to_test, layer=layer_to_gen)
                pulp_nn_factory.copy_file(src_tag='avgpool', key=avgp, dest_tag='pulp_nn_avgpool')
                pulp_nn_factory.allocation(path_tag='data_allocation_avgp', comp=avgp)
                pulp_nn_factory.golden(path_tag='golden_model_avgp', comp=avgp)
                pulp_nn_init.PULPNNCALL,pulp_nn_init.PULPNNMAKE,pulp_nn_init.PULPNNINCLUDE=pulp_nn_factory.generation(
                        call=pulp_nn_init.PULPNNCALL,
                        make=pulp_nn_init.PULPNNMAKE,
                        include=pulp_nn_init.PULPNNINCLUDE,
                        comp=avgp)

            elif pulp_nn_test_setup.TYPE_OF_KERNEL == 'add':
                kernel_to_test = pulp_nn_factory.PULPNNKernel(name='add', inp=pulp_nn_test_setup.in_precision, out=pulp_nn_test_setup.out_precision, wt=None, quant=None, act_prec=a, ext=pulp_nn_test_setup.ISA)
                add=pulp_nn_factory.PULPNNAdd(kernel=kernel_to_test, layer=layer_to_gen)
                pulp_nn_factory.copy_file(src_tag='add', key=add, dest_tag='pulp_nn_add')
                pulp_nn_factory.allocation(path_tag='data_allocation_add', comp=add)
                pulp_nn_factory.golden(path_tag='golden_model_add', comp=add)
                pulp_nn_init.PULPNNCALL,pulp_nn_init.PULPNNMAKE,pulp_nn_init.PULPNNINCLUDE=pulp_nn_factory.generation(
                        call=pulp_nn_init.PULPNNCALL,
                        make=pulp_nn_init.PULPNNMAKE,
                        include=pulp_nn_init.PULPNNINCLUDE,
                        comp=add)


        else:

            if pulp_nn_test_setup.TYPE_OF_KERNEL == 'matmul':
                for j in pulp_nn_init.PULPNNDataPrecisions:
                    for z in pulp_nn_init.PULPNNWeightsPrecisions:
                        for q in pulp_nn_init.PULPNNQuantizationMethods:
                            kernel_to_test = pulp_nn_factory.PULPNNKernel(name='matmul', inp=None, out=j, wt=z, quant=q, act_prec=a, ext=pulp_nn_test_setup.ISA)
                            matmul=pulp_nn_factory.PULPNNMatMul(kernel=kernel_to_test, layer=layer_to_gen)
                            pulp_nn_factory.copy_file(src_tag='matmul', key=matmul, dest_tag='pulp_nn_matmul')
                            pulp_nn_factory.allocation(path_tag='data_allocation_matm', comp=matmul)
                            pulp_nn_factory.golden(path_tag='golden_model_matm', comp=matmul)
                            pulp_nn_init.PULPNNCALL,pulp_nn_init.PULPNNMAKE,pulp_nn_init.PULPNNINCLUDE=pulp_nn_factory.generation(
                                    call=pulp_nn_init.PULPNNCALL,
                                    make=pulp_nn_init.PULPNNMAKE, 
                                    include=pulp_nn_init.PULPNNINCLUDE, 
                                    comp=matmul)

            elif pulp_nn_test_setup.TYPE_OF_KERNEL == 'convolution':
                for i in pulp_nn_init.PULPNNDataPrecisions:
                    for j in pulp_nn_init.PULPNNDataPrecisions:
                        for z in pulp_nn_init.PULPNNWeightsPrecisions:
                            for q in pulp_nn_init.PULPNNQuantizationMethods:
                                kernel_to_test = pulp_nn_factory.PULPNNKernel(name='convolution', inp=i, out=j, wt=z, quant=q, act_prec=a, ext=pulp_nn_test_setup.ISA)
                                conv=pulp_nn_factory.PULPNNConvolve(kernel=kernel_to_test, layer=layer_to_gen)                         
                                pulp_nn_factory.copy_file(src_tag='convolution', key=conv, dest_tag='pulp_nn_convolution')
                                pulp_nn_factory.allocation(path_tag='data_allocation_conv', comp=conv)
                                pulp_nn_factory.golden(path_tag='golden_model_conv', comp=conv)
                                pulp_nn_init.PULPNNCALL,pulp_nn_init.PULPNNMAKE,pulp_nn_init.PULPNNINCLUDE=pulp_nn_factory.generation(
                                        call=pulp_nn_init.PULPNNCALL,
                                        make=pulp_nn_init.PULPNNMAKE,
                                        include=pulp_nn_init.PULPNNINCLUDE,
                                        comp=conv)
                for j in pulp_nn_init.PULPNNDataPrecisions:
                    for z in pulp_nn_init.PULPNNWeightsPrecisions:
                        for q in pulp_nn_init.PULPNNQuantizationMethods:
                            kernel_matmul = pulp_nn_factory.PULPNNKernel(name='matmul', inp=None, out=j, wt=z, quant=q, act_prec=a, ext=pulp_nn_test_setup.ISA)
                            matmul=pulp_nn_factory.PULPNNMatMul(kernel=kernel_matmul, layer=layer_to_gen)
                            pulp_nn_factory.copy_file(src_tag='matmul', key=matmul, dest_tag='pulp_nn_matmul')
                            dummy0,pulp_nn_init.PULPNNMAKE,dummy1=pulp_nn_factory.generation(
                                    call=None,
                                    make=pulp_nn_init.PULPNNMAKE, 
                                    include=None, 
                                    comp=matmul)

            elif pulp_nn_test_setup.TYPE_OF_KERNEL == 'pointwise':
                for i in pulp_nn_init.PULPNNDataPrecisions:
                    for j in pulp_nn_init.PULPNNDataPrecisions:
                        for z in pulp_nn_init.PULPNNWeightsPrecisions:
                            for q in pulp_nn_init.PULPNNQuantizationMethods:
                                kernel_to_test = pulp_nn_factory.PULPNNKernel(name='pointwise', inp=i, out=j, wt=z, quant=q, act_prec=a, ext=pulp_nn_test_setup.ISA)
                                pw=pulp_nn_factory.PULPNNConvolvePointwise(kernel=kernel_to_test, layer=layer_to_gen)
                                pulp_nn_factory.copy_file(src_tag='pointwise', key=pw, dest_tag='pulp_nn_pointwise')
                                pulp_nn_factory.allocation(path_tag='data_allocation_pw', comp=pw)
                                pulp_nn_factory.golden(path_tag='golden_model_pw', comp=pw)
                                pulp_nn_init.PULPNNCALL,pulp_nn_init.PULPNNMAKE,pulp_nn_init.PULPNNINCLUDE=pulp_nn_factory.generation(
                                        call=pulp_nn_init.PULPNNCALL,
                                        make=pulp_nn_init.PULPNNMAKE,
                                        include=pulp_nn_init.PULPNNINCLUDE,
                                        comp=pw)                            
                for j in pulp_nn_init.PULPNNDataPrecisions:
                    for z in pulp_nn_init.PULPNNWeightsPrecisions:
                        for q in pulp_nn_init.PULPNNQuantizationMethods:
                            kernel_matmul = pulp_nn_factory.PULPNNKernel(name='matmul', inp=None, out=j, wt=z, quant=q, act_prec=a, ext=pulp_nn_test_setup.ISA)
                            matmul=pulp_nn_factory.PULPNNMatMul(kernel=kernel_matmul, layer=layer_to_gen)
                            pulp_nn_factory.copy_file(src_tag='matmul', key=matmul, dest_tag='pulp_nn_matmul')
                            dummy0,pulp_nn_init.PULPNNMAKE,dummy1=pulp_nn_factory.generation(
                                    call=None,
                                    make=pulp_nn_init.PULPNNMAKE, 
                                    include=None, 
                                    comp=matmul)

            elif pulp_nn_test_setup.TYPE_OF_KERNEL == 'depthwise':
                for i in pulp_nn_init.PULPNNDataPrecisions:
                    for j in pulp_nn_init.PULPNNDataPrecisions:
                        for z in pulp_nn_init.PULPNNWeightsPrecisions:
                            for q in pulp_nn_init.PULPNNQuantizationMethods:
                                kernel_to_test = pulp_nn_factory.PULPNNKernel(name='depthwise', inp=i, out=j, wt=z, quant=q, act_prec=a, ext=pulp_nn_test_setup.ISA)
                                dw=pulp_nn_factory.PULPNNConvolveDepthwise(kernel=kernel_to_test, layer=layer_to_gen)
                                pulp_nn_factory.copy_file(src_tag='depthwise', key=dw, dest_tag='pulp_nn_depthwise')
                                pulp_nn_factory.allocation(path_tag='data_allocation_dw', comp=dw)
                                pulp_nn_factory.golden(path_tag='golden_model_dw', comp=dw)
                                pulp_nn_init.PULPNNCALL,pulp_nn_init.PULPNNMAKE,pulp_nn_init.PULPNNINCLUDE=pulp_nn_factory.generation(
                                        call=pulp_nn_init.PULPNNCALL,
                                        make=pulp_nn_init.PULPNNMAKE,
                                        include=pulp_nn_init.PULPNNINCLUDE,
                                        comp=dw)

            elif pulp_nn_test_setup.TYPE_OF_KERNEL == 'linear_no_quant':
                for i in pulp_nn_init.PULPNNDataPrecisions:
                    for z in pulp_nn_init.PULPNNWeightsPrecisions:
                        kernel_to_test = pulp_nn_factory.PULPNNKernel(name='linear_no_quant', inp=i, out=32, wt=z, quant=None, act_prec=a, ext=pulp_nn_test_setup.ISA)
                        lin_nq=pulp_nn_factory.PULPNNLinearNoQuant(kernel=kernel_to_test, layer=layer_to_gen)
                        pulp_nn_factory.copy_file(src_tag='linear_nq', key=lin_nq, dest_tag='pulp_nn_linear_nq')
                        pulp_nn_factory.allocation(path_tag='data_allocation_ln_nq', comp=lin_nq)
                        pulp_nn_factory.golden(path_tag='golden_model_ln_nq', comp=lin_nq)
                        pulp_nn_init.PULPNNCALL,pulp_nn_init.PULPNNMAKE,pulp_nn_init.PULPNNINCLUDE=pulp_nn_factory.generation(
                                call=pulp_nn_init.PULPNNCALL,
                                make=pulp_nn_init.PULPNNMAKE,
                                include=pulp_nn_init.PULPNNINCLUDE,
                                comp=lin_nq)

            elif pulp_nn_test_setup.TYPE_OF_KERNEL == 'linear_quant':
                for i in pulp_nn_init.PULPNNDataPrecisions:
                    for j in pulp_nn_init.PULPNNDataPrecisions:
                        for z in pulp_nn_init.PULPNNWeightsPrecisions:
                            for q in pulp_nn_init.PULPNNQuantizationMethods:
                                kernel_to_test = pulp_nn_factory.PULPNNKernel(name='linear_quant', inp=i, out=j, wt=z, quant=q, act_prec=a, ext=pulp_nn_test_setup.ISA)
                                lin_q=pulp_nn_factory.PULPNNLinearQuant(kernel=kernel_to_test, layer=layer_to_gen)
                                pulp_nn_factory.copy_file(src_tag='linear_q', key=lin_q, dest_tag='pulp_nn_linear_q')
                                pulp_nn_factory.allocation(path_tag='data_allocation_ln_q', comp=lin_q)
                                pulp_nn_factory.golden(path_tag='golden_model_ln_q', comp=lin_q)
                                pulp_nn_init.PULPNNCALL,pulp_nn_init.PULPNNMAKE,pulp_nn_init.PULPNNINCLUDE=pulp_nn_factory.generation(
                                        call=pulp_nn_init.PULPNNCALL,
                                        make=pulp_nn_init.PULPNNMAKE,
                                        include=pulp_nn_init.PULPNNINCLUDE,
                                        comp=lin_q)

            elif pulp_nn_test_setup.TYPE_OF_KERNEL == 'maxpool':
                for i in pulp_nn_init.PULPNNDataPrecisions:
                    kernel_to_test = pulp_nn_factory.PULPNNKernel(name='maxpool', inp=i, out=None, wt=None, quant=None, act_prec=a, ext=pulp_nn_test_setup.ISA)
                    maxp=pulp_nn_factory.PULPNNMaxPool(kernel=kernel_to_test, layer=layer_to_gen)
                    pulp_nn_factory.copy_file(src_tag='maxpool', key=maxp, dest_tag='pulp_nn_maxpool')
                    pulp_nn_factory.allocation(path_tag='data_allocation_maxp', comp=maxp)
                    pulp_nn_factory.golden(path_tag='golden_model_maxp', comp=maxp)
                    pulp_nn_init.PULPNNCALL,pulp_nn_init.PULPNNMAKE,pulp_nn_init.PULPNNINCLUDE=pulp_nn_factory.generation(
                            call=pulp_nn_init.PULPNNCALL,
                            make=pulp_nn_init.PULPNNMAKE,
                            include=pulp_nn_init.PULPNNINCLUDE,
                            comp=maxp)

            elif pulp_nn_test_setup.TYPE_OF_KERNEL == 'avgpool':
                for i in pulp_nn_init.PULPNNDataPrecisions:
                    kernel_to_test = pulp_nn_factory.PULPNNKernel(name='avgpool', inp=i, out=None, wt=None, quant=None, act_prec=a, ext=pulp_nn_test_setup.ISA)
                    avg=pulp_nn_factory.PULPNNAvgPool(kernel=kernel_to_test, layer=layer_to_gen)
                    pulp_nn_factory.copy_file(src_tag='avgpool', key=avg, dest_tag='pulp_nn_avgpool')
                    pulp_nn_factory.allocation(path_tag='data_allocation_avgp', comp=avg)
                    pulp_nn_factory.golden(path_tag='golden_model_avgp', comp=avg)
                    pulp_nn_init.PULPNNCALL,pulp_nn_init.PULPNNMAKE,pulp_nn_init.PULPNNINCLUDE=pulp_nn_factory.generation(
                            call=pulp_nn_init.PULPNNCALL,
                            make=pulp_nn_init.PULPNNMAKE,
                            include=pulp_nn_init.PULPNNINCLUDE,
                            comp=avg)

            elif pulp_nn_test_setup.TYPE_OF_KERNEL == 'add':
                for i in pulp_nn_init.PULPNNDataPrecisions:
                    for j in pulp_nn_init.PULPNNDataPrecisions:
                        if j <= i:
                            kernel_to_test = pulp_nn_factory.PULPNNKernel(name='add', inp=i, out=j, wt=None, quant=None, act_prec=a, ext=pulp_nn_test_setup.ISA)
                            add=pulp_nn_factory.PULPNNAdd(kernel=kernel_to_test, layer=layer_to_gen)
                            pulp_nn_factory.copy_file(src_tag='add', key=add, dest_tag='pulp_nn_add')
                            pulp_nn_factory.allocation(path_tag='data_allocation_add', comp=add)
                            pulp_nn_factory.golden(path_tag='golden_model_add', comp=add)
                            pulp_nn_init.PULPNNCALL,pulp_nn_init.PULPNNMAKE,pulp_nn_init.PULPNNINCLUDE=pulp_nn_factory.generation(
                                    call=pulp_nn_init.PULPNNCALL,
                                    make=pulp_nn_init.PULPNNMAKE,
                                    include=pulp_nn_init.PULPNNINCLUDE,
                                    comp=add)

        pulp_nn_factory.makefile('test', kernel=kernel_to_test, make=pulp_nn_init.PULPNNMAKE)
        pulp_nn_factory.test('test', kernel=kernel_to_test, layer=layer_to_gen, include=pulp_nn_init.PULPNNINCLUDE, call=pulp_nn_init.PULPNNCALL)

        pulp_nn_init.PULPNNMAKE = ""
        pulp_nn_init.PULPNNINCLUDE = ""
        pulp_nn_init.PULPNNCALL = ""
        pulp_nn_init.PULPNNDEFINE = ""


if __name__ == '__main__':
    
    main()
