#
# pulp_nn_kernels_generator.py
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

################################# Mixed Precision Convolution kernels generator script ##########################

############################################################### Version 1.0 #########################################################

from include import pulp_nn_factory, pulp_nn_init, pulp_nn_struct

def main():

    for a in pulp_nn_init.BN_ACTIVATIONS:

        for e in pulp_nn_init.CORE_EXTENTIONS:

            print("ISA: " + e + " Model normalization: " + a)
            
            pulp_nn_struct.mkdir_src(act_prec=a, ext=e)

            pulp_nn_factory.utils(act_prec=a, ext=e)

            for i in pulp_nn_init.PULPNNDataPrecisions:
                for j in pulp_nn_init.PULPNNDataPrecisions:
                    for z in pulp_nn_init.PULPNNWeightsPrecisions:
                        for q in pulp_nn_init.PULPNNQuantizationMethods:
                            kernel_to_test = pulp_nn_factory.PULPNNKernel(name='convolution', inp=i, out=j, wt=z, quant=q, act_prec=a, ext=e)
                            conv=pulp_nn_factory.PULPNNConvolve(kernel=kernel_to_test, layer=None)
                            pulp_nn_init.PULPNNAPI = pulp_nn_factory.kernel(path_tag='convolution', comp=conv, api=pulp_nn_init.PULPNNAPI)

            for i in pulp_nn_init.PULPNNDataPrecisions:
                for j in pulp_nn_init.PULPNNDataPrecisions:
                    for z in pulp_nn_init.PULPNNWeightsPrecisions:
                        for q in pulp_nn_init.PULPNNQuantizationMethods:
                            kernel_to_test = pulp_nn_factory.PULPNNKernel(name='pointwise', inp=i, out=j, wt=z, quant=q, act_prec=a, ext=e)
                            pw=pulp_nn_factory.PULPNNConvolvePointwise(kernel=kernel_to_test, layer=None)
                            pulp_nn_init.PULPNNAPI = pulp_nn_factory.kernel(path_tag='pointwise', comp=pw, api=pulp_nn_init.PULPNNAPI)

            if e == 'XpulpV2':
                for i in pulp_nn_init.PULPNNDataPrecisions:
                    for j in pulp_nn_init.PULPNNWeightsPrecisions:
                        for q in pulp_nn_init.PULPNNQuantizationMethods:
                            kernel_to_test = pulp_nn_factory.PULPNNKernel(name='matmul', inp=8, out=i, wt=j, quant=q, act_prec=a, ext=e)
                            matmul=pulp_nn_factory.PULPNNMatMul(kernel=kernel_to_test, layer=None)
                            pulp_nn_init.PULPNNAPI = pulp_nn_factory.kernel(path_tag='matmul', comp=matmul, api=pulp_nn_init.PULPNNAPI)

            elif e == 'XpulpNN':
                for i in pulp_nn_init.PULPNNDataPrecisions:
                    for j in pulp_nn_init.PULPNNWeightsPrecisions:
                        for z in pulp_nn_init.PULPNNWeightsPrecisions:
                            for q in pulp_nn_init.PULPNNQuantizationMethods:
                                kernel_to_test = pulp_nn_factory.PULPNNKernel(name='matmul', inp=i, out=j, wt=z, quant=q, act_prec=a, ext=e)
                                matmul=pulp_nn_factory.PULPNNMatMul(kernel=kernel_to_test, layer=None)
                                pulp_nn_init.PULPNNAPI = pulp_nn_factory.kernel(path_tag='matmul', comp=matmul, api=pulp_nn_init.PULPNNAPI)                

            for i in pulp_nn_init.PULPNNDataPrecisions:
                for j in pulp_nn_init.PULPNNDataPrecisions:
                    for z in pulp_nn_init.PULPNNWeightsPrecisions:
                        for q in pulp_nn_init.PULPNNQuantizationMethods:
                            kernel_to_test = pulp_nn_factory.PULPNNKernel(name='depthwise', inp=i, out=j, wt=z, quant=q, act_prec=a, ext=e)
                            dw=pulp_nn_factory.PULPNNConvolveDepthwise(kernel=kernel_to_test, layer=None)
                            pulp_nn_init.PULPNNAPI = pulp_nn_factory.kernel(path_tag='depthwise', comp=dw, api=pulp_nn_init.PULPNNAPI)

            for i in pulp_nn_init.PULPNNDataPrecisions:
                for z in pulp_nn_init.PULPNNWeightsPrecisions:
                    kernel_to_test = pulp_nn_factory.PULPNNKernel(name='linear_no_quant', inp=i, out=32, wt=z, quant=None, act_prec=a, ext=e)
                    lin_nq=pulp_nn_factory.PULPNNLinearNoQuant(kernel=kernel_to_test, layer=None)
                    pulp_nn_init.PULPNNAPI = pulp_nn_factory.kernel(path_tag='linear_nq', comp=lin_nq, api=pulp_nn_init.PULPNNAPI)

            for i in pulp_nn_init.PULPNNDataPrecisions:
                for j in pulp_nn_init.PULPNNDataPrecisions:
                    for z in pulp_nn_init.PULPNNWeightsPrecisions:
                        for q in pulp_nn_init.PULPNNQuantizationMethods:
                            kernel_to_test = pulp_nn_factory.PULPNNKernel(name='linear_quant', inp=i, out=j, wt=z, quant=q, act_prec=a, ext=e)
                            lin_q=pulp_nn_factory.PULPNNLinearQuant(kernel=kernel_to_test, layer=None)
                            pulp_nn_init.PULPNNAPI = pulp_nn_factory.kernel(path_tag='linear_q', comp=lin_q, api=pulp_nn_init.PULPNNAPI)

            for i in pulp_nn_init.PULPNNDataPrecisions:
                kernel_to_test = pulp_nn_factory.PULPNNKernel(name='maxpool', inp=i, out=None, wt=None, quant=None, act_prec=a, ext=e)
                maxp=pulp_nn_factory.PULPNNMaxPool(kernel=kernel_to_test, layer=None)
                pulp_nn_init.PULPNNAPI = pulp_nn_factory.kernel(path_tag='maxpool', comp=maxp, api=pulp_nn_init.PULPNNAPI)

            for i in pulp_nn_init.PULPNNDataPrecisions:
                kernel_to_test = pulp_nn_factory.PULPNNKernel(name='avgpool', inp=i, out=None, wt=None, quant=None, act_prec=a, ext=e)
                avgp=pulp_nn_factory.PULPNNAvgPool(kernel=kernel_to_test, layer=None)
                pulp_nn_init.PULPNNAPI = pulp_nn_factory.kernel(path_tag='avgpool', comp=avgp, api=pulp_nn_init.PULPNNAPI)

            for i in pulp_nn_init.PULPNNDataPrecisions:
                for j in pulp_nn_init.PULPNNDataPrecisions:
                    if j <= i:
                        kernel_to_test = pulp_nn_factory.PULPNNKernel(name='add', inp=i, out=j, wt=None, quant=None, act_prec=a, ext=e)
                        add=pulp_nn_factory.PULPNNAdd(kernel=kernel_to_test, layer=None)
                        pulp_nn_init.PULPNNAPI = pulp_nn_factory.kernel(path_tag='add', comp=add, api=pulp_nn_init.PULPNNAPI)
                            
        pulp_nn_factory.header(act_prec=a, ext=e, api=pulp_nn_init.PULPNNAPI)
        pulp_nn_init.PULPNNAPI = ""


if __name__ == '__main__':
    
    main()
