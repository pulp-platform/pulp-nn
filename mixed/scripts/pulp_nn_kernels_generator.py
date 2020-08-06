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

from mako.template import Template
from include import struct_comp_gen, comp_gen, utils
from include.struct_comp import PULPNNSrcDirs32bit, PULPNNSrcDirs64bit

for a in utils.BN_ACTIVATIONS:
    struct_comp_gen.mkdir_str(a)

    for i in utils.PULPNNDataPrecisions:
        for j in utils.PULPNNDataPrecisions:
            for z in utils.PULPNNWeightsPrecisions:
                for q in utils.PULPNNQuantizationMethods:
                    c = comp_gen.PULPNNConvolve(in_data_t=i, out_data_t=j, wt_data_t=z, quantization=q, act_prec=a)
                    utils.PULPNNAPI += c.generate_api() + "\n"
                    if a == '32bit':
                        new_file = open(PULPNNSrcDirs32bit['pointwise_convolution'] + c.filename, 'w')
                    elif a == '64bit':
                        new_file = open(PULPNNSrcDirs64bit['pointwise_convolution'] + c.filename, 'w')
                    new_file.write(c.generate_code())
                    new_file.close()

    for i in utils.PULPNNDataPrecisions:
        for j in utils.PULPNNWeightsPrecisions:
            for q in utils.PULPNNQuantizationMethods:
                c = comp_gen.PULPNNMatMul(out_data_t=i, wt_data_t=j, quantization=q, act_prec=a)
                utils.PULPNNAPI += c.generate_api() + "\n"
                if a == '32bit':
                    new_file = open(PULPNNSrcDirs32bit['matmul'] + c.filename, 'w')
                elif a == '64bit':
                    new_file = open(PULPNNSrcDirs64bit['matmul'] + c.filename, 'w')
                new_file.write(c.generate_code())
                new_file.close()

    for i in utils.PULPNNDataPrecisions:
        for j in utils.PULPNNDataPrecisions:
            for z in utils.PULPNNWeightsPrecisions:
                for q in utils.PULPNNQuantizationMethods:
                    c = comp_gen.PULPNNDepthwise(in_data_t=i, out_data_t=j, wt_data_t=z, quantization=q, act_prec=a)
                    utils.PULPNNAPI += c.generate_api() + "\n"
                    if a == '32bit':
                        new_file = open(PULPNNSrcDirs32bit['depthwise_convolution'] + c.filename, 'w')
                    elif a == '64bit':
                        new_file = open(PULPNNSrcDirs64bit['depthwise_convolution'] + c.filename, 'w')
                    new_file.write(c.generate_code())
                    new_file.close()

    for i in utils.PULPNNDataPrecisions:
        for z in utils.PULPNNWeightsPrecisions:
            c = comp_gen.PULPNNLinearNoQuant(in_data_t=i, wt_data_t=z, act_prec=a)
            utils.PULPNNAPI += c.generate_api() + "\n"
            if a == '32bit':
                new_file = open(PULPNNSrcDirs32bit['linear_convolution_nq'] + c.filename, 'w')
            elif a == '64bit':
                new_file = open(PULPNNSrcDirs64bit['linear_convolution_nq'] + c.filename, 'w')
            new_file.write(c.generate_code())
            new_file.close()

    for i in utils.PULPNNDataPrecisions:
        for j in utils.PULPNNDataPrecisions:
            for z in utils.PULPNNWeightsPrecisions:
                c = comp_gen.PULPNNLinearQuant(in_data_t=i, out_data_t=j, wt_data_t=z, act_prec=a)
                utils.PULPNNAPI += c.generate_api() + "\n"
                if a == '32bit':
                    new_file = open(PULPNNSrcDirs32bit['linear_convolution_q'] + c.filename, 'w')
                elif a == '64bit':
                    new_file = open(PULPNNSrcDirs64bit['linear_convolution_q'] + c.filename, 'w')
                new_file.write(c.generate_code())
                new_file.close()
                    
    if a == '32bit':
        new_file = open(PULPNNSrcDirs32bit['inc'] + "/pulp_nn_kernels.h", 'w')
    elif a == '64bit':
        new_file = open(PULPNNSrcDirs64bit['inc'] + "/pulp_nn_kernels.h", 'w')
    new_file.write(Template(filename="templates/pulp_nn_kernels.h").render(PULPNNAPI=utils.PULPNNAPI))
    new_file.close()

    utils.PULPNNAPI = ""
