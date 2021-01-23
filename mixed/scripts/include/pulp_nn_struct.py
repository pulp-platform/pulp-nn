#
# pulp_nn_struct.py
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
import shutil

PULPNNInstallPath = cwd = os.getcwd() + "/../"
PULPNNSrcDirs = {'script': PULPNNInstallPath + "scripts/"}
PULPNNInstallPath32bit = cwd = os.getcwd() + "/../32bit/" 
PULPNNInstallPath64bit = cwd = os.getcwd() + "/../64bit/"
PULPNNTestFolder32bit = PULPNNInstallPath32bit + "test/"
PULPNNTestFolder64bit = PULPNNInstallPath64bit + "test/"
PULPNNSrcDirs32bit = {'src': PULPNNInstallPath32bit + "src/",
                'inc': PULPNNInstallPath32bit + "include/",
                'convolution': PULPNNInstallPath32bit + "src/Convolution/",
                'pointwise': PULPNNInstallPath32bit + "src/Pointwise/",
                'matmul': PULPNNInstallPath32bit + "src/MatrixMultiplication/",
                'depthwise': PULPNNInstallPath32bit + "src/Depthwise/",
                'linear_nq': PULPNNInstallPath32bit + "src/LinearNoQuant/",
                'linear_q': PULPNNInstallPath32bit + "src/LinearQuant/",
                'support_function': PULPNNInstallPath32bit + "src/SupportFunctions/",
                'pooling': PULPNNInstallPath32bit + "src/Pooling/",
                'maxpool': PULPNNInstallPath32bit + "src/Pooling/MaxPool/",
                'avgpool': PULPNNInstallPath32bit + "src/Pooling/AvgPool/",
                'pulp_nn_include': PULPNNTestFolder32bit + "include/",
                'pulp_nn_src': PULPNNTestFolder32bit + "src/",
                'pulp_nn_convolution': PULPNNTestFolder32bit + "src/Convolution/",
                'pulp_nn_pointwise': PULPNNTestFolder32bit + "src/Pointwise/",
                'pulp_nn_matmul': PULPNNTestFolder32bit + "src/MatrixMultiplication/",
                'pulp_nn_depthwise': PULPNNTestFolder32bit + "src/Depthwise/",
                'pulp_nn_linear_nq': PULPNNTestFolder32bit + "src/LinearNoQuant/",
                'pulp_nn_linear_q': PULPNNTestFolder32bit + "src/LinearQuant/",
                'pulp_nn_maxpool': PULPNNTestFolder32bit + "src/Pooling/MaxPool/",
                'pulp_nn_avgpool': PULPNNTestFolder32bit + "src/Pooling/AvgPool/",
                'pulp_nn_support_function': PULPNNTestFolder32bit + "src/SupportFunctions/",
                'data_allocation_matm': PULPNNTestFolder32bit + "include/DataAllocationMatMul/",
                'data_allocation_conv': PULPNNTestFolder32bit + "include/DataAllocationConvolution/",
                'data_allocation_pw': PULPNNTestFolder32bit + "include/DataAllocationPointwise/",
                'data_allocation_dw': PULPNNTestFolder32bit + "include/DataAllocationDepthwise/",
                'data_allocation_ln_nq': PULPNNTestFolder32bit + "include/DataAllocationLinearNoQuant/",
                'data_allocation_ln_q': PULPNNTestFolder32bit + "include/DataAllocationLinearQuant/",
                'data_allocation_maxp': PULPNNTestFolder32bit + "include/DataAllocationMaxPool/",
                'data_allocation_avgp': PULPNNTestFolder32bit + "include/DataAllocationAvgPool/",
                'golden_model_matm': PULPNNTestFolder32bit + "include/GoldenModelMatMul/",
                'golden_model_conv': PULPNNTestFolder32bit + "include/GoldenModelConvolution/",
                'golden_model_pw': PULPNNTestFolder32bit + "include/GoldenModelPointwise/",
                'golden_model_dw': PULPNNTestFolder32bit + "include/GoldenModelDepthwise/",
                'golden_model_ln_nq': PULPNNTestFolder32bit + "include/GoldenModelLinearNoQuant/",
                'golden_model_ln_q': PULPNNTestFolder32bit + "include/GoldenModelLinearQuant/",
                'golden_model_maxp': PULPNNTestFolder32bit + "include/GoldenModelMaxPool/",
                'golden_model_avgp': PULPNNTestFolder32bit + "include/GoldenModelAvgPool/",
                'test': PULPNNTestFolder32bit}
PULPNNSrcDirs64bit = {'src': PULPNNInstallPath64bit + "src/",
                'inc': PULPNNInstallPath64bit + "include/",
                'convolution': PULPNNInstallPath64bit + "src/Convolution/",
                'pointwise': PULPNNInstallPath64bit + "src/Pointwise/",
                'matmul': PULPNNInstallPath64bit + "src/MatrixMultiplication/",
                'depthwise': PULPNNInstallPath64bit + "src/Depthwise/",
                'linear_nq': PULPNNInstallPath64bit + "src/LinearNoQuant/",
                'linear_q': PULPNNInstallPath64bit + "src/LinearQuant/",
                'support_function': PULPNNInstallPath64bit + "src/SupportFunctions/",
                'pooling': PULPNNInstallPath64bit + "src/Pooling/",
                'maxpool': PULPNNInstallPath64bit + "src/Pooling/MaxPool/",
                'avgpool': PULPNNInstallPath64bit + "src/Pooling/AvgPool/",
                'pulp_nn_include': PULPNNTestFolder64bit + "include/",
                'pulp_nn_src': PULPNNTestFolder64bit + "src/",
                'pulp_nn_convolution': PULPNNTestFolder64bit + "src/Convolution/",
                'pulp_nn_pointwise': PULPNNTestFolder64bit + "src/Pointwise/",
                'pulp_nn_matmul': PULPNNTestFolder64bit + "src/MatrixMultiplication/",
                'pulp_nn_depthwise': PULPNNTestFolder64bit + "src/Depthwise/",
                'pulp_nn_linear_nq': PULPNNTestFolder64bit + "src/LinearNoQuant/",
                'pulp_nn_linear_q': PULPNNTestFolder64bit + "src/LinearQuant/",
                'pulp_nn_maxpool': PULPNNTestFolder64bit + "src/Pooling/MaxPool/",
                'pulp_nn_avgpool': PULPNNTestFolder64bit + "src/Pooling/AvgPool/",
                'pulp_nn_support_function': PULPNNTestFolder64bit + "src/SupportFunctions/",
                'data_allocation_matm': PULPNNTestFolder64bit + "include/DataAllocationMatMul/",
                'data_allocation_conv': PULPNNTestFolder64bit + "include/DataAllocationConvolution/",
                'data_allocation_pw': PULPNNTestFolder64bit + "include/DataAllocationPointwise/",
                'data_allocation_dw': PULPNNTestFolder64bit + "include/DataAllocationDepthwise/",
                'data_allocation_ln_nq': PULPNNTestFolder64bit + "include/DataAllocationLinearNoQuant/",
                'data_allocation_ln_q': PULPNNTestFolder64bit + "include/DataAllocationLinearQuant/",
                'data_allocation_maxp': PULPNNTestFolder64bit + "include/DataAllocationMaxPool/",
                'golden_model_matm': PULPNNTestFolder64bit + "include/GoldenModelMatMul/",
                'golden_model_conv': PULPNNTestFolder64bit + "include/GoldenModelConvolution/",
                'golden_model_conv': PULPNNTestFolder64bit + "include/GoldenModelConvolution/",
                'golden_model_pw': PULPNNTestFolder64bit + "include/GoldenModelPointwise/",
                'golden_model_dw': PULPNNTestFolder64bit + "include/GoldenModelDepthwise/",
                'golden_model_ln_nq': PULPNNTestFolder64bit + "include/GoldenModelLinearNoQuant/",
                'golden_model_ln_q': PULPNNTestFolder64bit + "include/GoldenModelLinearQuant/",
                'golden_model_maxp': PULPNNTestFolder64bit + "include/GoldenModelMaxPool/",
                'golden_model_avgp': PULPNNTestFolder64bit + "include/GoldenModelAvgPool/",
                'test': PULPNNTestFolder64bit}

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            raise

def mkdir_src(act_prec):
    if act_prec == '32bit':
        mkdir_p(PULPNNSrcDirs32bit['src'])
        mkdir_p(PULPNNSrcDirs32bit['inc'])
        mkdir_p(PULPNNSrcDirs32bit['convolution'])
        mkdir_p(PULPNNSrcDirs32bit['pointwise'])
        mkdir_p(PULPNNSrcDirs32bit['matmul'])
        mkdir_p(PULPNNSrcDirs32bit['depthwise'])
        mkdir_p(PULPNNSrcDirs32bit['linear_nq'])
        mkdir_p(PULPNNSrcDirs32bit['linear_q'])
        mkdir_p(PULPNNSrcDirs32bit['support_function'])
        mkdir_p(PULPNNSrcDirs32bit['pooling'])
        mkdir_p(PULPNNSrcDirs32bit['maxpool'])
        mkdir_p(PULPNNSrcDirs32bit['avgpool'])

    elif act_prec == '64bit':
        mkdir_p(PULPNNSrcDirs64bit['src'])
        mkdir_p(PULPNNSrcDirs64bit['inc'])
        mkdir_p(PULPNNSrcDirs64bit['convolution'])
        mkdir_p(PULPNNSrcDirs64bit['pointwise'])
        mkdir_p(PULPNNSrcDirs64bit['matmul'])
        mkdir_p(PULPNNSrcDirs64bit['depthwise'])
        mkdir_p(PULPNNSrcDirs64bit['linear_nq'])
        mkdir_p(PULPNNSrcDirs64bit['linear_q'])
        mkdir_p(PULPNNSrcDirs64bit['support_function'])
        mkdir_p(PULPNNSrcDirs64bit['pooling'])
        mkdir_p(PULPNNSrcDirs64bit['maxpool'])
        mkdir_p(PULPNNSrcDirs64bit['avgpool']) 

def mkdir_test(kernel, act_prec):
    if act_prec == '32bit':
        mkdir_p(PULPNNSrcDirs32bit['pulp_nn_src'])
        mkdir_p(PULPNNSrcDirs32bit['pulp_nn_support_function'])
        mkdir_p(PULPNNSrcDirs32bit['pulp_nn_include'])
        if kernel=='matmul':
            mkdir_p(PULPNNSrcDirs32bit['pulp_nn_matmul'])
            mkdir_p(PULPNNSrcDirs32bit['data_allocation_matm'])
            mkdir_p(PULPNNSrcDirs32bit['golden_model_matm'])            
        elif kernel=='pointwise':
            mkdir_p(PULPNNSrcDirs32bit['pulp_nn_pointwise'])
            mkdir_p(PULPNNSrcDirs32bit['pulp_nn_matmul'])
            mkdir_p(PULPNNSrcDirs32bit['data_allocation_pw'])
            mkdir_p(PULPNNSrcDirs32bit['golden_model_pw'])
        elif kernel=='convolution':
            mkdir_p(PULPNNSrcDirs32bit['pulp_nn_convolution'])
            mkdir_p(PULPNNSrcDirs32bit['pulp_nn_matmul'])
            mkdir_p(PULPNNSrcDirs32bit['data_allocation_conv'])
            mkdir_p(PULPNNSrcDirs32bit['golden_model_conv'])
        elif kernel=='depthwise':
            mkdir_p(PULPNNSrcDirs32bit['pulp_nn_depthwise'])
            mkdir_p(PULPNNSrcDirs32bit['data_allocation_dw'])
            mkdir_p(PULPNNSrcDirs32bit['golden_model_dw'])
        elif kernel=='linear_no_quant':
            mkdir_p(PULPNNSrcDirs32bit['pulp_nn_linear_nq'])
            mkdir_p(PULPNNSrcDirs32bit['data_allocation_ln_nq'])
            mkdir_p(PULPNNSrcDirs32bit['golden_model_ln_nq'])
        elif kernel=='linear_quant':
            mkdir_p(PULPNNSrcDirs32bit['pulp_nn_linear_q'])
            mkdir_p(PULPNNSrcDirs32bit['data_allocation_ln_q'])
            mkdir_p(PULPNNSrcDirs32bit['golden_model_ln_q'])
        elif kernel=='maxpool':
            mkdir_p(PULPNNSrcDirs32bit['pulp_nn_maxpool'])
            mkdir_p(PULPNNSrcDirs32bit['data_allocation_maxp'])
            mkdir_p(PULPNNSrcDirs32bit['golden_model_maxp'])
        elif kernel=='avgpool':
            mkdir_p(PULPNNSrcDirs32bit['pulp_nn_avgpool'])
            mkdir_p(PULPNNSrcDirs32bit['data_allocation_avgp'])
            mkdir_p(PULPNNSrcDirs32bit['golden_model_avgp'])

        try:
            os.remove(PULPNNInstallPath32bit + "Makefile")
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                pass
    elif act_prec == '64bit':
        mkdir_p(PULPNNSrcDirs64bit['pulp_nn_src'])
        mkdir_p(PULPNNSrcDirs64bit['pulp_nn_support_function'])
        mkdir_p(PULPNNSrcDirs64bit['pulp_nn_include'])
        if kernel=='matmul':
            mkdir_p(PULPNNSrcDirs64bit['pulp_nn_matmul'])
            mkdir_p(PULPNNSrcDirs64bit['data_allocation_matm'])
            mkdir_p(PULPNNSrcDirs64bit['golden_model_matm'])            
        elif kernel=='pointwise':
            mkdir_p(PULPNNSrcDirs64bit['pulp_nn_pointwise'])
            mkdir_p(PULPNNSrcDirs64bit['pulp_nn_matmul'])
            mkdir_p(PULPNNSrcDirs64bit['data_allocation_pw'])
            mkdir_p(PULPNNSrcDirs64bit['golden_model_pw'])
        elif kernel=='convolution':
            mkdir_p(PULPNNSrcDirs64bit['pulp_nn_convolution'])
            mkdir_p(PULPNNSrcDirs64bit['pulp_nn_matmul'])
            mkdir_p(PULPNNSrcDirs64bit['data_allocation_conv'])
            mkdir_p(PULPNNSrcDirs64bit['golden_model_conv'])
        elif kernel=='depthwise':
            mkdir_p(PULPNNSrcDirs64bit['pulp_nn_depthwise'])
            mkdir_p(PULPNNSrcDirs64bit['data_allocation_dw'])
            mkdir_p(PULPNNSrcDirs64bit['golden_model_dw'])
        elif kernel=='linear_no_quant':
            mkdir_p(PULPNNSrcDirs64bit['pulp_nn_linear_nq'])
            mkdir_p(PULPNNSrcDirs64bit['data_allocation_ln_nq'])
            mkdir_p(PULPNNSrcDirs64bit['golden_model_ln_nq'])
        elif kernel=='linear_quant':
            mkdir_p(PULPNNSrcDirs64bit['pulp_nn_linear_q'])
            mkdir_p(PULPNNSrcDirs64bit['data_allocation_ln_q'])
            mkdir_p(PULPNNSrcDirs64bit['golden_model_ln_q'])
        elif kernel=='maxpool':
            mkdir_p(PULPNNSrcDirs64bit['pulp_nn_maxpool'])
            mkdir_p(PULPNNSrcDirs64bit['data_allocation_maxp'])
            mkdir_p(PULPNNSrcDirs64bit['golden_model_maxp'])
        elif kernel=='avgpool':
            mkdir_p(PULPNNSrcDirs64bit['pulp_nn_avgpool'])
            mkdir_p(PULPNNSrcDirs64bit['data_allocation_avgp'])
            mkdir_p(PULPNNSrcDirs64bit['golden_model_avgp'])

        try:
            os.remove(PULPNNInstallPath64bit + "Makefile")
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                pass      