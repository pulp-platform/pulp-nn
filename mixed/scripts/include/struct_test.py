#
# struct_test.py
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

import os

PULPNNInstallPath = cwd = os.getcwd() + "/../"
PULPNNSrcDirs = {'script': PULPNNInstallPath + "scripts/"}
PULPNNInstallPath32bit = cwd = os.getcwd() + "/../32bit/"
PULPNNInstallPath64bit = cwd = os.getcwd() + "/../64bit/"
PULPNNTestFolder32bit = PULPNNInstallPath32bit + "test/"
PULPNNTestFolder64bit = PULPNNInstallPath64bit + "test/"
PULPNNSrcDirs32bit = {'pulp_nn_inc': PULPNNInstallPath32bit + "inc/",
                'pulp_nn_pointwise_convolution': PULPNNInstallPath32bit + "src/PointwiseConvolutions/",
                'pulp_nn_matmul': PULPNNInstallPath32bit + "src/MatrixMultiplications/",
                'pulp_nn_depthwise_convolution': PULPNNInstallPath32bit + "src/DepthwiseConvolutions/",
                'pulp_nn_linear_convolution_nq': PULPNNInstallPath32bit + "src/LinearConvolutionsNoQuant/",
                'pulp_nn_linear_convolution_q': PULPNNInstallPath32bit + "src/LinearConvolutionsQuant/",
                'pulp_nn_support_function': PULPNNInstallPath32bit + "src/SupportFunctions/",
                'include': PULPNNTestFolder32bit + "include/",
                'src': PULPNNTestFolder32bit + "src/",
                'pointwise_convolution': PULPNNTestFolder32bit + "src/PointwiseConvolutions/",
                'matmul': PULPNNTestFolder32bit + "src/MatrixMultiplications/",
                'depthwise_convolution': PULPNNTestFolder32bit + "src/DepthwiseConvolutions/",
                'linear_convolution_nq': PULPNNTestFolder32bit + "src/LinearConvolutionsNoQuant/",
                'linear_convolution_q': PULPNNTestFolder32bit + "src/LinearConvolutionsQuant/",
                'support_function': PULPNNTestFolder32bit + "src/SupportFunctions/",
                'data_allocation_pw': PULPNNTestFolder32bit + "include/DataAllocationPointwiseConvolutions/",
                'data_allocation_dw': PULPNNTestFolder32bit + "include/DataAllocationDepthwiseConvolutions/",
                'data_allocation_ln_nq': PULPNNTestFolder32bit + "include/DataAllocationLinearConvolutionsNoQuant/",
                'data_allocation_ln_q': PULPNNTestFolder32bit + "include/DataAllocationLinearConvolutionsQuant/",
                'golden_model_pw': PULPNNTestFolder32bit + "include/GoldenModelPointwiseConvolutions/",
                'golden_model_dw': PULPNNTestFolder32bit + "include/GoldenModelDepthwiseConvolutions/",
                'golden_model_ln_nq': PULPNNTestFolder32bit + "include/GoldenModelLinearConvolutionsNoQuant/",
                'golden_model_ln_q': PULPNNTestFolder32bit + "include/GoldenModelLinearConvolutionsQuant/",
                'test': PULPNNTestFolder32bit}
PULPNNSrcDirs64bit = {'pulp_nn_inc': PULPNNInstallPath64bit + "inc/",
                'pulp_nn_pointwise_convolution': PULPNNInstallPath64bit + "src/PointwiseConvolutions/",
                'pulp_nn_matmul': PULPNNInstallPath64bit + "src/MatrixMultiplications/",
                'pulp_nn_depthwise_convolution': PULPNNInstallPath64bit + "src/DepthwiseConvolutions/",
                'pulp_nn_linear_convolution_nq': PULPNNInstallPath64bit + "src/LinearConvolutionsNoQuant/",
                'pulp_nn_linear_convolution_q': PULPNNInstallPath64bit + "src/LinearConvolutionsQuant/",
                'pulp_nn_support_function': PULPNNInstallPath64bit + "src/SupportFunctions/",
                'include': PULPNNTestFolder64bit + "include/",
                'src': PULPNNTestFolder64bit + "src/",
                'pointwise_convolution': PULPNNTestFolder64bit + "src/PointwiseConvolutions/",
                'matmul': PULPNNTestFolder64bit + "src/MatrixMultiplications/",
                'depthwise_convolution': PULPNNTestFolder64bit + "src/DepthwiseConvolutions/",
                'linear_convolution_nq': PULPNNTestFolder64bit + "src/LinearConvolutionsNoQuant/",
                'linear_convolution_q': PULPNNTestFolder64bit + "src/LinearConvolutionsQuant/",
                'support_function': PULPNNTestFolder64bit + "src/SupportFunctions/",
                'data_allocation_pw': PULPNNTestFolder64bit + "include/DataAllocationPointwiseConvolutions/",
                'data_allocation_dw': PULPNNTestFolder64bit + "include/DataAllocationDepthwiseConvolutions/",
                'data_allocation_ln_nq': PULPNNTestFolder64bit + "include/DataAllocationLinearConvolutionsNoQuant/",
                'data_allocation_ln_q': PULPNNTestFolder64bit + "include/DataAllocationLinearConvolutionsQuant/",
                'golden_model_pw': PULPNNTestFolder64bit + "include/GoldenModelPointwiseConvolutions/",
                'golden_model_dw': PULPNNTestFolder64bit + "include/GoldenModelDepthwiseConvolutions/",
                'golden_model_ln_nq': PULPNNTestFolder64bit + "include/GoldenModelLinearConvolutionsNoQuant/",
                'golden_model_ln_q': PULPNNTestFolder64bit + "include/GoldenModelLinearConvolutionsQuant/",
                'test': PULPNNTestFolder64bit}