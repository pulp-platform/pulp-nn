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
PULPNNTestFolder = PULPNNInstallPath + "test/"
PULPNNSrcDirs = {'script': PULPNNInstallPath + "scripts/",
                'pulp_nn_inc': PULPNNInstallPath + "inc/",
                'pulp_nn_pointwise_convolution': PULPNNInstallPath + "src/PointwiseConvolutions/",
                'pulp_nn_matmul': PULPNNInstallPath + "src/MatrixMultiplications/",
                'pulp_nn_depthwise_convolution': PULPNNInstallPath + "src/DepthwiseConvolutions/",
                'pulp_nn_linear_convolution_nq': PULPNNInstallPath + "src/LinearConvolutionsNoQuant/",
                'pulp_nn_linear_convolution_q': PULPNNInstallPath + "src/LinearConvolutionsQuant/",
                'pulp_nn_support_function': PULPNNInstallPath + "src/SupportFunctions/",
                'include': PULPNNTestFolder + "include/",
                'src': PULPNNTestFolder + "src/",
                'pointwise_convolution': PULPNNTestFolder + "src/PointwiseConvolutions/",
                'matmul': PULPNNTestFolder + "src/MatrixMultiplications/",
                'depthwise_convolution': PULPNNTestFolder + "src/DepthwiseConvolutions/",
                'linear_convolution_nq': PULPNNTestFolder + "src/LinearConvolutionsNoQuant/",
                'linear_convolution_q': PULPNNTestFolder + "src/LinearConvolutionsQuant/",
                'support_function': PULPNNTestFolder + "src/SupportFunctions/",
                'data_allocation_pw': PULPNNTestFolder + "include/DataAllocationPointwiseConvolutions/",
                'data_allocation_dw': PULPNNTestFolder + "include/DataAllocationDepthwiseConvolutions/",
                'data_allocation_ln_nq': PULPNNTestFolder + "include/DataAllocationLinearConvolutionsNoQuant/",
                'data_allocation_ln_q': PULPNNTestFolder + "include/DataAllocationLinearConvolutionsQuant/",
                'golden_model_pw': PULPNNTestFolder + "include/GoldenModelPointwiseConvolutions/",
                'golden_model_dw': PULPNNTestFolder + "include/GoldenModelDepthwiseConvolutions/",
                'golden_model_ln_nq': PULPNNTestFolder + "include/GoldenModelLinearConvolutionsNoQuant/",
                'golden_model_ln_q': PULPNNTestFolder + "include/GoldenModelLinearConvolutionsQuant/",
                'test': PULPNNTestFolder}