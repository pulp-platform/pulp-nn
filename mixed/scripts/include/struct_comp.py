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
PULPNNSrcDirs32bit = {'src': PULPNNInstallPath32bit + "src/",
                'inc': PULPNNInstallPath32bit + "include/",
                'pointwise_convolution': PULPNNInstallPath32bit + "src/PointwiseConvolutions/",
                'matmul': PULPNNInstallPath32bit + "src/MatrixMultiplications/",
                'depthwise_convolution': PULPNNInstallPath32bit + "src/DepthwiseConvolutions/",
                'linear_convolution_nq': PULPNNInstallPath32bit + "src/LinearConvolutionsNoQuant/",
                'linear_convolution_q': PULPNNInstallPath32bit + "src/LinearConvolutionsQuant/",
                'support_function': PULPNNInstallPath32bit + "src/SupportFunctions/"}
PULPNNSrcDirs64bit = {'src': PULPNNInstallPath64bit + "src/",
                'inc': PULPNNInstallPath64bit + "include/",
                'pointwise_convolution': PULPNNInstallPath64bit + "src/PointwiseConvolutions/",
                'matmul': PULPNNInstallPath64bit + "src/MatrixMultiplications/",
                'depthwise_convolution': PULPNNInstallPath64bit + "src/DepthwiseConvolutions/",
                'linear_convolution_nq': PULPNNInstallPath64bit + "src/LinearConvolutionsNoQuant/",
                'linear_convolution_q': PULPNNInstallPath64bit + "src/LinearConvolutionsQuant/",
                'support_function': PULPNNInstallPath64bit + "src/SupportFunctions/"}