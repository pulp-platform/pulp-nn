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
PULPNNSrcDirs = {'script': PULPNNInstallPath + "scripts/",
                'src': PULPNNInstallPath + "src/",
                'inc': PULPNNInstallPath + "inc/",
                'pointwise_convolution': PULPNNInstallPath + "src/PointwiseConvolutions/",
                'matmul': PULPNNInstallPath + "src/MatrixMultiplications/",
                'depthwise_convolution': PULPNNInstallPath + "src/DepthwiseConvolutions/",
                'linear_convolution_nq': PULPNNInstallPath + "src/LinearConvolutionsNoQuant/",
                'linear_convolution_q': PULPNNInstallPath + "src/LinearConvolutionsQuant/",
                'support_function': PULPNNInstallPath + "src/SupportFunctions/"}