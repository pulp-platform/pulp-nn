#
# struct_test_gen.py
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
from include.struct_test import PULPNNSrcDirs, PULPNNInstallPath

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            raise

def mkdir_str(type):
    mkdir_p(PULPNNSrcDirs['src'])
    mkdir_p(PULPNNSrcDirs['support_function'])
    mkdir_p(PULPNNSrcDirs['include'])
    if type=='pointwise':
        mkdir_p(PULPNNSrcDirs['pointwise_convolution'])
        mkdir_p(PULPNNSrcDirs['matmul'])
        mkdir_p(PULPNNSrcDirs['data_allocation_pw'])
        mkdir_p(PULPNNSrcDirs['golden_model_pw'])
    elif type=='depthwise':
        mkdir_p(PULPNNSrcDirs['depthwise_convolution'])
        mkdir_p(PULPNNSrcDirs['data_allocation_dw'])
        mkdir_p(PULPNNSrcDirs['golden_model_dw'])
    elif type=='linear_no_quant':
        mkdir_p(PULPNNSrcDirs['linear_convolution_nq'])
        mkdir_p(PULPNNSrcDirs['data_allocation_ln_nq'])
        mkdir_p(PULPNNSrcDirs['golden_model_ln_nq'])
    elif type=='linear_quant':
        mkdir_p(PULPNNSrcDirs['linear_convolution_q'])
        mkdir_p(PULPNNSrcDirs['data_allocation_ln_q'])
        mkdir_p(PULPNNSrcDirs['golden_model_ln_q'])

    try:
        os.remove(PULPNNInstallPath + "Makefile")
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            pass