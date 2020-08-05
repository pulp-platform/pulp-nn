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
from include.struct_test import PULPNNSrcDirs, PULPNNSrcDirs32bit, PULPNNSrcDirs64bit, PULPNNInstallPath32bit, PULPNNInstallPath64bit 

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            raise

def mkdir_str(type, act_prec):
    if act_prec == '32bit':
        mkdir_p(PULPNNSrcDirs32bit['src'])
        mkdir_p(PULPNNSrcDirs32bit['support_function'])
        mkdir_p(PULPNNSrcDirs32bit['include'])
        if type=='pointwise':
            mkdir_p(PULPNNSrcDirs32bit['pointwise_convolution'])
            mkdir_p(PULPNNSrcDirs32bit['matmul'])
            mkdir_p(PULPNNSrcDirs32bit['data_allocation_pw'])
            mkdir_p(PULPNNSrcDirs32bit['golden_model_pw'])
        elif type=='depthwise':
            mkdir_p(PULPNNSrcDirs32bit['depthwise_convolution'])
            mkdir_p(PULPNNSrcDirs32bit['data_allocation_dw'])
            mkdir_p(PULPNNSrcDirs32bit['golden_model_dw'])
        elif type=='linear_no_quant':
            mkdir_p(PULPNNSrcDirs32bit['linear_convolution_nq'])
            mkdir_p(PULPNNSrcDirs32bit['data_allocation_ln_nq'])
            mkdir_p(PULPNNSrcDirs32bit['golden_model_ln_nq'])
        elif type=='linear_quant':
            mkdir_p(PULPNNSrcDirs32bit['linear_convolution_q'])
            mkdir_p(PULPNNSrcDirs32bit['data_allocation_ln_q'])
            mkdir_p(PULPNNSrcDirs32bit['golden_model_ln_q'])

        try:
            os.remove(PULPNNInstallPath32bit + "Makefile")
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                pass
    elif act_prec == '64bit':
        mkdir_p(PULPNNSrcDirs64bit['src'])
        mkdir_p(PULPNNSrcDirs64bit['support_function'])
        mkdir_p(PULPNNSrcDirs64bit['include'])
        if type=='pointwise':
            mkdir_p(PULPNNSrcDirs64bit['pointwise_convolution'])
            mkdir_p(PULPNNSrcDirs64bit['matmul'])
            mkdir_p(PULPNNSrcDirs64bit['data_allocation_pw'])
            mkdir_p(PULPNNSrcDirs64bit['golden_model_pw'])
        elif type=='depthwise':
            mkdir_p(PULPNNSrcDirs64bit['depthwise_convolution'])
            mkdir_p(PULPNNSrcDirs64bit['data_allocation_dw'])
            mkdir_p(PULPNNSrcDirs64bit['golden_model_dw'])
        elif type=='linear_no_quant':
            mkdir_p(PULPNNSrcDirs64bit['linear_convolution_nq'])
            mkdir_p(PULPNNSrcDirs64bit['data_allocation_ln_nq'])
            mkdir_p(PULPNNSrcDirs64bit['golden_model_ln_nq'])
        elif type=='linear_quant':
            mkdir_p(PULPNNSrcDirs64bit['linear_convolution_q'])
            mkdir_p(PULPNNSrcDirs64bit['data_allocation_ln_q'])
            mkdir_p(PULPNNSrcDirs64bit['golden_model_ln_q'])

        try:
            os.remove(PULPNNInstallPath64bit + "Makefile")
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                pass