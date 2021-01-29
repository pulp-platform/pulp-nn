/*
 * ${config.filename}
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __PULPNN_TEST_DATA_ALLOCATION__
#define __PULPNN_TEST_DATA_ALLOCATION__

#define DIM_IM_IN_X ${config.layer.dim_in_x}
#define DIM_IM_IN_Y ${config.layer.dim_in_y}
#define CH_IM_IN ${config.layer.ch_in}
#define DIM_IM_OUT_X ${config.layer.dim_out_x}
#define DIM_IM_OUT_Y ${config.layer.dim_out_y}
#define CH_IM_OUT ${config.layer.ch_out}
%if config.kernel.type != 'add':
#define DIM_KERNEL_X ${config.layer.ker_x}
#define DIM_KERNEL_Y ${config.layer.ker_y}
#define PADDING_Y_TOP ${config.layer.pad_y_top}
#define PADDING_Y_BOTTOM ${config.layer.pad_y_bot}
#define PADDING_X_LEFT ${config.layer.pad_x_left}
#define PADDING_X_RIGHT ${config.layer.pad_x_right}
#define STRIDE_X ${config.layer.stride_x}
#define STRIDE_Y ${config.layer.stride_y}
%endif
%if config.kernel.type == 'maxpool' or config.kernel.type == 'avgpool':
#define POOL_KERNEL ${config.layer.pool_kernel}
#define POOL_STRIDE ${config.layer.pool_stride}
%endif

%if config.kernel.type != 'add':
%if config.layer.bn == True:
PI_L2 int32_t KAPPA_L2[CH_IM_OUT] = KAPPA;
PI_L1 int32_t KAPPA_L1[CH_IM_OUT];
PI_L2 int32_t LAMBDA_L2[CH_IM_OUT] = LAMBDA;
PI_L1 int32_t LAMBDA_L1[CH_IM_OUT];
%else:
PI_L1 int32_t *KAPPA_L1 = NULL;
PI_L1 int32_t *LAMBDA_L1 = NULL;
%endif
%if config.kernel.quantization == 'thresholds' and config.kernel.out_data_t == 2:
PI_L2 int16_t THR_INT2_L2[CH_IM_OUT << 2] = THR_INT2;
PI_L1 int16_t THR_INT2_L1[CH_IM_OUT << 2];
%elif config.kernel.quantization == 'thresholds' and config.kernel.out_data_t == 4:
PI_L2 int16_t THR_INT4_L2[CH_IM_OUT << 4] = THR_INT4;
PI_L1 int16_t THR_INT4_L1[CH_IM_OUT << 4];
%endif
%endif

%if config.kernel.type == 'matmul':
PI_L2 uint8_t IN_INT8_L2[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN)] = IN_INT8;
PI_L1 uint8_t IN_INT8_L1[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN)];
%if config.kernel.out_data_t == 2:
PI_L2 uint8_t OUT_INT2_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT2;
PI_L2 uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
PI_L2 uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
PI_L1 uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
%elif config.kernel.out_data_t == 4:
PI_L2 uint8_t OUT_INT4_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT4;
PI_L2 uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
PI_L2 uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
PI_L1 uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
%else:
PI_L2 uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT8;
PI_L2 uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)];
PI_L1 uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)];
%endif
%if config.kernel.wt_data_t == 2:
PI_L2 int8_t WEIGHT_INT2_L2[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT)] = WEIGHT_INT2;
PI_L1 int8_t WEIGHT_INT8_L1[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT) >> 2];
%elif config.kernel.wt_data_t == 4:
PI_L2 int8_t WEIGHT_INT4_L2[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT)] = WEIGHT_INT4;
PI_L1 int8_t WEIGHT_INT8_L1[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT) >> 1];
%elif config.kernel.wt_data_t == 8:
PI_L2 int8_t WEIGHT_INT8_L2[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT)] = WEIGHT_INT8;
PI_L1 int8_t WEIGHT_INT8_L1[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT)];
%endif
%if config.layer.bias == True:
PI_L1 int8_t BIAS_L1[CH_IM_OUT] = BIAS;
%else:
PI_L1 int8_t BIAS_L1[CH_IM_OUT] = {0};
%endif
%elif config.kernel.type == 'convolution' or config.kernel.type == 'pointwise':
%if config.kernel.in_data_t == 2:
PI_L2 uint8_t IN_INT2_L2[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN)] = IN_INT2;
PI_L1 uint8_t IN_INT8_L1[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 2];
%elif config.kernel.in_data_t == 4:
PI_L2 uint8_t IN_INT4_L2[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN)] = IN_INT4;
PI_L1 uint8_t IN_INT8_L1[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 1];
%else:
PI_L2 uint8_t IN_INT8_L2[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN)] = IN_INT8;
PI_L1 uint8_t IN_INT8_L1[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN)];
%endif
%if config.kernel.out_data_t == 2:
PI_L2 uint8_t OUT_INT2_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT2;
PI_L2 uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
PI_L2 uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
PI_L1 uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
%elif config.kernel.out_data_t == 4:
PI_L2 uint8_t OUT_INT4_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT4;
PI_L2 uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
PI_L2 uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
PI_L1 uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
%else:
PI_L2 uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT8;
PI_L2 uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)];
PI_L1 uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)];
%endif
%if config.kernel.wt_data_t == 2:
PI_L2 int8_t WEIGHT_INT2_L2[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT)] = WEIGHT_INT2;
PI_L1 int8_t WEIGHT_INT8_L1[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT) >> 2];
%elif config.kernel.wt_data_t == 4:
PI_L2 int8_t WEIGHT_INT4_L2[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT)] = WEIGHT_INT4;
PI_L1 int8_t WEIGHT_INT8_L1[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT) >> 1];
%elif config.kernel.wt_data_t == 8:
PI_L2 int8_t WEIGHT_INT8_L2[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT)] = WEIGHT_INT8;
PI_L1 int8_t WEIGHT_INT8_L1[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT)];
%endif
PI_L1 uint8_t IM2COL_L1[((CH_IM_IN * DIM_KERNEL_X * DIM_KERNEL_Y) << 1) * NUM_CORES];
%if config.layer.bias == True:
PI_L1 int8_t BIAS_L1[CH_IM_OUT] = BIAS;
%else:
PI_L1 int8_t BIAS_L1[CH_IM_OUT] = {0};
%endif
%elif config.kernel.type == 'depthwise':
%if config.kernel.in_data_t == 2:
PI_L2 uint8_t IN_INT2_L2_HWC[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN)] = IN_INT2;
PI_L2 uint8_t IN_INT8_L2_HWC[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 2];
PI_L1 uint8_t IN_INT8_L1_CHW[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 2];
%elif config.kernel.in_data_t == 4:
PI_L2 uint8_t IN_INT4_L2_HWC[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN)] = IN_INT4;
PI_L2 uint8_t IN_INT8_L2_HWC[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 1];
PI_L1 uint8_t IN_INT8_L1_CHW[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 1];
%else:
PI_L2 uint8_t IN_INT8_L2_HWC[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN)] = IN_INT8;
PI_L1 uint8_t IN_INT8_L1_CHW[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN)];
%endif
%if config.kernel.out_data_t == 2:
PI_L2 uint8_t OUT_INT2_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT2;
PI_L2 uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
PI_L2 uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
PI_L1 uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
%elif config.kernel.out_data_t == 4:
PI_L2 uint8_t OUT_INT4_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT4;
PI_L2 uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
PI_L2 uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
PI_L1 uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
%else:
PI_L2 uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT8;
PI_L2 uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)];
PI_L1 uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)];
%endif
%if config.kernel.wt_data_t == 2:
PI_L2 int8_t WEIGHT_INT2_L2_CHW[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN)] = WEIGHT_INT2;
PI_L2 int8_t WEIGHT_INT2_L2_HWC[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN)];
PI_L2 int8_t WEIGHT_INT8_L2_HWC[((DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN) >> 2)];
PI_L1 int8_t WEIGHT_INT8_L1_CHW[((DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN) >> 2)];
%elif config.kernel.wt_data_t == 4:
PI_L2 int8_t WEIGHT_INT4_L2_CHW[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN)] = WEIGHT_INT4;
PI_L2 int8_t WEIGHT_INT4_L2_HWC[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN)];
PI_L2 int8_t WEIGHT_INT8_L2_HWC[((DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN) >> 1)];
PI_L1 int8_t WEIGHT_INT8_L1_CHW[((DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN) >> 1)];
%elif config.kernel.wt_data_t == 8:
PI_L2 int8_t WEIGHT_INT8_L2_CHW[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN)] = WEIGHT_INT8;
PI_L1 int8_t WEIGHT_INT8_L1_CHW[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN)];
%endif
%if config.less_precision == 2:
PI_L1 uint8_t IM2COL_L1[(((DIM_KERNEL_X * (DIM_IM_IN_Y + PADDING_Y_TOP + PADDING_Y_BOTTOM)) + DIM_KERNEL_X) << 2) * NUM_CORES];
PI_L1 int8_t WTBUFF_L1[((DIM_KERNEL_Y * DIM_KERNEL_X) << 2) * NUM_CORES];
%elif config.less_precision == 4:
PI_L1 uint8_t IM2COL_L1[(((DIM_KERNEL_X * (DIM_IM_IN_Y + PADDING_Y_TOP + PADDING_Y_BOTTOM)) + DIM_KERNEL_X) << 1) * NUM_CORES];
PI_L1 int8_t WTBUFF_L1[((DIM_KERNEL_Y * DIM_KERNEL_X) << 1) * NUM_CORES];
%elif config.less_precision == 8:
PI_L1 uint8_t IM2COL_L1[((DIM_KERNEL_X * (DIM_IM_IN_Y + PADDING_Y_TOP + PADDING_Y_BOTTOM)) + DIM_KERNEL_X) * NUM_CORES];
PI_L1 int8_t WTBUFF_L1[DIM_KERNEL_Y * DIM_KERNEL_X * NUM_CORES];
%endif
%if config.layer.bias == True:
PI_L1 int8_t BIAS_L1[CH_IM_OUT] = BIAS;
%else:
PI_L1 int8_t BIAS_L1[CH_IM_OUT] = {0};
%endif
%elif config.kernel.type == 'linear_no_quant':
%if config.kernel.in_data_t == 8:
PI_L2 uint8_t IN_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN_INT8;
PI_L1 uint8_t IN_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)];
%elif config.kernel.in_data_t == 4:
PI_L2 uint8_t IN_INT4_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN_INT4;
PI_L2 uint8_t IN_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 1];
PI_L1 uint8_t IN_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 1];
%elif config.kernel.in_data_t == 2:
PI_L2 uint8_t IN_INT2_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN_INT2;
PI_L2 uint8_t IN_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 2];
PI_L1 uint8_t IN_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 2];
%endif
%if config.kernel.wt_data_t == 8:
PI_L2 int8_t WEIGHT_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT)] = WEIGHT_INT8;
PI_L1 int8_t WEIGHT_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT)];
%elif config.kernel.wt_data_t == 4:
PI_L2 int8_t WEIGHT_INT4_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT)] = WEIGHT_INT4;
PI_L2 int8_t WEIGHT_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT) >> 1];
PI_L1 int8_t WEIGHT_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT) >> 1];
%elif config.kernel.wt_data_t == 2:
PI_L2 int8_t WEIGHT_INT2_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT)] = WEIGHT_INT2;
PI_L2 int8_t WEIGHT_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT) >> 2];
PI_L1 int8_t WEIGHT_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT) >> 2];
%endif
PI_L2 int32_t OUT_L2[CH_IM_OUT] = OUT_INT32;
PI_L1 int32_t OUT_L1[CH_IM_OUT];
%if config.layer.bias == True:
PI_L1 int8_t BIAS_L1[CH_IM_OUT] = BIAS;
%else:
PI_L1 int8_t BIAS_L1[CH_IM_OUT] = {0};
%endif
%elif config.kernel.type == 'linear_quant':
%if config.kernel.in_data_t == 8:
PI_L2 uint8_t IN_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN_INT8;
PI_L1 uint8_t IN_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)];
%elif config.kernel.in_data_t == 4:
PI_L2 uint8_t IN_INT4_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN_INT4;
PI_L2 uint8_t IN_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 1];
PI_L1 uint8_t IN_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 1];
%elif config.kernel.in_data_t == 2:
PI_L2 uint8_t IN_INT2_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN_INT2;
PI_L2 uint8_t IN_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 2];
PI_L1 uint8_t IN_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 2];
%endif
%if config.kernel.wt_data_t == 8:
PI_L2 int8_t WEIGHT_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT)] = WEIGHT_INT8;
PI_L1 int8_t WEIGHT_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT)];
%elif config.kernel.wt_data_t == 4:
PI_L2 int8_t WEIGHT_INT4_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT)] = WEIGHT_INT4;
PI_L2 int8_t WEIGHT_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT) >> 1];
PI_L1 int8_t WEIGHT_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT) >> 1];
%elif config.kernel.wt_data_t == 2:
PI_L2 int8_t WEIGHT_INT2_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT)] = WEIGHT_INT2;
PI_L2 int8_t WEIGHT_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT) >> 2];
PI_L1 int8_t WEIGHT_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT) >> 2];
%endif
%if config.kernel.out_data_t == 8:
PI_L2 uint8_t OUT_INT8_L2[CH_IM_OUT] = OUT_INT8;
PI_L2 uint8_t OUT_L2[CH_IM_OUT];
PI_L1 uint8_t OUT_L1[CH_IM_OUT];
%elif config.kernel.out_data_t == 4:
PI_L2 uint8_t OUT_INT4_L2[CH_IM_OUT] = OUT_INT4;
PI_L2 uint8_t OUT_INT8_L2[CH_IM_OUT >> 1];
PI_L2 uint8_t OUT_L2[CH_IM_OUT >> 1];
PI_L1 uint8_t OUT_L1[CH_IM_OUT >> 1];
%elif config.kernel.out_data_t == 2:
PI_L2 uint8_t OUT_INT2_L2[CH_IM_OUT] = OUT_INT2;
PI_L2 uint8_t OUT_INT8_L2[CH_IM_OUT >> 2];
PI_L2 uint8_t OUT_L2[CH_IM_OUT >> 2];
PI_L1 uint8_t OUT_L1[CH_IM_OUT >> 2];
%endif
%if config.layer.bias == True:
PI_L1 int8_t BIAS_L1[CH_IM_OUT] = BIAS;
%else:
PI_L1 int8_t BIAS_L1[CH_IM_OUT] = {0};
%endif
%elif config.kernel.type == 'maxpool' or config.kernel.type == 'avgpool':
%if config.kernel.in_data_t == 8:
PI_L2 uint8_t IN_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN_INT8;
PI_L1 uint8_t IN_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)];
PI_L2 uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT8;
PI_L2 uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)];
PI_L1 uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)];
%elif config.kernel.in_data_t == 4:
PI_L2 uint8_t IN_INT4_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN_INT4;
PI_L2 uint8_t IN_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 1];
PI_L1 uint8_t IN_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 1];
PI_L2 uint8_t OUT_INT4_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT4;
PI_L2 uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
PI_L2 uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
PI_L1 uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
%elif config.kernel.in_data_t == 2:
PI_L2 uint8_t IN_INT2_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN_INT2;
PI_L2 uint8_t IN_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 2];
PI_L1 uint8_t IN_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 2];
PI_L2 uint8_t OUT_INT2_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT2;
PI_L2 uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
PI_L2 uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
PI_L1 uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
%endif
%endif

%if config.kernel.type == 'add':
%if config.kernel.in_data_t == 8:
PI_L2 uint8_t IN1_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN1_INT8;
PI_L1 uint8_t IN1_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)];
%elif config.kernel.in_data_t == 4:
PI_L2 uint8_t IN1_INT4_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN1_INT4;
PI_L2 uint8_t IN1_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 1];
PI_L1 uint8_t IN1_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 1];
%elif config.kernel.in_data_t == 2:
PI_L2 uint8_t IN1_INT2_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN1_INT2;
PI_L2 uint8_t IN1_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 2];
PI_L1 uint8_t IN1_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 2];
%endif
%if config.kernel.out_data_t == 8:
PI_L2 uint8_t IN2_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN2_INT8;
PI_L1 uint8_t IN2_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)];
%elif config.kernel.out_data_t == 4:
PI_L2 uint8_t IN2_INT4_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN2_INT4;
PI_L2 uint8_t IN2_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 1];
PI_L1 uint8_t IN2_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 1];
%elif config.kernel.out_data_t == 2:
PI_L2 uint8_t IN2_INT2_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN2_INT2;
PI_L2 uint8_t IN2_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 2];
PI_L1 uint8_t IN2_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 2];
%endif
%if config.kernel.in_data_t == 8 and (config.kernel.in_data_t >= config.kernel.out_data_t):
PI_L2 uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT8;
PI_L2 uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)];
PI_L1 uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)];
%elif config.kernel.in_data_t == 4 and (config.kernel.in_data_t >= config.kernel.out_data_t):
PI_L2 uint8_t OUT_INT4_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT4;
PI_L2 uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
PI_L2 uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
PI_L1 uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
%elif config.kernel.in_data_t == 2 and (config.kernel.in_data_t >= config.kernel.out_data_t):
PI_L2 uint8_t OUT_INT2_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT2;
PI_L2 uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
PI_L2 uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
PI_L1 uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
%endif
%endif

#endif
