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

#ifndef __DATA_ALLOCATION__
#define __DATA_ALLOCATION__

#define DIM_IM_IN_X ${config.dim_im_in_x}
#define DIM_IM_IN_Y ${config.dim_im_in_y}
#define CH_IM_IN ${config.ch_im_in}
#define DIM_IM_OUT_X ${config.dim_im_out_x}
#define DIM_IM_OUT_Y ${config.dim_im_out_y}
#define CH_IM_OUT ${config.ch_im_out}
%if config.api != 'PULPNNDataAllocationNoQuant':
#define DIM_KERNEL_X ${config.dim_ker_x}
#define DIM_KERNEL_Y ${config.dim_ker_y}
#define PADDING_Y_TOP ${config.pad_y_top}
#define PADDING_Y_BOTTOM ${config.pad_y_bot}
#define PADDING_X_LEFT ${config.pad_x_left}
#define PADDING_X_RIGHT ${config.pad_x_right}
#define STRIDE_X ${config.stride_x}
#define STRIDE_Y ${config.stride_y}
%endif
#define BIAS_SHIFT ${config.bias_shift}
%if config.out_data_t == 8 or config.quantization != "thresholds":
#define OUT_MULT ${config.out_mult}

RT_L2_DATA int32_t KAPPA_L2[CH_IM_OUT] = KAPPA;
RT_L1_DATA int32_t KAPPA_L1[CH_IM_OUT];
RT_L2_DATA int32_t LAMBDA_L2[CH_IM_OUT] = LAMBDA;
RT_L1_DATA int32_t LAMBDA_L1[CH_IM_OUT];
%elif config.out_data_t == 2:
RT_L2_DATA int16_t THR_INT2_L2[CH_IM_OUT << 2] = THR_INT2;
RT_L1_DATA int16_t THR_INT2_L1[CH_IM_OUT << 2];
%elif config.out_data_t == 4:
RT_L2_DATA int16_t THR_INT4_L2[CH_IM_OUT << 4] = THR_INT4;
RT_L1_DATA int16_t THR_INT4_L1[CH_IM_OUT << 4];
%endif

%if config.type == 'pointwise':
	%if config.in_data_t == 2:
RT_L2_DATA uint8_t IN_INT2_L2[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN)] = IN_INT2;
RT_L1_DATA uint8_t IN_INT8_L1[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 2];
	%elif config.in_data_t == 4:
RT_L2_DATA uint8_t IN_INT4_L2[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN)] = IN_INT4;
RT_L1_DATA uint8_t IN_INT8_L1[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 1];
	%else:
RT_L2_DATA uint8_t IN_INT8_L2[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN)] = IN_INT8;
RT_L1_DATA uint8_t IN_INT8_L1[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN)];
	%endif
	%if config.out_data_t == 2:
RT_L2_DATA uint8_t OUT_INT2_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT2;
RT_L2_DATA uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
RT_L2_DATA uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
RT_L1_DATA uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
	%elif config.out_data_t == 4:
RT_L2_DATA uint8_t OUT_INT4_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT4;
RT_L2_DATA uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
RT_L2_DATA uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
RT_L1_DATA uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
	%else:
RT_L2_DATA uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT8;
RT_L2_DATA uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)];
RT_L1_DATA uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)];
	%endif
	%if config.wt_data_t == 2:
RT_L2_DATA int8_t WEIGHT_INT2_L2[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT)] = WEIGHT_INT2;
RT_L1_DATA int8_t WEIGHT_INT8_L1[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT) >> 2];
	%elif config.wt_data_t == 4:
RT_L2_DATA int8_t WEIGHT_INT4_L2[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT)] = WEIGHT_INT4;
RT_L1_DATA int8_t WEIGHT_INT8_L1[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT) >> 1];
	%elif config.wt_data_t == 8:
RT_L2_DATA int8_t WEIGHT_INT8_L2[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT)] = WEIGHT_INT8;
RT_L1_DATA int8_t WEIGHT_INT8_L1[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT)];
	%endif
RT_L1_DATA uint8_t IM2COL_L1[((CH_IM_IN * DIM_KERNEL_X * DIM_KERNEL_Y) << 1) * NUM_CORES];
RT_L1_DATA int8_t BIAS_L1[CH_IM_OUT] = BIAS;
%elif config.type == 'depthwise':
%if config.in_data_t == 2:
RT_L2_DATA uint8_t IN_INT2_L2_HWC[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN)] = IN_INT2;
RT_L2_DATA uint8_t IN_INT8_L2_HWC[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 2];
RT_L1_DATA uint8_t IN_INT8_L1_CHW[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 2];
%elif config.in_data_t == 4:
RT_L2_DATA uint8_t IN_INT4_L2_HWC[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN)] = IN_INT4;
RT_L2_DATA uint8_t IN_INT8_L2_HWC[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 1];
RT_L1_DATA uint8_t IN_INT8_L1_CHW[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 1];
%else:
RT_L2_DATA uint8_t IN_INT8_L2_HWC[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN)] = IN_INT8;
RT_L1_DATA uint8_t IN_INT8_L1_CHW[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN)];
%endif
%if config.out_data_t == 2:
RT_L2_DATA uint8_t OUT_INT2_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT2;
RT_L2_DATA uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
RT_L2_DATA uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
RT_L1_DATA uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2];
%elif config.out_data_t == 4:
RT_L2_DATA uint8_t OUT_INT4_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT4;
RT_L2_DATA uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
RT_L2_DATA uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
RT_L1_DATA uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1];
%else:
RT_L2_DATA uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT8;
RT_L2_DATA uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)];
RT_L1_DATA uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)];
%endif
%if config.wt_data_t == 2:
RT_L2_DATA int8_t WEIGHT_INT2_L2_CHW[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN)] = WEIGHT_INT2;
RT_L2_DATA int8_t WEIGHT_INT2_L2_HWC[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN)];
RT_L2_DATA int8_t WEIGHT_INT8_L2_HWC[((DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN) >> 2)];
RT_L1_DATA int8_t WEIGHT_INT8_L1_CHW[((DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN) >> 2)];
%elif config.wt_data_t == 4:
RT_L2_DATA int8_t WEIGHT_INT4_L2_CHW[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN)] = WEIGHT_INT4;
RT_L2_DATA int8_t WEIGHT_INT4_L2_HWC[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN)];
RT_L2_DATA int8_t WEIGHT_INT8_L2_HWC[((DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN) >> 1)];
RT_L1_DATA int8_t WEIGHT_INT8_L1_CHW[((DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN) >> 1)];
%elif config.wt_data_t == 8:
RT_L2_DATA int8_t WEIGHT_INT8_L2_CHW[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN)] = WEIGHT_INT8;
RT_L1_DATA int8_t WEIGHT_INT8_L1_CHW[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN)];
%endif
%if config.less_precision == 2:
RT_L1_DATA uint8_t IM2COL_L1[(((DIM_KERNEL_X * (DIM_IM_IN_Y + PADDING_Y_TOP + PADDING_Y_BOTTOM)) + DIM_KERNEL_X) << 2) * NUM_CORES];
RT_L1_DATA int8_t WTBUFF_L1[((DIM_KERNEL_Y * DIM_KERNEL_X) << 2) * NUM_CORES];
%elif config.less_precision == 4:
RT_L1_DATA uint8_t IM2COL_L1[(((DIM_KERNEL_X * (DIM_IM_IN_Y + PADDING_Y_TOP + PADDING_Y_BOTTOM)) + DIM_KERNEL_X) << 1) * NUM_CORES];
RT_L1_DATA int8_t WTBUFF_L1[((DIM_KERNEL_Y * DIM_KERNEL_X) << 1) * NUM_CORES];
%elif config.less_precision == 8:
RT_L1_DATA uint8_t IM2COL_L1[((DIM_KERNEL_X * (DIM_IM_IN_Y + PADDING_Y_TOP + PADDING_Y_BOTTOM)) + DIM_KERNEL_X) * NUM_CORES];
RT_L1_DATA int8_t WTBUFF_L1[DIM_KERNEL_Y * DIM_KERNEL_X * NUM_CORES];
%endif
RT_L1_DATA int8_t BIAS_L1[CH_IM_OUT] = BIAS;
%elif config.type == 'linear_no_quant':
%if config.in_data_t == 8:
RT_L2_DATA uint8_t IN_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN_INT8;
RT_L1_DATA uint8_t IN_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)];
%elif config.in_data_t == 4:
RT_L2_DATA uint8_t IN_INT4_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN_INT4;
RT_L2_DATA uint8_t IN_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 1];
RT_L1_DATA uint8_t IN_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 1];
%elif config.in_data_t == 2:
RT_L2_DATA uint8_t IN_INT2_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN_INT2;
RT_L2_DATA uint8_t IN_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 2];
RT_L1_DATA uint8_t IN_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 2];
%endif
%if config.wt_data_t == 8:
RT_L2_DATA int8_t WEIGHT_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT)] = WEIGHT_INT8;
RT_L1_DATA int8_t WEIGHT_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT)];
%elif config.wt_data_t == 4:
RT_L2_DATA int8_t WEIGHT_INT4_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT)] = WEIGHT_INT4;
RT_L2_DATA int8_t WEIGHT_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT) >> 1];
RT_L1_DATA int8_t WEIGHT_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT) >> 1];
%elif config.wt_data_t == 2:
RT_L2_DATA int8_t WEIGHT_INT2_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT)] = WEIGHT_INT2;
RT_L2_DATA int8_t WEIGHT_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT) >> 2];
RT_L1_DATA int8_t WEIGHT_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT) >> 2];
%endif
RT_L2_DATA int32_t OUT_L2[CH_IM_OUT] = OUT_INT32;
RT_L1_DATA int32_t OUT_L1[CH_IM_OUT];
%elif config.type == 'linear_quant':
%if config.in_data_t == 8:
RT_L2_DATA uint8_t IN_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN_INT8;
RT_L1_DATA uint8_t IN_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)];
%elif config.in_data_t == 4:
RT_L2_DATA uint8_t IN_INT4_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN_INT4;
RT_L2_DATA uint8_t IN_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 1];
RT_L1_DATA uint8_t IN_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 1];
%elif config.in_data_t == 2:
RT_L2_DATA uint8_t IN_INT2_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y)] = IN_INT2;
RT_L2_DATA uint8_t IN_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 2];
RT_L1_DATA uint8_t IN_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y) >> 2];
%endif
%if config.wt_data_t == 8:
RT_L2_DATA int8_t WEIGHT_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT)] = WEIGHT_INT8;
RT_L1_DATA int8_t WEIGHT_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT)];
%elif config.wt_data_t == 4:
RT_L2_DATA int8_t WEIGHT_INT4_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT)] = WEIGHT_INT4;
RT_L2_DATA int8_t WEIGHT_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT) >> 1];
RT_L1_DATA int8_t WEIGHT_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT) >> 1];
%elif config.wt_data_t == 2:
RT_L2_DATA int8_t WEIGHT_INT2_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT)] = WEIGHT_INT2;
RT_L2_DATA int8_t WEIGHT_INT8_L2[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT) >> 2];
RT_L1_DATA int8_t WEIGHT_INT8_L1[(CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT) >> 2];
%endif
%if config.out_data_t == 8:
RT_L2_DATA uint8_t OUT_INT8_L2[CH_IM_OUT] = OUT_INT8;
RT_L2_DATA uint8_t OUT_L2[CH_IM_OUT];
RT_L1_DATA uint8_t OUT_L1[CH_IM_OUT];
%elif config.out_data_t == 4:
RT_L2_DATA uint8_t OUT_INT4_L2[CH_IM_OUT] = OUT_INT4;
RT_L2_DATA uint8_t OUT_INT8_L2[CH_IM_OUT >> 1];
RT_L2_DATA uint8_t OUT_L2[CH_IM_OUT >> 1];
RT_L1_DATA uint8_t OUT_L1[CH_IM_OUT >> 1];
%elif config.out_data_t == 2:
RT_L2_DATA uint8_t OUT_INT2_L2[CH_IM_OUT] = OUT_INT2;
RT_L2_DATA uint8_t OUT_INT8_L2[CH_IM_OUT >> 2];
RT_L2_DATA uint8_t OUT_L2[CH_IM_OUT >> 2];
RT_L1_DATA uint8_t OUT_L1[CH_IM_OUT >> 2];
%endif
%endif

#endif
