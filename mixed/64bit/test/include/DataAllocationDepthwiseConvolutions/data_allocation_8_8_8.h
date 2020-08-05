/*
 * data_allocation_8_8_8.h
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

#define DIM_IM_IN_X 8
#define DIM_IM_IN_Y 8
#define CH_IM_IN 128
#define DIM_IM_OUT_X 8
#define DIM_IM_OUT_Y 8
#define CH_IM_OUT 128
#define DIM_KERNEL_X 3
#define DIM_KERNEL_Y 3
#define PADDING_Y_TOP 1
#define PADDING_Y_BOTTOM 1
#define PADDING_X_LEFT 1
#define PADDING_X_RIGHT 1
#define STRIDE_X 1
#define STRIDE_Y 1
#define BIAS_SHIFT 0
#define OUT_MULT 10

RT_L2_DATA int32_t KAPPA_L2[CH_IM_OUT] = KAPPA;
RT_L1_DATA int32_t KAPPA_L1[CH_IM_OUT];
RT_L2_DATA int32_t LAMBDA_L2[CH_IM_OUT] = LAMBDA;
RT_L1_DATA int32_t LAMBDA_L1[CH_IM_OUT];

RT_L2_DATA uint8_t IN_INT8_L2_HWC[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN)] = IN_INT8;
RT_L1_DATA uint8_t IN_INT8_L1_CHW[(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN)];
RT_L2_DATA uint8_t OUT_INT8_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)] = OUT_INT8;
RT_L2_DATA uint8_t OUT_L2[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)];
RT_L1_DATA uint8_t OUT_L1[(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT)];
RT_L2_DATA int8_t WEIGHT_INT8_L2_CHW[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN)] = WEIGHT_INT8;
RT_L1_DATA int8_t WEIGHT_INT8_L1_CHW[(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN)];
RT_L1_DATA uint8_t IM2COL_L1[((DIM_KERNEL_X * (DIM_IM_IN_Y + PADDING_Y_TOP + PADDING_Y_BOTTOM)) + DIM_KERNEL_X) * NUM_CORES];
RT_L1_DATA int8_t WTBUFF_L1[DIM_KERNEL_Y * DIM_KERNEL_X * NUM_CORES];
RT_L1_DATA int8_t BIAS_L1[CH_IM_OUT] = BIAS;

#endif
