/*
 * cifar10_data_allocation.h
 * Angelo Garofalo <angelo.garofalo@unibo.it>
 *
 * Copyright (C) 2019 University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef __CIFAR10_DATA_ALLOCATION__
#define __CIFAR10_DATA_ALLOCATION__
#endif


// include the input and weights

RT_L2_DATA int8_t conv1_wt[CONV1_IN_CH * CONV1_KER_DIM * CONV1_KER_DIM * CONV1_OUT_CH] = CONV1_WT;
RT_L2_DATA int8_t conv1_bias[CONV1_OUT_CH] = CONV1_BIAS;
RT_L2_DATA int8_t out_layer1[CONV1_OUT_CH * POOL1_OUT_DIM* POOL1_OUT_DIM];

RT_L2_DATA int8_t conv2_wt[CONV2_IN_CH * CONV2_KER_DIM * CONV2_KER_DIM * CONV2_OUT_CH] = CONV2_WT;
RT_L2_DATA int8_t conv2_bias[CONV2_OUT_CH] = CONV2_BIAS;
RT_L2_DATA int8_t out_layer2[CONV2_OUT_CH * POOL2_OUT_DIM* POOL2_OUT_DIM];

RT_L2_DATA int8_t conv3_wt[CONV3_IN_CH * CONV3_KER_DIM * CONV3_KER_DIM * CONV3_OUT_CH] = CONV3_WT;
RT_L2_DATA int8_t conv3_bias[CONV3_OUT_CH] = CONV3_BIAS;
RT_L2_DATA int8_t out_layer3[CONV3_OUT_CH * POOL3_OUT_DIM* POOL3_OUT_DIM];

RT_L2_DATA int8_t ip1_wt[IP1_IN_DIM * IP1_OUT_DIM] = IP1_WT;
RT_L2_DATA int8_t ip1_bias[IP1_OUT_DIM] = IP1_BIAS;




/* Here the image_data should be the raw uint8 type RGB image in [RGB, RGB, RGB ... RGB] format */
RT_L2_DATA uint8_t image_data[(CONV1_IN_CH-1) * CONV1_IN_DIM * CONV1_IN_DIM] = IMG_DATA;
RT_L2_DATA int8_t output_data[IP1_OUT_DIM];

//vector buffer: max(im2col buffer,average pool buffer, fully connected buffer)
RT_L1_DATA int8_t  col_buffer[2 *5*5*32*NUM_CORES];
RT_L1_DATA int8_t L1_buffer [44192];
