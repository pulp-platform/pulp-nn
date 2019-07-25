/*
 * test_data_allocation.h
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


#ifndef __PULP_NN_TEST_DATA_ALLOCATION__
#define __PULP_NN_TEST_DATA_ALLOCATION__
#endif


#ifdef CONV_LAYER_TEST

	#ifdef Q8
	RT_FC_SHARED_DATA int8_t conv_int8_out_L2[(OFM_CH* OFM_W * OFM_H)];
	RT_L1_DATA int8_t input_data_int8_L1[(IFM_W * IFM_H * IFM_CH)];
	RT_L1_DATA int8_t conv_wt_int8_L1[(IFM_CH * DIM_KER * DIM_KER * OFM_CH)];
	RT_L1_DATA int8_t conv_bias_int8_L1[OFM_CH];
	RT_L1_DATA int8_t col_buffer[(2 * NUM_CORES * IFM_CH * DIM_KER  * DIM_KER)];
	RT_L1_DATA int8_t conv_int8_out_L1[(OFM_CH* OFM_W * OFM_H)];
	#endif

#endif

#ifdef FULLY_LAYER_TEST

	#ifdef Q8
	RT_FC_SHARED_DATA int8_t fully_int8_out_L2[(OUT_NEURONS)];
	RT_L1_DATA int8_t input_data_fully_int8_L1[(IFM_H_FC * IFM_H_FC * IFM_CH_FC)];
	RT_L1_DATA int8_t fully_wt_int8_L1[(OUT_NEURONS * IFM_H_FC * IFM_H_FC * IFM_CH_FC)];
	RT_L1_DATA int8_t fully_bias_int8_L1[OUT_NEURONS];
	RT_L1_DATA int8_t fully_int8_out_L1[(OUT_NEURONS)];
	#endif
// to be completed here

#endif


#ifdef MAX_POOLING_TEST

	#ifdef Q8
	RT_L1_DATA int8_t maxpool_data_int8_L1[IFM_CH_MP * IFM_H_MP * IFM_H_MP];
	RT_L1_DATA int8_t maxpool_out_int8_L1[OFM_CH_MP * OFM_H_MP * OFM_H_MP];
	RT_FC_SHARED_DATA int8_t maxpool_out_int8_L2[OFM_CH_MP * OFM_H_MP * OFM_H_MP];
	#endif

#endif


#ifdef RELU_TEST

// to be completed here

#endif
