/*
* pulp_nn_linear_int8.c
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


#include "rt/rt_api.h"

#define NN_ROUND(out_shift) 		    ((out_shift) ? (0x1 << (out_shift -1)) : (0))
#define SumDotp(a, b, c)          	__builtin_pulp_sdotsp4(a, b, c)
#define MIN(a,b) 						        ((a)<(b)?(a):(b))
#define CLIP8(x)                    __builtin_pulp_clip(x,-128, 127)

/**
* @brief INT8 linear kernel function
* @param[in]       pIn            	pointer to input tensor
* @param[in]       pWeights         pointer to kernel weights
* @param[in]       dim_vec     		flattened input tensor dimension
* @param[in]       num_o_neurons    number of output neurons
* @param[in]       ch_im_in         number of input tensor channels
* @param[in]       bias             pointer to bias
* @param[in]       bias_shift       amount of shift on bias
* @param[in]       out_shift        amount of shift on output
* @param[in,out]   pOut           	pointer to output tensor
* @return          The function returns either
*/

void pulp_nn_linear_int8(
	int8_t *pIn,
	int8_t *pWeights,
	uint16_t dim_vec,
	uint16_t num_o_neurons,
	uint16_t bias_shift,
	uint16_t out_shift,
	int8_t *bias,
	int8_t *pOut)

	{
		int8_t *pB ;
		int8_t *pB2;
		int8_t *pA;
		int8_t *p0;
		int8_t *pBias;
		uint16_t colCnt = dim_vec & 0x7;

		/* instructions to parallelize the workload:
		each core computes a balanced number of neurons */
		int core_id = rt_core_id();
		int n_cores = NUM_CORES;
		int chunck = 1;
		/* handle the case when number of neurons
		is less than number of cores: chunck=1 */
		if(num_o_neurons < NUM_CORES)
		{
			n_cores = num_o_neurons;
		}
		else
		{
			int Log2Core = __builtin_pulp_fl1(n_cores);
			chunck = (num_o_neurons >> Log2Core) + ((num_o_neurons & (n_cores -1))!=0);
		}
		/* start and stop neuron to be computed, for each core */
		int start = MIN(chunck * core_id,num_o_neurons);
		int stop = MIN(start + chunck, num_o_neurons);
		int a = ((stop-start)>>1);

		/* handle the pointers */
		p0    = pOut + start;
		pBias = bias + start;

		v4s vecA;
		v4s vecB;
		v4s vecB2;

		for(int i = start ; i< ((a<<1)+start); i+=2)
		{

			int bias1 = ((int) (bias[i]) << bias_shift) + NN_ROUND(out_shift);
			int bias2 = ((int) (bias[i+1]) << bias_shift) + NN_ROUND(out_shift);

			int     sum =  bias1;
			int     sum2 = bias2;

			pA = pIn;
			pB = pWeights + i * dim_vec;
			pB2 = pB + dim_vec;

			for (int i = 0; i < dim_vec >> 3; i++)
			{
				vecA  = *((v4s*) pA );
				vecB  = *((v4s*) pB );
				vecB2 = *((v4s*) pB2);

				sum  = SumDotp (vecA, vecB,  sum );
				sum2 = SumDotp (vecA, vecB2, sum2);

				pA  += 4;
				pB  += 4;
				pB2 += 4;

				vecA  = *((v4s*) pA );
				vecB  = *((v4s*) pB );
				vecB2 = *((v4s*) pB2);

				sum  = SumDotp (vecA, vecB,  sum );
				sum2 = SumDotp (vecA, vecB2, sum2);

				pA  += 4;
				pB  += 4;
				pB2 += 4;
			}

			if (colCnt)
			{
				for (int i = 0 ; i< colCnt; i++)
				{
					int8_t inV  = *pA++;
					int8_t inM  = *pB++;
					int8_t inM2 = *pB2++;

					sum  += inV * inM;
					sum2 += inV * inM2;
				}
			}

			*p0++ = (int8_t) CLIP8(sum   >> out_shift);
			*p0++ = (int8_t) CLIP8(sum2  >> out_shift);
		}

		/* handle the loftover computation (for each core) */
		uint16_t rowCnt = ((stop-start)&0x01);
		while(rowCnt)
		{
			int sum = ((int)(*bias++)    << bias_shift)+ NN_ROUND(out_shift);

			for( int i = 0 ; i < dim_vec >> 2; i++)
			{
				vecA = *((v4s*) pA);
				vecB = *((v4s*) pB);

				sum = SumDotp (vecA, vecB, sum);

				pA +=4;
				pB +=4;
			}

			colCnt = dim_vec & 0x3;
			while(colCnt)
			{
				int8_t inV = *pA++;
				int8_t inM = *pB++;

				sum += inV * inM;

				colCnt--;
			}

			*p0++ = (int8_t) CLIP8(sum   >> out_shift);
			rowCnt--;
		}
		rt_team_barrier();
	}
