/*
* pulp_nn_matmul_2x2_int8.c
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

#define SumDotp(a, b, c)            __builtin_pulp_sdotsp4(a, b, c)
#define NN_ROUND(out_shift)        ((out_shift) ? (0x1 << (out_shift -1)) : (0))
#define MIN(a,b)                   ((a)<(b)?(a):(b))
#define CLIP8(x)                   __builtin_pulp_clip(x,-128, 127)


/**
* @brief INT8 Matrix Multiplication function for convolution
* @param[in]       pWeights         pointer to weights (Operand A of MatMul)
* @param[in]       pInBuffer        pointer to the input columns buffer (Operand B of MatMul)
* @param[in]       ch_im_out        number of output channels (number of rows of Operand A)
* @param[in]       numCol_A         receptive field dimension (number of columns of Operand B)
* @param[in]       bias_shift       amount of shift on bias
* @param[in]       out_shift        amount of shift on output
* @param[in]       bias             pointer to the bias
* @param[in,out]   pOut             pointer to the output
* @return          The function returns the incremented output pointer
*/
int8_t __attribute__ ((noinline)) *pulp_nn_matmul_2x2_int8(
	int8_t * pWeight,
	int8_t * pInBuffer,
	uint16_t ch_im_out,
	uint16_t numCol_A,
	uint16_t bias_shift,
	uint16_t out_shift,
	int8_t * bias,
	int8_t * pOut)
	{
		int8_t  *pOut2 = pOut + ch_im_out;
		int8_t  *pA = pWeight;
		uint16_t  colCnt =numCol_A & 0x3;

		v4s vecA;
		v4s vecA2;
		v4s vecB;
		v4s vecB2;
		v4s vecB3;


		/* this loop over the OFM channels */
		for (int i = 0; i < ch_im_out; i+=2)
		{
			int8_t *pB =  pInBuffer ;
			int8_t *pB2 = (pB + numCol_A);
			int8_t *pA2 = (pA + numCol_A);

			/* init the accumulators with corresponding biases */
			int     bias1  =  ((int) (bias[i])    << bias_shift) + NN_ROUND(out_shift);
			int     bias2  =  ((int) (bias[i +1]) << bias_shift) + NN_ROUND(out_shift);

			int sum  = bias1;
			int sum2 = bias1;
			int sum3 = bias2;
			int sum4 = bias2;

			for (int j=0; j < numCol_A >> 2 ; j++)
			{
				vecA  = * ( (v4s*) pA  );
				vecA2 = * ( (v4s*) pA2 );
				vecB  = * ( (v4s*) pB  );
				vecB2 = * ( (v4s*) pB2 );

				sum  =  SumDotp (vecA,  vecB,  sum  );
				sum2 =  SumDotp (vecA,  vecB2, sum2 );
				sum3 =  SumDotp (vecA2, vecB,  sum3 );
				sum4 =  SumDotp (vecA2, vecB2, sum4 );


				pA  += 4;
				pB  += 4;
				pB2 += 4;
				pA2 += 4;
			}
			if(colCnt)
			{
				for (int i=0; i< colCnt; i++)
				{
					int8_t      inA = *pA++;
					int8_t      inB = *pB++;
					int8_t      inA2 = *pA2++;
					int8_t      inB2 = *pB2++;

					sum +=  inA * inB;
					sum2 += inA * inB2;
					sum3 += inA2 * inB;
					sum4 += inA2 * inB2;
				}
			}

			*pOut = (int8_t)  CLIP8((sum  >> out_shift));
			pOut++;
			*pOut  = (int8_t) CLIP8((sum3 >> out_shift));
			pOut++;
			*pOut2 = (int8_t) CLIP8((sum2 >> out_shift));
			pOut2++;
			*pOut2 = (int8_t) CLIP8((sum4 >> out_shift));
			pOut2++;

			pA +=  numCol_A; //skip one

		}
		pOut +=  ch_im_out;
		return pOut;
	}
