/*
* pulp_nn_matmul_4x2_int8.c
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
#define NN_ROUND(out_shift)         ((out_shift) ? (0x1 << (out_shift -1)) : (0))
#define MIN(a,b)                    ((a)<(b)?(a):(b))
#define CLIP8(x)                    __builtin_pulp_clip(x,-128, 127)



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
int8_t __attribute__ ((noinline)) *pulp_nn_matmul_4x2_int8(
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
		uint16_t chan_left = ch_im_out & 0x3;
		//printf("ch_im_out: %d, numCol_A:%d\n", ch_im_out,numCol_A );

		v4s vecA;
		v4s vecA2;
		v4s vecA3;
		v4s vecA4;
		v4s vecB;
		v4s vecB2;

		/* this loop over the OFM channels */
		for (int i = 0; i < ch_im_out>>2; i++)
		{
			int8_t *pB  =  pInBuffer ;
			int8_t *pB2 = (pB + numCol_A);
			int8_t *pA2 = (pA + numCol_A);
			int8_t *pA3 = (pA2 + numCol_A);
			int8_t *pA4 = (pA3 + numCol_A);

			int bias1 = ((int) (*bias++)  << bias_shift) + NN_ROUND(out_shift);
			int bias2 = ((int) (*bias++)  << bias_shift) + NN_ROUND(out_shift);
			int bias3 = ((int) (*bias++)  << bias_shift) + NN_ROUND(out_shift);
			int bias4 = ((int) (*bias++)  << bias_shift) + NN_ROUND(out_shift);

			/* init the accumulators with corresponding biases */
			int     sum =  bias1;
			int     sum2 = bias2;
			int     sum3 = bias3;
			int     sum4 = bias4;
			int     sum5 = bias1;
			int     sum6 = bias2;
			int     sum7 = bias3;
			int     sum8 = bias4;

			uint16_t  colCnt =numCol_A & 0x3;

			for (int j=0; j < numCol_A >> 2 ; j++)
			{
				vecA  = * ( (v4s*) pA  );
				vecA2 = * ( (v4s*) pA2 );
				vecA3 = * ( (v4s*) pA3 );
				vecA4 = * ( (v4s*) pA4 );
				vecB  = * ( (v4s*) pB  );
				vecB2 = * ( (v4s*) pB2 );

				sum  =  SumDotp (vecA,  vecB,  sum  );
				sum2 =  SumDotp (vecA2, vecB,  sum2 );
				sum3 =  SumDotp (vecA3, vecB,  sum3 );
				sum4 =  SumDotp (vecA4, vecB,  sum4 );

				sum5 =  SumDotp (vecA,  vecB2, sum5 );
				sum6 =  SumDotp (vecA2, vecB2, sum6 );
				sum7 =  SumDotp (vecA3, vecB2, sum7 );
				sum8 =  SumDotp (vecA4, vecB2, sum8 );

				pA  += 4;
				pA2 += 4;
				pA3 += 4;
				pA4 += 4;
				pB  += 4;
				pB2 += 4;
			}

			while(colCnt) //for (int i=0; i< colCnt; i++)
			{

				int8_t      inA  = *pA++;
				int8_t      inA2 = *pA2++;
				int8_t      inA3 = *pA3++;
				int8_t      inA4 = *pA4++;
				int8_t      inB  = *pB++;
				int8_t      inB2 = *pB2++;

				sum  += inA  * inB;
				sum2 += inA2 * inB;
				sum3 += inA3 * inB;
				sum4 += inA4 * inB;
				sum5 +=  inA * inB2;
				sum6 += inA2 * inB2;
				sum7 += inA3 * inB2;
				sum8 += inA4 * inB2;

				colCnt--;
			}

			*pOut  = (int8_t)  CLIP8( sum  >> out_shift);
			pOut++;
			*pOut  = (int8_t)  CLIP8( sum2 >> out_shift);
			pOut++;
			*pOut  = (int8_t)  CLIP8( sum3 >> out_shift);
			pOut++;
			*pOut  = (int8_t)  CLIP8( sum4 >> out_shift);
			pOut++;

			*pOut2 = (int8_t)  CLIP8( sum5 >> out_shift);
			pOut2++;
			*pOut2 = (int8_t)  CLIP8( sum6 >> out_shift);
			pOut2++;
			*pOut2 = (int8_t)  CLIP8( sum7 >> out_shift);
			pOut2++;
			*pOut2 = (int8_t)  CLIP8( sum8 >> out_shift);
			pOut2++;

			pA +=  3 * numCol_A;
		}

		while(chan_left)
		{
			int8_t *pB  =  pInBuffer ;
			int8_t *pB2 = (pB + numCol_A);

			int bias1 = ((int) (*bias++)  << bias_shift) + NN_ROUND(out_shift);
			int     sum  =  bias1;
			int     sum2 =  bias1;

			for (int j=0; j < numCol_A >> 2 ; j++)
			{
				vecA  = * ( (v4s*) pA  );
				vecB  = * ( (v4s*) pB  );
				vecB2 = * ( (v4s*) pB2 );

				sum  =  SumDotp (vecA,  vecB,  sum  );
				sum2 =  SumDotp (vecA,  vecB2, sum2 );

				pA  += 4;
				pB  += 4;
				pB2 += 4;
			}

			uint16_t  colCnt =numCol_A & 0x3;

			while(colCnt) //for (int i=0; i< colCnt; i++)
			{
				int8_t      inA  = *pA++;
				int8_t      inB  = *pB++;
				int8_t      inB2 = *pB2++;

				sum  += inA  * inB;
				sum2 +=  inA * inB2;

				colCnt--;
			}

			*pOut   = (int8_t)  CLIP8( sum  >> out_shift);
			pOut++;
			*pOut2  = (int8_t) CLIP8( sum2  >> out_shift);
			pOut2++;

			chan_left--;
		}
		pOut +=   ch_im_out;
		return pOut;
	}
