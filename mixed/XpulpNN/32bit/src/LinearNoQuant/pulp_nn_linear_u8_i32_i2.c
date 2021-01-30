/*
 * pulp_nn_linear_u8_i32_i2.c
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

#include "pmsis.h"
#include "pulp_nn_utils.h"
#include "pulp_nn_kernels.h"

#define log2(x) __builtin_pulp_fl1(x)
#define min(a,b) ((a)<(b)?(a):(b))
#define SumDotp(a, b, c) __builtin_pulp_sdotusp4(a, b, c)
#define bitins(dst,not_mask_imm,src,mask_imm,off) __builtin_pulp_binsert(dst,not_mask_imm,src,mask_imm,off)
#define bitext(x,size,off) __builtin_pulp_bextract(x,size,off)

void pulp_nn_linear_u8_i32_i2(
                  uint8_t *pInBuffer,
                  int8_t *pWeights,
                  uint16_t dim_vec,
                  uint16_t num_o_neurons,
                  int8_t *bias,
                  uint16_t bias_shift,
                  int8_t out_shift,
                  uint16_t out_mult,
                  int32_t *k,
                  int32_t *lambda,
                  uint8_t *pOutBuffer,
                  int flag_relu,
                  int flag_batch_norm,
                  unsigned int * memory_chan
)
{
	uint16_t dim_vec_in = dim_vec;
	uint16_t dim_vec_wt = dim_vec >> 2;

	int core_id = pi_core_id();
	int Log2Core = log2(NUM_CORES);
	int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);
	int start = min(chunk * core_id, num_o_neurons);
	int stop = min(start + chunk, num_o_neurons);

	v4u vecA[4];
	v4s vecB[4];
	v4s vecB2[4];

	int32_t *pOut = (int32_t *) pOutBuffer + start;

      int lft_neurons = chunk & 0x01;
	int stop_even = stop - lft_neurons;
	int i;

	for(i=start; i<stop_even; i+=2)
	{
		int sum = 0;
		int sum2 = 0;

		uint8_t *pA = pInBuffer;
		int8_t *pB = pWeights + (i * dim_vec_wt);
		int8_t *pB2 = pB + dim_vec_wt;

		for (int j=0; j<(dim_vec >> 4); j++)
		{
      	  	vecA[0] = *((v4u*)pA);
      	  	pA+=4;
      	  	vecA[1] = *((v4u*)pA);
      	  	pA+=4;
      	  	vecA[2] = *((v4u*)pA);
      	  	pA+=4;
      	  	vecA[3] = *((v4u*)pA);
		 	pulp_nn_i2_to_i8(pB,vecB);
		 	pulp_nn_i2_to_i8(pB2,vecB2);
            	sum = SumDotp(vecA[0], vecB[0], sum);
                  sum = SumDotp(vecA[1], vecB[1], sum);
                  sum = SumDotp(vecA[2], vecB[2], sum);
                  sum = SumDotp(vecA[3], vecB[3], sum);
                  sum2 = SumDotp(vecA[0], vecB2[0], sum2);
                  sum2 = SumDotp(vecA[1], vecB2[1], sum2);
                  sum2 = SumDotp(vecA[2], vecB2[2], sum2);
                  sum2 = SumDotp(vecA[3], vecB2[3], sum2);
                  pA+=4;
                  pB+=4;
                  pB2+=4;
		}
            uint16_t col_cnt = dim_vec & 0xf;
            while (col_cnt)
            {
                  uint8_t inA = *pA;
                  pA++;
                  uint8_t inA2 = *pA;
                  pA++;
                  uint8_t inA3 = *pA;
                  pA++;
                  uint8_t inA4 = *pA;
                  pA++;
                  int8_t inB = (int8_t) bitext((int) *pB, 2, 0);
                  int8_t inB2 = (int8_t) bitext((int) *pB, 2, 2);
                  int8_t inB3 = (int8_t) bitext((int) *pB, 2, 4);
                  int8_t inB4 = (int8_t) bitext((int) *pB, 2, 6);
                  pB++;
                  int8_t inB5 = (int8_t) bitext((int) *pB2, 2, 0);
                  int8_t inB6 = (int8_t) bitext((int) *pB2, 2, 2);
                  int8_t inB7 = (int8_t) bitext((int) *pB2, 2, 4);
                  int8_t inB8 = (int8_t) bitext((int) *pB2, 2, 6);
                  pB2++;
       	  	sum += inA * inB;
       	  	sum += inA2 * inB2;
       	  	sum += inA3 * inB3;
       	  	sum += inA4 * inB4;
       	  	sum2 += inA * inB5;
       	  	sum2 += inA2 * inB6;
       	  	sum2 += inA3 * inB7;
       	  	sum2 += inA4 * inB8;
                  col_cnt--;
            }
		*pOut = sum;
		pOut++;
		*pOut = sum2;
		pOut++;
	}
	if (lft_neurons && (stop - start) > 0)
	{
		int sum = 0;

		uint8_t *pA = pInBuffer;
		int8_t *pB = pWeights + (i * dim_vec_wt);

		for (int j=0; j<(dim_vec >> 4); j++)
		{
		    vecA[0] = *((v4u*)pA);
		    pA+=4;
		    vecA[1] = *((v4u*)pA);
		    pA+=4;
		    vecA[2] = *((v4u*)pA);
		    pA+=4;
		    vecA[3] = *((v4u*)pA);
		   pulp_nn_i2_to_i8(pB,vecB);
		   sum = SumDotp(vecA[0], vecB[0], sum);
	         sum = SumDotp(vecA[1], vecB[1], sum);
	         sum = SumDotp(vecA[2], vecB[2], sum);
	         sum = SumDotp(vecA[3], vecB[3], sum);
      	   pA+=4;
      	   pB+=4;
		}
            uint16_t col_cnt = dim_vec & 0xf;
            while (col_cnt)
            {
                  uint8_t inA = *pA;
                  pA++;
                  uint8_t inA2 = *pA;
                  pA++;
                  uint8_t inA3 = *pA;
                  pA++;
                  uint8_t inA4 = *pA;
                  pA++;
                  int8_t inB = (int8_t) bitext((int) *pB, 2, 0);
                  int8_t inB2 = (int8_t) bitext((int) *pB, 2, 2);
                  int8_t inB3 = (int8_t) bitext((int) *pB, 2, 4);
                  int8_t inB4 = (int8_t) bitext((int) *pB, 2, 6);
                  pB++;
       	  	sum += inA * inB;
       	  	sum += inA2 * inB2;
       	  	sum += inA3 * inB3;
       	  	sum += inA4 * inB4;
                  col_cnt--;
            }
		*pOut = sum;
		pOut++;
	}
	pi_cl_team_barrier(0);
}
