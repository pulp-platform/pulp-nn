/*
 * pulp_nn_linear_u8_u2_i8.c
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
#define clip8(x) __builtin_pulp_clipu_r(x, 3)

void pulp_nn_linear_u8_u2_i8(
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
	int8_t mask2 = 0x0c;
	int8_t n_mask2 = ~ mask2;
	int8_t mask4 = 0x30;
	int8_t n_mask4 = ~ mask4;
	int8_t mask6 = 0xc0;
	int8_t n_mask6 = ~ mask6;
	int8_t off2 = 2;
	int8_t off4 = 4;
	int8_t off6 = 6;
	uint16_t dim_vec_in = dim_vec;
	uint16_t dim_vec_wt = dim_vec;

	int core_id = pi_core_id();
	int Log2Core = log2(NUM_CORES);
	int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);
	int neuron_left = 0;
	if (chunk & 0x3)
	{
		neuron_left = (4 - (chunk & 0x7));
	}
	int start = min((chunk + neuron_left) * core_id, num_o_neurons);
	int stop = min(start + chunk + neuron_left, num_o_neurons);

	v4u vecA;
	v4s vecB;
	v4s vecB2;
	v4s vecB3;
	v4s vecB4;

	uint8_t *pOut = (uint8_t *) pOutBuffer + (start >> 2);

	int i;
	int32_t *k1 = k + start;
	int32_t *lambda1 = lambda + start;

	for(i=start; i<stop; i+=4)
	{
		int sum = 0;
		int sum2 = 0;
		int sum3 = 0;
		int sum4 = 0;

		uint8_t *pA = pInBuffer;
		int8_t *pB = pWeights + (i * dim_vec_wt);
		int8_t *pB2 = pB + dim_vec_wt;
		int8_t *pB3 = pB2 + dim_vec_wt;
		int8_t *pB4 = pB3 + dim_vec_wt;

		for (int j=0; j<(dim_vec >> 2); j++)
		{
		  vecA = *((v4u*)pA);
		  vecB = *((v4s*)pB);
		  vecB2 = *((v4s*)pB2);
		  vecB3 = *((v4s*)pB3);
		  vecB4 = *((v4s*)pB4);
		  sum = SumDotp(vecA, vecB, sum);
		  sum2 = SumDotp(vecA, vecB2, sum2);
		  sum3 = SumDotp(vecA, vecB3, sum3);
		  sum4 = SumDotp(vecA, vecB4, sum4);
	      //pA+=4;
	      //pB+=4;
	      //pB2+=4;
		  //pB3+=4;
	      //pB4+=4;
		}
    	uint16_t col_cnt = dim_vec & 0x3;
	    while (col_cnt)
	    {
	      uint8_t inA = *pA;
	      pA++;
	      int8_t inB = *pB;
	      pB++;
	      int8_t inB2 = *pB2;
	      pB2++;
		  int8_t inB3 = *pB3;
	      pB3++;
	      int8_t inB4 = *pB4;
	      pB4++;
	      sum += inA * inB;
	 	  sum2 += inA * inB2;
	      sum3 += inA * inB3;
	 	  sum4 += inA * inB4;
      	  col_cnt--;
    	}
	    if (flag_batch_norm && flag_relu)
	    {
		  sum = pulp_nn_bn_quant_u2(sum, *k1, *lambda1, out_shift);
		  sum2 = pulp_nn_bn_quant_u2(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
		  sum3 = pulp_nn_bn_quant_u2(sum3, *(k1 + 2), *(lambda1 + 2), out_shift);
	      sum4 = pulp_nn_bn_quant_u2(sum4, *(k1 + 3), *(lambda1 + 3), out_shift);
		  k1+=4;
		  lambda1+=4;
	      sum = bitins(sum, n_mask2, sum2, mask2, off2);
	      sum = bitins(sum, n_mask4, sum3, mask4, off4);
	      *pOut = bitins(sum, n_mask6, sum4, mask6, off6);
	      pOut++;
    	}
	    else
	    {
	      if (flag_relu == 1)
	      {
	        sum = pulp_nn_quant_u2(sum, out_mult, out_shift);
	        sum2 = pulp_nn_quant_u2(sum2, out_mult, out_shift);
	        sum3 = pulp_nn_quant_u2(sum3, out_mult, out_shift);
	        sum4 = pulp_nn_quant_u2(sum4, out_mult, out_shift);
	        sum = bitins(sum, n_mask2, sum2, mask2, off2);
	        sum = bitins(sum, n_mask4, sum3, mask4, off4);
	        *pOut = bitins(sum, n_mask6, sum4, mask6, off6);
	        pOut++;
	      }
	      else
	      {
	        sum = (uint8_t) clip8(sum >> out_shift);
	        sum2 = (uint8_t) clip8(sum2 >> out_shift);
	        sum3 = (uint8_t) clip8(sum3 >> out_shift);
	        sum4 = (uint8_t) clip8(sum4 >> out_shift);
	        sum = bitins(sum, n_mask2, sum2, mask2, off2);
	        sum = bitins(sum, n_mask4, sum3, mask4, off4);
	        *pOut = bitins(sum, n_mask6, sum4, mask6, off6);
	        pOut++;
      	  }
    	}
	}
	pi_cl_team_barrier(0);
}
