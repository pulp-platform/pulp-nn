/*
 * pulp_nn_linear_u2_u2_i2.c
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

void pulp_nn_linear_u2_u2_i2(
                  uint8_t *pInBuffer,
                  int8_t *pWeights,
                  uint16_t dim_vec,
                  uint16_t num_o_neurons,
                  int8_t *bias,
                  uint16_t bias_shift,
                  int8_t out_shift,
                  uint16_t out_mult,
                  int64_t *k,
                  int64_t *lambda,
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
	uint16_t dim_vec_in = dim_vec >> 2;
	uint16_t dim_vec_wt = dim_vec >> 2;

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

	v4u vecA[4];
	v4s vecB[4];
	v4s vecB2[4];
	v4s vecB3[4];
	v4s vecB4[4];

	uint8_t *pOut = (uint8_t *) pOutBuffer + (start >> 2);

	int i;
	int64_t *k1 = k + start;
	int64_t *lambda1 = lambda + start;

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

		for (int j=0; j<(dim_vec >> 4); j++)
		{
	      pA = pulp_nn_u2_to_u8(pA,vecA);
		  pB = pulp_nn_i2_to_i8(pB,vecB);
		  pB2 = pulp_nn_i2_to_i8(pB2,vecB2);
		  pB3 = pulp_nn_i2_to_i8(pB3,vecB3);
		  pB4 = pulp_nn_i2_to_i8(pB4,vecB4);
		  sum = SumDotp(vecA[0], vecB[0], sum);
	      sum = SumDotp(vecA[1], vecB[1], sum);
	      sum = SumDotp(vecA[2], vecB[2], sum);
	      sum = SumDotp(vecA[3], vecB[3], sum);
	      sum2 = SumDotp(vecA[0], vecB2[0], sum2);
	      sum2 = SumDotp(vecA[1], vecB2[1], sum2);
	      sum2 = SumDotp(vecA[2], vecB2[2], sum2);
	      sum2 = SumDotp(vecA[3], vecB2[3], sum2);
		  sum3 = SumDotp(vecA[0], vecB3[0], sum3);
		  sum3 = SumDotp(vecA[1], vecB3[1], sum3);
		  sum3 = SumDotp(vecA[2], vecB3[2], sum3);
		  sum3 = SumDotp(vecA[3], vecB3[3], sum3);
		  sum4 = SumDotp(vecA[0], vecB4[0], sum4);
		  sum4 = SumDotp(vecA[1], vecB4[1], sum4);
		  sum4 = SumDotp(vecA[2], vecB4[2], sum4);
		  sum4 = SumDotp(vecA[3], vecB4[3], sum4);
	      //pA+=4;
	      //pB+=4;
	      //pB2+=4;
		  //pB3+=4;
	      //pB4+=4;
		}
    	uint16_t col_cnt = dim_vec & 0xf;
	    while (col_cnt)
	    {
	      uint8_t inA = (uint8_t) bitext((unsigned int) *pA, 2, 0);
	      uint8_t inA2 = (uint8_t) bitext((unsigned int) *pA, 2, 2);
	      uint8_t inA3 = (uint8_t) bitext((unsigned int) *pA, 2, 4);
	      uint8_t inA4 = (uint8_t) bitext((unsigned int) *pA, 2, 6);
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
		  int8_t inB9 = (int8_t) bitext((int) *pB3, 2, 0);
	      int8_t inB10 = (int8_t) bitext((int) *pB3, 2, 2);
	      int8_t inB11 = (int8_t) bitext((int) *pB3, 2, 4);
	      int8_t inB12 = (int8_t) bitext((int) *pB3, 2, 6);
	      pB3++;
		  int8_t inB13 = (int8_t) bitext((int) *pB4, 2, 0);
	      int8_t inB14 = (int8_t) bitext((int) *pB4, 2, 2);
	      int8_t inB15 = (int8_t) bitext((int) *pB4, 2, 4);
	      int8_t inB16 = (int8_t) bitext((int) *pB4, 2, 6);
	      pB4++;
 	  	  sum += inA * inB;
 	  	  sum += inA2 * inB2;
 	  	  sum += inA3 * inB3;
 	  	  sum += inA4 * inB4;
 	  	  sum2 += inA * inB5;
 	  	  sum2 += inA2 * inB6;
 	  	  sum2 += inA3 * inB7;
 	  	  sum2 += inA4 * inB8;
		  sum3 += inA * inB9;
 	  	  sum3 += inA2 * inB10;
 	  	  sum3 += inA3 * inB11;
 	  	  sum3 += inA4 * inB12;
		  sum4 += inA * inB13;
 	  	  sum4 += inA2 * inB14;
 	  	  sum4 += inA3 * inB15;
 	  	  sum4 += inA4 * inB16;
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
