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

#include "pmsis.h"
#include "pulp_nn_utils.h"
#include "pulp_nn_kernels.h"

#define log2(x) __builtin_pulp_fl1(x)
#define min(a,b) ((a)<(b)?(a):(b))
#define SumDotp(a, b, c) __builtin_pulp_sdotusp4(a, b, c)
%if config.kernel.in_data_t != 8 or config.kernel.out_data_t != 8 or config.kernel.wt_data_t != 8:
#define bitins(dst,not_mask_imm,src,mask_imm,off) __builtin_pulp_binsert(dst,not_mask_imm,src,mask_imm,off)
#define bitext(x,size,off) __builtin_pulp_bextract(x,size,off)
%endif

void ${config.fn_name}(
                  uint8_t *pInBuffer,
                  int8_t *pWeights,
                  uint16_t dim_vec,
                  uint16_t num_o_neurons,
                  int8_t *bias,
                  uint16_t bias_shift,
                  int8_t out_shift,
                  uint16_t out_mult,
%if config.kernel.act_prec == '32bit':
                  int32_t *k,
                  int32_t *lambda,
%elif config.kernel.act_prec == '64bit':
                  int64_t *k,
                  int64_t *lambda,
%endif
%if config.kernel.quantization == 'thresholds':
                  int16_t *pThr,
%endif
                  uint8_t *pOutBuffer,
                  int flag_relu,
                  int flag_batch_norm,
                  unsigned int * memory_chan
)
{
%if config.kernel.in_data_t == 8:
	uint16_t dim_vec_in = dim_vec;
%elif config.kernel.in_data_t == 4:
	uint16_t dim_vec_in = dim_vec >> 1;
%elif config.kernel.in_data_t == 2:
	uint16_t dim_vec_in = dim_vec >> 2;
%endif
%if config.kernel.wt_data_t == 8:
	uint16_t dim_vec_wt = dim_vec;
%elif config.kernel.wt_data_t == 4:
	uint16_t dim_vec_wt = dim_vec >> 1;
%elif config.kernel.wt_data_t == 2:
	uint16_t dim_vec_wt = dim_vec >> 2;
%endif

	int core_id = pi_core_id();
	int Log2Core = log2(NUM_CORES);
	int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);
	int start = min(chunk * core_id, num_o_neurons);
	int stop = min(start + chunk, num_o_neurons);

%if config.less_precision == 8:
	v4u vecA;
	v4s vecB;
	v4s vecB2;
%elif config.less_precision == 4:
	v4u vecA[2];
	v4s vecB[2];
	v4s vecB2[2];
%elif config.less_precision == 2:
	v4u vecA[4];
	v4s vecB[4];
	v4s vecB2[4];
%endif

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

%if config.less_precision == 8:
		for (int j=0; j<(dim_vec >> 2); j++)
%elif config.less_precision == 4:
		for (int j=0; j<(dim_vec >> 3); j++)
%elif config.less_precision == 2:
		for (int j=0; j<(dim_vec >> 4); j++)
%endif
		{
%if config.less_precision == 8:
	           vecA = *((v4u*)pA);
	           vecB = *((v4s*)pB);
	           vecB2 = *((v4s*)pB2);
	  	     sum = SumDotp(vecA, vecB, sum);
	  	     sum2 = SumDotp(vecA, vecB2, sum2);
%elif config.less_precision == 4:
%if config.kernel.in_data_t == 8:
		     vecA[0] = *((v4u*)pA);
		     pA+=4;
		     vecA[1] = *((v4u*)pA);
%else:
	  	     pA = ${config.unpack_in_fn}(pA,vecA);
%endif
%if config.kernel.wt_data_t == 8:
			vecB[0] = *((v4s*)pB);
			vecB2[0] = *((v4s*)pB2);
			pB+=4;
			pB2+=4;
			vecB[1] = *((v4s*)pB);
    	            vecB2[1] = *((v4s*)pB2);
%else:
      	  	pB = ${config.unpack_wt_fn}(pB,vecB);
      	  	pB2 = ${config.unpack_wt_fn}(pB2,vecB2);
%endif
            	sum = SumDotp(vecA[0], vecB[0], sum);
                  sum = SumDotp(vecA[1], vecB[1], sum);
                  sum2 = SumDotp(vecA[0], vecB2[0], sum2);
                  sum2 = SumDotp(vecA[1], vecB2[1], sum2);
%elif config.less_precision == 2:
%if config.kernel.in_data_t == 8:
      	  	vecA[0] = *((v4u*)pA);
      	  	pA+=4;
      	  	vecA[1] = *((v4u*)pA);
      	  	pA+=4;
      	  	vecA[2] = *((v4u*)pA);
      	  	pA+=4;
      	  	vecA[3] = *((v4u*)pA);
%elif config.kernel.in_data_t == 4:
                  pA = ${config.unpack_in_fn}(pA,vecA);
                  //pA+=4;
                  pA = ${config.unpack_in_fn}(pA,vecA + 2);
%elif config.kernel.in_data_t == 2:
                  pA = ${config.unpack_in_fn}(pA,vecA);
%endif
%if config.kernel.wt_data_t == 8:
      	  	vecB[0] = *((v4s*)pB);
      	  	vecB2[0] = *((v4s*)pB2);
      	  	pB+=4;
      	  	pB2+=4;
      	  	vecB[1] = *((v4s*)pB);
                  vecB2[1] = *((v4s*)pB2);
      		pB+=4;
      	  	pB2+=4;
                  vecB[2] = *((v4s*)pB);
      	  	vecB2[2] = *((v4s*)pB2);
      	  	pB+=4;
      	  	pB2+=4;
      	  	vecB[3] = *((v4s*)pB);
                  vecB2[3] = *((v4s*)pB2);
%elif config.kernel.wt_data_t == 4:
	  	      pB = ${config.unpack_wt_fn}(pB,vecB);
                  pB2 = ${config.unpack_wt_fn}(pB2,vecB2);
                  //pB+=4;
            	//pB2+=4;
                  pB = ${config.unpack_wt_fn}(pB,vecB + 2);
                  pB2 = ${config.unpack_wt_fn}(pB2,vecB2 + 2);
%elif config.kernel.wt_data_t == 2:
		 	pB = ${config.unpack_wt_fn}(pB,vecB);
		 	pB2 = ${config.unpack_wt_fn}(pB2,vecB2);
%endif
            	sum = SumDotp(vecA[0], vecB[0], sum);
                  sum = SumDotp(vecA[1], vecB[1], sum);
                  sum = SumDotp(vecA[2], vecB[2], sum);
                  sum = SumDotp(vecA[3], vecB[3], sum);
                  sum2 = SumDotp(vecA[0], vecB2[0], sum2);
                  sum2 = SumDotp(vecA[1], vecB2[1], sum2);
                  sum2 = SumDotp(vecA[2], vecB2[2], sum2);
                  sum2 = SumDotp(vecA[3], vecB2[3], sum2);
%endif
                  //pA+=4;
                  //pB+=4;
                  //pB2+=4;
		}
%if config.less_precision == 2:
            uint16_t col_cnt = dim_vec & 0xf;
%elif config.less_precision == 4:
            uint16_t col_cnt = dim_vec & 0x7;
%elif config.less_precision == 8:
            uint16_t col_cnt = dim_vec & 0x3;
%endif
            while (col_cnt)
            {
%if config.less_precision == 2:
%if config.kernel.in_data_t == 2:
                  uint8_t inA = (uint8_t) bitext((unsigned int) *pA, 2, 0);
                  uint8_t inA2 = (uint8_t) bitext((unsigned int) *pA, 2, 2);
                  uint8_t inA3 = (uint8_t) bitext((unsigned int) *pA, 2, 4);
                  uint8_t inA4 = (uint8_t) bitext((unsigned int) *pA, 2, 6);
                  pA++;
%elif config.kernel.in_data_t == 4:
                  uint8_t inA = (uint8_t) bitext((unsigned int) *pA, 4, 0);
                  uint8_t inA2 = (uint8_t) bitext((unsigned int) *pA, 4, 4);
                  pA++;
                  uint8_t inA3 = (uint8_t) bitext((unsigned int) *pA, 4, 0);
                  uint8_t inA4 = (uint8_t) bitext((unsigned int) *pA, 4, 4);
                  pA++;
%elif config.kernel.in_data_t == 8:
                  uint8_t inA = *pA;
                  pA++;
                  uint8_t inA2 = *pA;
                  pA++;
                  uint8_t inA3 = *pA;
                  pA++;
                  uint8_t inA4 = *pA;
                  pA++;
%endif
%if config.kernel.wt_data_t == 2:
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
%elif config.kernel.wt_data_t == 4:
                  int8_t inB = (int8_t) bitext((int) *pB, 4, 0);
                  int8_t inB2 = (int8_t) bitext((int) *pB, 4, 4);
                  pB++;
                  int8_t inB3 = (int8_t) bitext((int) *pB, 4, 0);
                  int8_t inB4 = (int8_t) bitext((int) *pB, 4, 4);
                  pB++;
                  int8_t inB5 = (int8_t) bitext((int) *pB2, 4, 0);
                  int8_t inB6 = (int8_t) bitext((int) *pB2, 4, 4);
                  pB2++;
                  int8_t inB7 = (int8_t) bitext((int) *pB2, 4, 0);
                  int8_t inB8 = (int8_t) bitext((int) *pB2, 4, 4);
                  pB2++;
%elif config.kernel.wt_data_t == 8:
                  int8_t inB = *pB;
                  pB++;
                  int8_t inB2 = *pB;
                  pB++;
                  int8_t inB3 = *pB;
                  pB++;
                  int8_t inB4 = *pB;
                  pB++;
                  int8_t inB5 = *pB2;
                  pB2++;
                  int8_t inB6 = *pB2;
                  pB2++;
                  int8_t inB7 = *pB2;
                  pB2++;
                  int8_t inB8 = *pB2;
                  pB2++;
%endif
       	  	sum += inA * inB;
       	  	sum += inA2 * inB2;
       	  	sum += inA3 * inB3;
       	  	sum += inA4 * inB4;
       	  	sum2 += inA * inB5;
       	  	sum2 += inA2 * inB6;
       	  	sum2 += inA3 * inB7;
       	  	sum2 += inA4 * inB8;
%elif config.less_precision == 4:
%if config.kernel.in_data_t == 4:
                  uint8_t inA = (uint8_t) bitext((unsigned int) *pA, 4, 0);
                  uint8_t inA2 = (uint8_t) bitext((unsigned int) *pA, 4, 4);
                  pA++;
%elif config.kernel.in_data_t == 8:
                  uint8_t inA = *pA;
                  pA++;
                  uint8_t inA2 = *pA;
                  pA++;
%endif
%if config.kernel.wt_data_t == 4:
                  int8_t inB = (int8_t) bitext((int) *pB, 4, 0);
                  int8_t inB2 = (int8_t) bitext((int) *pB, 4, 4);
                  pB++;
                  int8_t inB5 = (int8_t) bitext((int) *pB2, 4, 0);
                  int8_t inB6 = (int8_t) bitext((int) *pB2, 4, 4);
                  pB2++;
%elif config.kernel.wt_data_t == 8:
                  int8_t inB = *pB;
                  pB++;
                  int8_t inB2 = *pB;
                  pB++;
                  int8_t inB5 = *pB2;
                  pB2++;
                  int8_t inB6 = *pB2;
                  pB2++;
%endif
       	  	sum += inA * inB;
       	  	sum += inA2 * inB2;
       	  	sum2 += inA * inB5;
       	  	sum2 += inA2 * inB6;
%elif config.less_precision == 8:
                  uint8_t inA = *pA;
                  pA++;
                  int8_t inB = *pB;
                  pB++;
                  int8_t inB5 = *pB2;
                  pB2++;
                  sum += inA * inB;
             	sum2 += inA * inB5;
%endif
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

%if config.less_precision == 8:
		for (int j=0; j<(dim_vec >> 2); j++)
%elif config.less_precision == 4:
		for (int j=0; j<(dim_vec >> 3); j++)
%elif config.less_precision == 2:
		for (int j=0; j<(dim_vec >> 4); j++)
%endif
		{
%if config.less_precision == 8:
      	    vecA = *((v4u*)pA);
      	    vecB = *((v4s*)pB);
      	    sum = SumDotp(vecA, vecB, sum);
%elif config.less_precision == 4:
%if config.kernel.in_data_t == 8:
		    vecA[0] = *((v4u*)pA);
		    pA+=4;
		    vecA[1] = *((v4u*)pA);
%else:
		    pA = ${config.unpack_in_fn}(pA,vecA);
%endif
%if config.kernel.wt_data_t == 8:
		    vecB[0] = *((v4s*)pB);
		    pB+=4;
		    vecB[1] = *((v4s*)pB);
%else:
		    pB = ${config.unpack_wt_fn}(pB,vecB);
%endif
		    sum = SumDotp(vecA[0], vecB[0], sum);
	          sum = SumDotp(vecA[1], vecB[1], sum);
%elif config.less_precision == 2:
%if config.kernel.in_data_t == 8:
		    vecA[0] = *((v4u*)pA);
		    pA+=4;
		    vecA[1] = *((v4u*)pA);
		    pA+=4;
		    vecA[2] = *((v4u*)pA);
		    pA+=4;
		    vecA[3] = *((v4u*)pA);
%elif config.kernel.in_data_t == 4:
	          pA = ${config.unpack_in_fn}(pA,vecA);
	          //pA+=4;
	          pA = ${config.unpack_in_fn}(pA,vecA + 2);
%elif config.kernel.in_data_t == 2:
	         pA = ${config.unpack_in_fn}(pA,vecA);
%endif
%if config.kernel.wt_data_t == 8:
		   vecB[0] = *((v4s*)pB);
		   pB+=4;
		   vecB[1] = *((v4s*)pB);
		   pB+=4;
	         vecB[2] = *((v4s*)pB);
		   pB+=4;
		   vecB[3] = *((v4s*)pB);
%elif config.kernel.wt_data_t == 4:
      	   pB = ${config.unpack_wt_fn}(pB,vecB);
      	   //pB+=4;
               pB = ${config.unpack_wt_fn}(pB,vecB + 2);
%elif config.kernel.wt_data_t == 2:
		   pB = ${config.unpack_wt_fn}(pB,vecB);
%endif
		   sum = SumDotp(vecA[0], vecB[0], sum);
	         sum = SumDotp(vecA[1], vecB[1], sum);
	         sum = SumDotp(vecA[2], vecB[2], sum);
	         sum = SumDotp(vecA[3], vecB[3], sum);
%endif
      	   //pA+=4;
      	   //pB+=4;
		}
%if config.less_precision == 2:
            uint16_t col_cnt = dim_vec & 0xf;
%elif config.less_precision == 4:
            uint16_t col_cnt = dim_vec & 0x7;
%elif config.less_precision == 8:
            uint16_t col_cnt = dim_vec & 0x3;
%endif
            while (col_cnt)
            {
%if config.less_precision == 2:
%if config.kernel.in_data_t == 2:
                  uint8_t inA = (uint8_t) bitext((unsigned int) *pA, 2, 0);
                  uint8_t inA2 = (uint8_t) bitext((unsigned int) *pA, 2, 2);
                  uint8_t inA3 = (uint8_t) bitext((unsigned int) *pA, 2, 4);
                  uint8_t inA4 = (uint8_t) bitext((unsigned int) *pA, 2, 6);
                  pA++;
%elif config.kernel.in_data_t == 4:
                  uint8_t inA = (uint8_t) bitext((unsigned int) *pA, 4, 0);
                  uint8_t inA2 = (uint8_t) bitext((unsigned int) *pA, 4, 4);
                  pA++;
                  uint8_t inA3 = (uint8_t) bitext((unsigned int) *pA, 4, 0);
                  uint8_t inA4 = (uint8_t) bitext((unsigned int) *pA, 4, 4);
                  pA++;
%elif config.kernel.in_data_t == 8:
                  uint8_t inA = *pA;
                  pA++;
                  uint8_t inA2 = *pA;
                  pA++;
                  uint8_t inA3 = *pA;
                  pA++;
                  uint8_t inA4 = *pA;
                  pA++;
%endif
%if config.kernel.wt_data_t == 2:
                  int8_t inB = (int8_t) bitext((int) *pB, 2, 0);
                  int8_t inB2 = (int8_t) bitext((int) *pB, 2, 2);
                  int8_t inB3 = (int8_t) bitext((int) *pB, 2, 4);
                  int8_t inB4 = (int8_t) bitext((int) *pB, 2, 6);
                  pB++;
%elif config.kernel.wt_data_t == 4:
                  int8_t inB = (int8_t) bitext((int) *pB, 4, 0);
                  int8_t inB2 = (int8_t) bitext((int) *pB, 4, 4);
                  pB++;
                  int8_t inB3 = (int8_t) bitext((int) *pB, 4, 0);
                  int8_t inB4 = (int8_t) bitext((int) *pB, 4, 4);
                  pB++;
%elif config.kernel.wt_data_t == 8:
                  int8_t inB = *pB;
                  pB++;
                  int8_t inB2 = *pB;
                  pB++;
                  int8_t inB3 = *pB;
                  pB++;
                  int8_t inB4 = *pB;
                  pB++;
%endif
       	  	sum += inA * inB;
       	  	sum += inA2 * inB2;
       	  	sum += inA3 * inB3;
       	  	sum += inA4 * inB4;
%elif config.less_precision == 4:
%if config.kernel.in_data_t == 4:
                  uint8_t inA = (uint8_t) bitext((unsigned int) *pA, 4, 0);
                  uint8_t inA2 = (uint8_t) bitext((unsigned int) *pA, 4, 4);
                  pA++;
%elif config.kernel.in_data_t == 8:
                  uint8_t inA = *pA;
                  pA++;
                  uint8_t inA2 = *pA;
                  pA++;
%endif
%if config.kernel.wt_data_t == 4:
                  int8_t inB = (int8_t) bitext((int) *pB, 4, 0);
                  int8_t inB2 = (int8_t) bitext((int) *pB, 4, 4);
                  pB++;
%elif config.kernel.wt_data_t == 8:
                  int8_t inB = *pB;
                  pB++;
                  int8_t inB2 = *pB;
                  pB++;
%endif
       	  	sum += inA * inB;
       	  	sum += inA2 * inB2;
%elif config.less_precision == 8:
                  uint8_t inA = *pA;
                  pA++;
                  int8_t inB = *pB;
                  pB++;
                  sum += inA * inB;
%endif
                  col_cnt--;
            }
		*pOut = sum;
		pOut++;
	}
	pi_cl_team_barrier(0);
}
