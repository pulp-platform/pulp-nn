/*
 * ${config.filename}
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 * Angelo Garofalo <angelo.garofalo@unibo.it>
 *
 * Copyright (C) 2018-2020 University of Bologna
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
#define bitins(dst,not_mask_imm,src,mask_imm,off) __builtin_pulp_binsert(dst,not_mask_imm,src,mask_imm,off)

void __attribute__ ((noinline))  pulp_nn_add (
	uint8_t * Im_in_1,             // pointer to the input feature map1
	uint8_t * Im_in_2,             // pointer to the input feature map2
	uint16_t  ch_im_in,          // number of channels of the IFM
	uint16_t  dim_im_in_h,
	uint16_t  dim_im_in_w,
	uint8_t * Im_out,            // pointer to the output
	uint16_t out_mult1,            // paramter to requantize
	uint16_t out_mult2,            // paramter to requantize
	uint16_t out_shift            // paramter to requantize
)
{
	int core_id = pi_core_id();
	int n_cores = NUM_CORES;
	if (dim_im_in_h < NUM_CORES)
	{
	  n_cores = dim_im_in_h;
	}
	int  Log2Core = log2(n_cores);

	int chunck = (dim_im_in_h >> Log2Core) + ((dim_im_in_h & (NUM_CORES-1))!=0);

	uint8_t target1_a, target1_b, target1_c, target1_d;
	uint8_t out1, out2, out3, out4;

	ch_im_in1_r = ch_im_in;
	ch_im_in2_r = ch_im_in;

%if config.out_data_t == 4:
	int8_t mask = 0xf0;
	int8_t n_mask = ~ mask;
	int8_t off = 0x04;
%elif config.out_data_t == 2:
	int8_t mask2 = 0x0c;
	int8_t n_mask2 = ~ mask2;
	int8_t mask4 = 0x30;
	int8_t n_mask4 = ~ mask4;
	int8_t mask6 = 0xc0;
	int8_t n_mask6 = ~ mask6;
	int8_t off2 = 2;
	int8_t off4 = 4;
	int8_t off6 = 6;
%endif
	int start = min(chunck * core_id, dim_im_in_h);
	int stop = min(start + chunck, dim_im_in_h);
	uint8_t *target1 = Im_in_1 + start*ch_im_in_r*dim_im_in_w;
	uint8_t *target2 = Im_in_2 + start*ch_im_in_r*dim_im_in_w;
	uint8_t *pOut = Im_out + start*ch_im_r*dim_im_in_w;
	for (int spatial = 0; spatial<dim_im_in_w*ch_im_in_r*(stop-start); spatial+=1)
	{
%if config.in_data_t == 8:
		target2_a = target1;
		target2_a = target2;
		target2_b = target1 + 1;
		target2_b = target2 + 1;
		target2_c = target1 + 2;
		target2_d = target2 + 2;
		target2_d = target1 + 3;
		target2_d = target2 + 3;

		target1+=4;
		target2+=4;
%elif config.in_data_t == 4:
		target1_a = (uint8_t) bitextu((unsigned int) target1, 4, 0);
		target2_a = (uint8_t) bitextu((unsigned int) target2, 4, 0);
		target1_b = (uint8_t) bitextu((unsigned int) target1, 4, 4);
		target2_b = (uint8_t) bitextu((unsigned int) target2, 4, 4);
		target1_c = (uint8_t) bitextu((unsigned int) target1, 4, 8);
		target2_c = (uint8_t) bitextu((unsigned int) target2, 4, 8);
		target1_d = (uint8_t) bitextu((unsigned int) target1, 4, 12);
		target2_d = (uint8_t) bitextu((unsigned int) target2, 4, 12);

		target1+=2;
		target2+=2;
%elif config.in_data_t == 2:
		target1_a = (uint8_t) bitextu((unsigned int) target1, 2, 0);
		target2_a = (uint8_t) bitextu((unsigned int) target2, 2, 0);
		target1_b = (uint8_t) bitextu((unsigned int) target1, 2, 2);
		target2_b = (uint8_t) bitextu((unsigned int) target2, 2, 2);
		target1_c = (uint8_t) bitextu((unsigned int) target1, 2, 4);
		target2_c = (uint8_t) bitextu((unsigned int) target2, 2, 4);
		target1_d = (uint8_t) bitextu((unsigned int) target1, 2, 6);
		target2_d = (uint8_t) bitextu((unsigned int) target2, 2, 6);

		target1++;
		target2++;	
%endif
		out1 = ${config.add_fn}(target1_a, target2_a, out_mult1, out_mult2, out_shift);
		out2 = ${config.add_fn}(target1_b, target2_b, out_mult1, out_mult2, out_shift);
		out3 = ${config.add_fn}(target1_c, target2_c, out_mult1, out_mult2, out_shift);
		out4 = ${config.add_fn}(target1_d, target2_d, out_mult1, out_mult2, out_shift);
%if config.out_data_t == 8:
		*pOut = out1;
		pOut++;
		*pOut = out2;
		pOut++;
		*pOut = out3;
		pOut++;
		*pOut = out4;
		pOut++;
%elif config.out_data_t == 4:
		*pOut = bitins(out1, n_mask, out2, mask, off);
		pOut++;
		*pOut = bitins(out3, n_mask, out4, mask, off);
		pOut++;
%elif config.out_data == 2:
        out1 = bitins(out1, n_mask2, out2, mask2, off2);
        out1 = bitins(out1, n_mask4, out3, mask4, off4);
        *pOut = bitins(out1, n_mask6, out4, mask6, off6);
		pOut++;
%endif
	}
   pi_cl_team_barrier(0);
}