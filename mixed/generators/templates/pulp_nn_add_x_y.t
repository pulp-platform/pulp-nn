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

void __attribute__ ((noinline))  ${config.fn_name}(
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

	uint8_t out1, out2, out3, out4;

%if config.in1_data_t == 8:
	int ch_im_in1_r = ch_im_in;
%elif config.in1_data_t == 4:
	int ch_im_in1_r = ch_im_in >> 1;
%elif config.in1_data_t == 2:
	int ch_im_in1_r = ch_im_in >> 2;
%endif
%if config.in2_data_t == 8:
	int ch_im_in2_r = ch_im_in;
%elif config.in2_data_t == 4:
	int ch_im_in2_r = ch_im_in >> 1;
%elif config.in2_data_t == 2:
	int ch_im_in2_r = ch_im_in >> 2;
%endif

%if config.max_precision == 8:
	int ch_im_out = ch_im_in;

%elif config.max_precision == 4:
	int ch_im_out = ch_im_in >> 1;

	int8_t mask = 0xf0;
	int8_t n_mask = ~ mask;
	int8_t off = 0x04;

%elif config.max_precision == 2:
	int ch_im_out = ch_im_in >> 2;

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

	uint8_t *target1 = Im_in_1 + start * ch_im_in1_r * dim_im_in_w;
	uint8_t *target2 = Im_in_2 + start * ch_im_in2_r * dim_im_in_w;
	uint8_t *pOut = Im_out + start * ch_im_out * dim_im_in_w;

	int a = 0;
	int b = 0;

	uint8_t *target1_ext = &a;
	uint8_t *target2_ext = &b;

%if config.max_precision == 8:
	for (int i=start; i<((stop * ch_im_out * dim_im_in_w) >> 2); i++)
%elif config.max_precision == 4:
	for (int i=start; i<((stop * ch_im_out * dim_im_in_w) >> 1); i++)
%elif config.max_precision == 2:
	for (int i=start; i<(stop * ch_im_out * dim_im_in_w); i++)
%endif
	{
%if config.in1_data_t == 8:
		target1_ext = target1;
		target1+=4;

%elif config.in1_data_t == 4:
		*((v4u*)target1_ext) = ${config.unpack_in1_fn}(target1);
		target1+=2;

%elif config.in1_data_t == 2:
		*((v4u*)target1_ext) = ${config.unpack_in1_fn}(target1);
		target1++;

%endif
%if config.in2_data_t == 8:		
		target2_ext = target2;
		target2+=4;

%elif config.in2_data_t == 4:	
		*((v4u*)target2_ext) = ${config.unpack_in2_fn}(target2);
		target2+=2;

%elif config.in2_data_t == 2:

		*((v4u*)target2_ext) = ${config.unpack_in2_fn}(target2);
		target2++;

%endif
		out1 = ${config.add_fn}(*target1_ext, *target2_ext, out_mult1, out_mult2, out_shift);
		out2 = ${config.add_fn}(*(target1_ext + 1), *(target2_ext + 1), out_mult1, out_mult2, out_shift);
		out3 = ${config.add_fn}(*(target1_ext + 2), *(target2_ext + 2), out_mult1, out_mult2, out_shift);
		out4 = ${config.add_fn}(*(target1_ext + 3), *(target2_ext + 3), out_mult1, out_mult2, out_shift);

%if config.max_precision == 8:
		*pOut = out1;
		pOut++;
		*pOut = out2;
		pOut++;
		*pOut = out3;
		pOut++;
		*pOut = out4;
		pOut++;

%elif config.max_precision == 4:
		*pOut = bitins(out1, n_mask, out2, mask, off);
		pOut++;
		*pOut = bitins(out3, n_mask, out4, mask, off);
		pOut++;
%elif config.max_precision == 2:
        out1 = bitins(out1, n_mask2, out2, mask2, off2);
        out1 = bitins(out1, n_mask4, out3, mask4, off4);
        *pOut = bitins(out1, n_mask6, out4, mask6, off6);
		pOut++;
%endif
	}
   pi_cl_team_barrier(0);
}