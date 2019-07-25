/*
 * pulp_nn_relu_int8.c
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

#define max(x,y) 		__builtin_pulp_max4(x,y)
#define MIN(a,b) 		((a)<(b)?(a):(b))


  /**
   * @brief INT8 RectifiedLinearUnit function
   * @param[in,out]   data        pointer to input
   * @param[in]       dim_im_in   spatial size of the IFM
   * @param[in]       ch_im_in    number of IFM channels
   * @return none.
   */

void pulp_nn_relu_int8(
                  			  int8_t * data,
		              		    uint16_t dim_im_in,
			            	      uint16_t ch_im_in)
{
	int core_id = rt_core_id();
  	int Log2Core = __builtin_pulp_fl1(NUM_CORES );
  	int chunck = (ch_im_in >> Log2Core ) + (ch_im_in & (NUM_CORES-1)!=0);
  	int start = chunck * core_id;
  	int stop = MIN (start + chunck, ch_im_in);

  	int8_t *pOut = data + start * dim_im_in * dim_im_in;
  	int8_t *pIn = data + start * dim_im_in * dim_im_in;

  	v4s in;
  	v4s in2;
  	v4s mask =  (v4s) 0x00000000;

  	for(int i=0; i< (dim_im_in * dim_im_in * chunck)>>3; i++)
    {
    	in = *((v4s*) (pIn));
      	pIn +=4;
      	in2 = *((v4s*) (pIn));
      	*((v4s*) (pOut)) = max(in,mask);
      	*((v4s*) (pIn)) = max(in2,mask);
      	pIn +=4;
      	pOut +=8;
    }
  	rt_team_barrier();
}
