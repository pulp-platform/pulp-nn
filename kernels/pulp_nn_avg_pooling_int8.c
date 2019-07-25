/*
* pulp_nn_avg_pooling_int8.c
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

#define pulp_avg(a,b)  __builtin_pulp_avg4(a,b);
#define MIN(a,b) 			((a)<(b)?(a):(b))


void __attribute__ ((noinline)) pulp_nn_compare_avg_int8(
  int8_t * base,           // baseline for comparison
  int8_t * target,         // compare target
  uint16_t length          // data size
)
{
  char     *pIn  = base;
  char     *pCom = target;
  v4s inp;
  v4s com;
  uint16_t  cnt = length >> 2;

  while (cnt > 0u)
  {
    inp = *((v4s*)pIn);
    com = *((v4s*)pCom);
    pCom += 4;

    *((v4s*)pIn) = pulp_avg(inp, com);
    pIn += 4;
    cnt--;
  }
}

/**
* @brief INT8 max pooling function
* @param[in, out]  Im_in       pointer to input feature map
* @param[in]       dim_im_in   spatial dimension of the input feature map
* @param[in]       ch_im_in    number of IFM channels
* @param[in]       dim_kernel  spatial dimension of the pooling filter
* @param[in]       padding     amount of padding
* @param[in]       stride      amount of stride
* @param[in]       dim_im_out  reduced spatial dimension of output
* @param[in,out]   bufferA     not used
* @param[in,out]   Im_out      pointer to output feature map
* @return none.
*
* @details
*
* <b>Buffer size:</b>
*
* bufferA size:  0
*
* The pooling function is implemented as split x-pooling then
* y-pooling.
*
* This pooling function is input-destructive. Input data is undefined
* after calling this function.
*
*/

void __attribute__ ((noinline))  pulp_nn_avg_pooling_int8(
  int8_t * Im_in,
  uint16_t dim_im_in,
  uint16_t ch_im_in,
  uint16_t dim_kernel,
  uint16_t padding,
  uint16_t stride,
  uint16_t dim_im_out,
  int8_t * bufferA,
  int8_t * Im_out
)
{
  /* parallelization */
  int core_id = rt_core_id();
  int n_cores = NUM_CORES;
  int Log2Core = __builtin_pulp_fl1(n_cores);
  int chunck = (dim_im_in >> Log2Core) + (dim_im_in & (n_cores -1)!=0);
  int start = chunck * core_id;
  int stop = MIN(start + chunck, dim_im_in);
  int16_t   i_x, i_y;

  /* start kernel: pooling along the x axis */
  for (i_y = start; i_y < stop; i_y++)
  {
    for (i_x = 0; i_x < dim_im_out; i_x++)
    {
      /* define target and the kernel windows */
      int8_t     *target = Im_in + (i_y * dim_im_in + i_x) * ch_im_in;
      int8_t     *win_start;
      int8_t     *win_stop;
      if (i_x * stride - padding < 0)
      {
        win_start = target;
      }
      else
      {
        win_start = Im_in + (i_y * dim_im_in + i_x * stride - padding) * ch_im_in;
      }

      if (i_x * stride - padding + dim_kernel >= dim_im_in)
      {
        win_stop = Im_in + (i_y * dim_im_in + dim_im_in) * ch_im_in;
      }
      else
      {
        win_stop = Im_in + (i_y * dim_im_in + i_x * stride - padding + dim_kernel) * ch_im_in;
      }

      /* copy the data into target */
      for (int i = 0; i< ch_im_in; i++) target[i] = win_start[i];

      /* start the avg operation (comparison) */
      win_start += ch_im_in;
      for (; win_start < win_stop; win_start += ch_im_in)
      {
        pulp_nn_compare_avg_int8(target, win_start, ch_im_in);
      }
    }
  }

  /* synch barrier + parallelization for the second pooling phase */
  rt_team_barrier();
  if (dim_im_out < NUM_CORES)
  n_cores = dim_im_out;
  Log2Core = __builtin_pulp_fl1(n_cores);
  int chunck2 = (dim_im_out >> Log2Core) + (dim_im_out & (n_cores -1)!=0);
  int start2 = chunck2 * core_id;
  int stop2 = MIN(start2 + chunck2, dim_im_out);

  /* pooling along y axis */
  for (i_y = start2; i_y < stop2; i_y++)
  {

    int8_t     *target = Im_out + i_y * dim_im_out * ch_im_in;
    int8_t     *row_start;
    int8_t     *row_end;

    /* define the starting row */
    if (i_y * stride - padding < 0)
    {
      row_start = Im_in;
    }
    else
    {
      row_start = Im_in + (i_y * stride - padding) * dim_im_in * ch_im_in;
    }

    /* define the stopping row */
    if (i_y * stride - padding + dim_kernel >= dim_im_in)
    {
      row_end = Im_in + dim_im_in * dim_im_in * ch_im_in;
    }
    else
    {
      row_end = Im_in + (i_y * stride - padding + dim_kernel) * dim_im_in * ch_im_in;
    }

    /* copy data of the first row*/
    for (int i = 0; i< dim_im_out * ch_im_in; i++) target[i] = row_start[i];

    /* move over to next row */
    row_start += ch_im_in * dim_im_in;

    for (; row_start < row_end; row_start += dim_im_in * ch_im_in)
    {
      pulp_nn_compare_avg_int8(target, row_start, dim_im_out * ch_im_in);
    }
  }

  rt_team_barrier();
}
