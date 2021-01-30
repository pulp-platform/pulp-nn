/*
 * pulp_nn_avgpool_u4.c
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

void __attribute__ ((noinline))  pulp_nn_avgpool_u4(
  uint8_t * Im_in,
  uint16_t dim_im_in,
  uint16_t ch_im_in,
  uint16_t dim_kernel,
  uint16_t padding,
  uint16_t stride,
  uint16_t dim_im_out,
  uint8_t * Im_out,
  unsigned int * memory_chan
)
{
  /* parallelization */
  int core_id = pi_core_id();
  int n_cores = NUM_CORES;
  int Log2Core = log2(n_cores);
  int chunck = (dim_im_in >> Log2Core) + (dim_im_in & (n_cores -1)!=0);
  int start = chunck * core_id;
  int stop = min(start + chunck, dim_im_in);
  int   i_x, i_y;
  int ch_im_in_r = ch_im_in >> 1;

  /* start kernel: pooling along the x axis */
  for (i_y = start; i_y < stop; i_y++)
  {
    for (i_x = 0; i_x < dim_im_out; i_x++)
    {
      /* define target and the kernel windows */
      uint8_t     *target = Im_in + (i_y * dim_im_in + i_x) * ch_im_in_r;
      uint8_t     *win_start;
      uint8_t     *win_stop;
      if (i_x * stride - padding < 0)
      {
        win_start = target;
      }
      else
      {
        win_start = Im_in + (i_y * dim_im_in + i_x * stride - padding) * ch_im_in_r;
      }

      if (i_x * stride - padding + dim_kernel >= dim_im_in)
      {
        win_stop = Im_in + (i_y * dim_im_in + dim_im_in) * ch_im_in_r;
      }
      else
      {
        win_stop = Im_in + (i_y * dim_im_in + i_x * stride - padding + dim_kernel) * ch_im_in_r;
      }

      /* copy the data into target */
      for (int i = 0; i< ch_im_in_r; i++) target[i] = win_start[i];

      /* start the avg operation (comparison) */
      win_start += ch_im_in_r;
      for (; win_start < win_stop; win_start += ch_im_in_r)
      {
        pulp_nn_avg_and_replace_u4(target, win_start, ch_im_in_r);
      }
    }
  }

  /* synch barrier + parallelization for the second pooling phase */
  pi_cl_team_barrier(0);
  if (dim_im_out < NUM_CORES)
  n_cores = dim_im_out;
  Log2Core = log2(n_cores);
  int chunck2 = (dim_im_out >> Log2Core) + (dim_im_out & (n_cores -1)!=0);
  int start2 = chunck2 * core_id;
  int stop2 = min(start2 + chunck2, dim_im_out);

  /* pooling along y axis */
  for (i_y = start2; i_y < stop2; i_y++)
  {

    uint8_t     *target = Im_out + i_y * dim_im_out * ch_im_in_r;
    uint8_t     *row_start;
    uint8_t     *row_end;

    /* define the starting row */
    if (i_y * stride - padding < 0)
    {
      row_start = Im_in;
    }
    else
    {
      row_start = Im_in + (i_y * stride - padding) * dim_im_in * ch_im_in_r;
    }

    /* define the stopping row */
    if (i_y * stride - padding + dim_kernel >= dim_im_in)
    {
      row_end = Im_in + dim_im_in * dim_im_in * ch_im_in_r;
    }
    else
    {
      row_end = Im_in + (i_y * stride - padding + dim_kernel) * dim_im_in * ch_im_in_r;
    }

    /* copy data of the first row*/
    for (int i = 0; i< dim_im_out * ch_im_in_r; i++) target[i] = row_start[i];

    /* move over to next row */
    row_start += ch_im_in_r * dim_im_in;

    for (; row_start < row_end; row_start += dim_im_in * ch_im_in_r)
    {
      pulp_nn_avg_and_replace_u4(target, row_start, dim_im_out * ch_im_in_r);
    }
  }

 pi_cl_team_barrier(0);
}