/*
 * pulp_nn_avgpool.c
 * Georg Rutishauser <georgr@iis.ee.ethz.ch>
 *
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
#define clip8(x) __builtin_pulp_clipu_r(x, 255)
#define bitext_u(x, size, off) __builtin_pulp_bextractu(x, size, off)
#define mins32(a, b) __builtin_pulp_minsi(a, b)
#define maxs32(a, b) __builtin_pulp_maxsi(a, b)

void __attribute__ ((noinline))  pulp_nn_avgpool(
  uint8_t *  Im_in,
  uint8_t * Im_out,
  int64_t   lambda,
  uint16_t  out_shift,
  int64_t   out_add,
  uint16_t  dim_im_in_x,
  uint16_t  dim_im_in_y,
  uint16_t  ch_im_in,
  uint16_t  dim_im_out_x,
  uint16_t  dim_im_out_y,
  uint16_t  dim_kernel_x,
  uint16_t  dim_kernel_y,
  uint16_t  padding_t,
  uint16_t  padding_b,
  uint16_t  padding_l,
  uint16_t  padding_r,
  uint16_t  stride_x,
  uint16_t  stride_y,
  int       flag_requant
)
{
  /* parallelization */
  int core_id = pi_core_id();
  int n_cores = NUM_CORES;
  if (dim_im_out_y < NUM_CORES)
  {
    n_cores = dim_im_out_y;
  }
  int Log2Core = log2(n_cores);
  int chunck = (dim_im_out_y >> Log2Core) + ((dim_im_out_y & (n_cores -1))!=0);
  int start = chunck * core_id;
  int stop = mins32(start + chunck, dim_im_out_y);
  int   i_x, i_y;



  uint32_t kernel_size_tot = dim_kernel_x * dim_kernel_y;
  int ch_im_in_r = ch_im_in >> 0;
  int ch_im_out_r = ch_im_in >> 0;
  int oc_slice;
  uint32_t sum[1] = {0};
  for (i_y = start; i_y < stop; i_y++)
    {
        for (i_x = 0; i_x < dim_im_out_x; i_x++)
        {
            int k_y_start, k_y_end;
            int k_x_start, k_x_end;

            int32_t out_ch_cnt = 0;
            const int8_t *pTmp, *pTmpInner;
            int8_t *pDst;

            k_y_start = maxs32(0, i_y * stride_y - padding_b);
            k_y_end = mins32(i_y * stride_y - padding_t + dim_kernel_y, dim_im_in_y);

            k_x_start = maxs32(0, i_x * stride_x - padding_l);
            k_x_end = mins32(i_x * stride_x - padding_r + dim_kernel_x, dim_im_in_x);

            pTmp = Im_in;
            pDst = &Im_out[ch_im_out_r * (i_x + i_y * dim_im_out_x)];
            int k_x, k_y;

            for (int ch_cnt = 0; ch_cnt < ch_im_in_r; ch_cnt++)
            {
              sum[0] = 0;
              uint8_t out_el = 0;
                for (k_y = k_y_start; k_y < k_y_end; k_y++)
                {
                    for (k_x = k_x_start; k_x < k_x_end; k_x++)
                    {
                        pTmpInner = pTmp + (ch_im_in_r * (k_x + k_y * dim_im_in_x));
                        uint8_t cur_chans = *pTmpInner;

                        sum[0] += (uint32_t) bitext_u((unsigned int) cur_chans, 8, 0);
                    }
                }
                int64_t out_large;
                if (flag_requant) {
                  out_large = (sum[0] * lambda / kernel_size_tot + out_add) >> out_shift;
                  out_el = clip8(out_large);
                  pDst[(ch_cnt >> (0)) + 0] = out_el;
                  } else {
                  out_large = sum[0] / kernel_size_tot;
                  out_el = clip8(out_large);
                  pDst[(ch_cnt >> (0)) + 0] = out_el;
                }
                pTmp++;
            }
        }
    }
 pi_cl_team_barrier(0);
}
