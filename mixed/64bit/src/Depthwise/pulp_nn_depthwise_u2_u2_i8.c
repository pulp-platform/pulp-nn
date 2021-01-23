/*
 * pulp_nn_depthwise_u2_u2_i8.c
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
#define nn_round(out_shift) (0x1 << (out_shift -1))
#define bitins(dst,not_mask_imm,src,mask_imm,off) __builtin_pulp_binsert(dst,not_mask_imm,src,mask_imm,off)
#define bitext(x,size,off) __builtin_pulp_bextract(x,size,off)
#define bitextu(x,size,off) __builtin_pulp_bextractu(x,size,off)
#define clip8(x) __builtin_pulp_clipu(x, 0, 3)

void pulp_nn_depthwise_u2_u2_i8(
        const uint8_t * pInBuffer,
        const uint16_t dim_in_x,
        const uint16_t dim_in_y,
        const uint16_t ch_in,
        const int8_t * pWeightBuffer,
        const uint16_t ch_out,
        const uint16_t dim_kernel_x,
        const uint16_t dim_kernel_y,
        const uint16_t padding_y_top,
        const uint16_t padding_y_bottom,
        const uint16_t padding_x_left,
        const uint16_t padding_x_right,
        const uint16_t stride_x,
        const uint16_t stride_y,
        const int8_t * bias,
        const uint16_t bias_shift,
        const int8_t out_shift,
        const uint16_t out_mult,
        uint8_t * pOutBuffer,
        const uint16_t dim_out_x,
        const uint16_t dim_out_y,
        int64_t * k,
        int64_t * lambda,
        uint8_t * pIm2ColBuffer,
        int8_t * pWtBuffer,
        int flag_relu,
        int flag_batch_norm,
        unsigned int * memory_chan
) {
  uint8_t core_id = pi_core_id();
  uint8_t Log2Core = log2(NUM_CORES);

  uint16_t ch_out_r = ch_out >> 2;
  uint16_t ch_in_r = ch_out >> 2;
  uint16_t ch_wt_r = ch_out;

  uint16_t ch_min = ch_out >> 2;

  int chunk = (ch_min >> Log2Core) + ((ch_min & (NUM_CORES - 1)) != 0);

  int start_channel = min(chunk * core_id, ch_min);
  int stop_channel = min(start_channel + chunk, ch_min);

  uint16_t dim_kernel_x_size_rem = dim_kernel_x & 0x3;
  uint16_t dim_kernel_x_size_padded = (dim_kernel_x >> 2) + (dim_kernel_x_size_rem != 0);
  uint16_t dim_incr = (dim_kernel_x_size_padded << 2) - dim_kernel_x;
  uint16_t dim_incr_pad_left = (dim_kernel_x_size_padded << 2) - dim_kernel_x + padding_x_left;
  uint16_t dim_incr_pad_right = (dim_kernel_x_size_padded << 2) - dim_kernel_x + 1;
  uint16_t kernel_size = dim_kernel_x * dim_kernel_y;
  uint16_t im2col_size = ((dim_kernel_x * (dim_in_y + padding_y_top + padding_y_bottom)) + dim_kernel_x);
  uint16_t in_image_size = dim_in_x * dim_in_y;

  uint8_t * pIm2ColBase = pIm2ColBuffer + (core_id * im2col_size << 2);
  int8_t * pWtBase = pWtBuffer + (core_id * (kernel_size << 2));

  int i_out_x, i_buff_y;
  uint16_t colCnt = kernel_size >> 2;
  uint16_t leftCnt = kernel_size & 0x3;

  int8_t mask2 = 0x0c;
  int8_t n_mask2 = ~ mask2;
  int8_t mask4 = 0x30;
  int8_t n_mask4 = ~ mask4;
  int8_t mask6 = 0xc0;
  int8_t n_mask6 = ~ mask6;
  int8_t off2 = 2;
  int8_t off4 = 4;
  int8_t off6 = 6;

  int i_out_ch = start_channel;
  int i_in_ch = start_channel * in_image_size;
  int i_wt_ch = (start_channel << 2) * kernel_size;

  int64_t *k1 = k + core_id * (chunk << 2);
  int64_t *lambda1 = lambda + core_id * (chunk << 2);

  for(int i_ch = start_channel; i_ch < stop_channel; i_ch++)
  {
    i_out_x = 0;
    int8_t * pWt = pWeightBuffer + i_wt_ch;
    int8_t * pWt2 = pWt + kernel_size;
    int8_t * pWt3 = pWt2 + kernel_size;
    int8_t * pWt4 = pWt3 + kernel_size;
    if(padding_x_left > 0)
    {
      do
      {
        uint8_t *pOut = pOutBuffer + i_out_ch + (i_out_x * ch_out_r);
        uint8_t *pIm2Col = pIm2ColBase;
        uint8_t *pIm2Col2 = pIm2Col + im2col_size;
        uint8_t *pIm2Col3 = pIm2Col2 + im2col_size;
        uint8_t *pIm2Col4 = pIm2Col3 + im2col_size;
        i_buff_y = - padding_y_top;
        if(padding_y_top > 0)
        {
          do
          {
            int i=0;
            do
            {
              *(v4u *) pIm2Col = (v4u) {0, 0, 0, 0};
              pIm2Col+=4;
              *(v4u *) pIm2Col2 = (v4u) {0, 0, 0, 0};
              pIm2Col2+=4;
              *(v4u *) pIm2Col3 = (v4u) {0, 0, 0, 0};
              pIm2Col3+=4;
              *(v4u *) pIm2Col4 = (v4u) {0, 0, 0, 0};
              pIm2Col4+=4;
              i++;
            }while(i<dim_kernel_x_size_padded);
            pIm2Col-=dim_incr;
            pIm2Col2-=dim_incr;
            pIm2Col3-=dim_incr;
            pIm2Col4-=dim_incr;
            i_buff_y++;
          }while(i_buff_y < 0);
        }
        int const1 = (i_out_x * stride_x);
        int base_ptr = pInBuffer + i_in_ch;
        do
        {
          for(int j=0; j< (padding_x_left - const1); j++)
          {
            *(uint8_t *) pIm2Col = 0;
            pIm2Col++;
            *(uint8_t *) pIm2Col2 = 0;
            pIm2Col2++;
            *(uint8_t *) pIm2Col3 = 0;
            pIm2Col3++;
            *(uint8_t *) pIm2Col4 = 0;
            pIm2Col4++;
          }
          int idx = 0;
          int i = 0;
          do
          {
            v4u src_in = *((v4u*) (base_ptr + idx));
            *pIm2Col = (uint8_t) bitextu((unsigned int) src_in, 2, 0);
            *pIm2Col2 = (uint8_t) bitextu((unsigned int) src_in, 2, 2);
            *pIm2Col3 = (uint8_t) bitextu((unsigned int) src_in, 2, 4);
            *pIm2Col4 = (uint8_t) bitextu((unsigned int) src_in, 2, 6);
            pIm2Col++;
            pIm2Col2++;
            pIm2Col3++;
            pIm2Col4++;
            *pIm2Col = (uint8_t) bitextu((unsigned int) src_in, 2, 8);
            *pIm2Col2 = (uint8_t) bitextu((unsigned int) src_in, 2, 10);
            *pIm2Col3 = (uint8_t) bitextu((unsigned int) src_in, 2, 12);
            *pIm2Col4 = (uint8_t) bitextu((unsigned int) src_in, 2, 14);
            pIm2Col++;
            pIm2Col2++;
            pIm2Col3++;
            pIm2Col4++;
            *pIm2Col = (uint8_t) bitextu((unsigned int) src_in, 2, 16);
            *pIm2Col2 = (uint8_t) bitextu((unsigned int) src_in, 2, 18);
            *pIm2Col3 = (uint8_t) bitextu((unsigned int) src_in, 2, 20);
            *pIm2Col4 = (uint8_t) bitextu((unsigned int) src_in, 2, 22);
            pIm2Col++;
            pIm2Col2++;
            pIm2Col3++;
            pIm2Col4++;
            *pIm2Col = (uint8_t) bitextu((unsigned int) src_in, 2, 24);
            *pIm2Col2 = (uint8_t) bitextu((unsigned int) src_in, 2, 26);
            *pIm2Col3 = (uint8_t) bitextu((unsigned int) src_in, 2, 28);
            *pIm2Col4 = (uint8_t) bitextu((unsigned int) src_in, 2, 30);
            pIm2Col++;
            pIm2Col2++;
            pIm2Col3++;
            pIm2Col4++;
            idx+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
          pIm2Col-=(dim_incr_pad_left - const1);
          pIm2Col2-=(dim_incr_pad_left - const1);
          pIm2Col3-=(dim_incr_pad_left - const1);
          pIm2Col4-=(dim_incr_pad_left - const1);
          base_ptr+=dim_in_x;
          i_buff_y++;
        }while(i_buff_y < dim_in_y);
        for (i_buff_y; i_buff_y < dim_in_y + padding_y_bottom; i_buff_y++)
        {
          int i=0;
          do
          {
            *(v4u *) pIm2Col = (v4u) {0, 0, 0, 0};
            pIm2Col+=4;
            *(v4u *) pIm2Col2 = (v4u) {0, 0, 0, 0};
            pIm2Col2+=4;
            *(v4u *) pIm2Col3 = (v4u) {0, 0, 0, 0};
            pIm2Col3+=4;
            *(v4u *) pIm2Col4 = (v4u) {0, 0, 0, 0};
            pIm2Col4+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
          pIm2Col-=dim_incr;
          pIm2Col2-=dim_incr;
          pIm2Col3-=dim_incr;
          pIm2Col4-=dim_incr;
        }

        int l=0;
        do
        {
          pWt = pWeightBuffer + i_wt_ch;
          pWt2 = pWt + kernel_size;
          pWt3 = pWt2 + kernel_size;
          pWt4 = pWt3 + kernel_size;
          int sum = 0;
          int sum2 = 0;
          int sum3 = 0;
          int sum4 = 0;
          pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
          pIm2Col2 = pIm2Col + im2col_size;
          pIm2Col3 = pIm2Col2 + im2col_size;
          pIm2Col4 = pIm2Col3 + im2col_size;
          int j=0;
          do
          {
            v4s w = *(v4s *) pWt;
            v4u x = *(v4u *) pIm2Col;
            sum = SumDotp(x, w, sum);
            pWt += 4;
            pIm2Col += 4;
            v4s w2 = *(v4s *) pWt2;
            v4u x2 = *(v4u *) pIm2Col2;
            sum2 = SumDotp(x2, w2, sum2);
            pWt2 += 4;
            pIm2Col2 += 4;
            v4s w3 = *(v4s *) pWt3;
            v4u x3 = *(v4u *) pIm2Col3;
            sum3 = SumDotp(x3, w3, sum3);
            pWt3 += 4;
            pIm2Col3 += 4;
            v4s w4 = *(v4s *) pWt4;
            v4u x4 = *(v4u *) pIm2Col4;
            sum4 = SumDotp(x4, w4, sum4);
            pWt4 += 4;
            pIm2Col4 += 4;
            j++;
          }while(j<colCnt);
          for(int j=0; j<leftCnt; j++)
          {
            int8_t w = *(int8_t *) pWt++;
            uint8_t x = *(uint8_t *) pIm2Col++;
            sum += x * w;
            int8_t w2 = *(int8_t *) pWt2++;
            uint8_t x2 = *(uint8_t *) pIm2Col2++;
            sum2 += x2 * w2;
            int8_t w3 = *(int8_t *) pWt3++;
            uint8_t x3 = *(uint8_t *) pIm2Col3++;
            sum3 += x3 * w3;
            int8_t w4 = *(int8_t *) pWt4++;
            uint8_t x4 = *(uint8_t *) pIm2Col4++;
            sum4 += x4 * w4;
          }
          if (flag_batch_norm && flag_relu)
          {
            sum = pulp_nn_bn_quant_u2(sum, *k1, *lambda1, out_shift);
            sum2 = pulp_nn_bn_quant_u2(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
            sum3 = pulp_nn_bn_quant_u2(sum3, *(k1 + 2), *(lambda1 + 2), out_shift);
            sum4 = pulp_nn_bn_quant_u2(sum4, *(k1 + 3), *(lambda1 + 3), out_shift);
            sum = bitins(sum, n_mask2, sum2, mask2, off2);
            sum = bitins(sum, n_mask4, sum3, mask4, off4);
            *pOut = bitins(sum, n_mask6, sum4, mask6, off6);
          }
          else
          {
            if(flag_relu == 1)
            {
              sum = pulp_nn_quant_u2(sum, out_mult, out_shift);
              sum2 = pulp_nn_quant_u2(sum2, out_mult, out_shift);
              sum3 = pulp_nn_quant_u2(sum3, out_mult, out_shift);
              sum4 = pulp_nn_quant_u2(sum4, out_mult, out_shift);
              sum = bitins(sum, n_mask2, sum2, mask2, off2);
              sum = bitins(sum, n_mask4, sum3, mask4, off4);
              *pOut = bitins(sum, n_mask6, sum4, mask6, off6);
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
            }
          }
          pOut+=(dim_out_x * ch_out_r);
          l++;
        }while(l<dim_out_y);
        i_out_x++;
      }while((i_out_x * stride_x) < padding_x_left);
    }
    do
    {
      uint8_t *pOut = pOutBuffer + i_out_ch + (i_out_x * ch_out_r);
      uint8_t *pIm2Col = pIm2ColBase;
      uint8_t *pIm2Col2 = pIm2Col + im2col_size;
      uint8_t *pIm2Col3 = pIm2Col2 + im2col_size;
      uint8_t *pIm2Col4 = pIm2Col3 + im2col_size;
      i_buff_y = - padding_y_top;
      if(padding_y_top > 0)
      {
        do
        {
          int i=0;
          do
          {
            *(v4u *) pIm2Col = (v4u) {0, 0, 0, 0};
            pIm2Col+=4;
            *(v4u *) pIm2Col2 = (v4u) {0, 0, 0, 0};
            pIm2Col2+=4;
            *(v4u *) pIm2Col3 = (v4u) {0, 0, 0, 0};
            pIm2Col3+=4;
            *(v4u *) pIm2Col4 = (v4u) {0, 0, 0, 0};
            pIm2Col4+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
          pIm2Col-=dim_incr;
          pIm2Col2-=dim_incr;
          pIm2Col3-=dim_incr;
          pIm2Col4-=dim_incr;
          i_buff_y++;
        }while(i_buff_y < 0);
      }
      int base_ptr = pInBuffer + i_in_ch + (i_out_x * stride_x) - padding_x_left;
      do
      {
        int idx = 0;
        for (int i=0; i<dim_kernel_x_size_padded; i++)
        {
          v4u src_in = *((v4u*) (base_ptr + idx));
          *pIm2Col = (uint8_t) bitextu((unsigned int) src_in, 2, 0);
          *pIm2Col2 = (uint8_t) bitextu((unsigned int) src_in, 2, 2);
          *pIm2Col3 = (uint8_t) bitextu((unsigned int) src_in, 2, 4);
          *pIm2Col4 = (uint8_t) bitextu((unsigned int) src_in, 2, 6);
          pIm2Col++;
          pIm2Col2++;
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col = (uint8_t) bitextu((unsigned int) src_in, 2, 8);
          *pIm2Col2 = (uint8_t) bitextu((unsigned int) src_in, 2, 10);
          *pIm2Col3 = (uint8_t) bitextu((unsigned int) src_in, 2, 12);
          *pIm2Col4 = (uint8_t) bitextu((unsigned int) src_in, 2, 14);
          pIm2Col++;
          pIm2Col2++;
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col = (uint8_t) bitextu((unsigned int) src_in, 2, 16);
          *pIm2Col2 = (uint8_t) bitextu((unsigned int) src_in, 2, 18);
          *pIm2Col3 = (uint8_t) bitextu((unsigned int) src_in, 2, 20);
          *pIm2Col4 = (uint8_t) bitextu((unsigned int) src_in, 2, 22);
          pIm2Col++;
          pIm2Col2++;
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col = (uint8_t) bitextu((unsigned int) src_in, 2, 24);
          *pIm2Col2 = (uint8_t) bitextu((unsigned int) src_in, 2, 26);
          *pIm2Col3 = (uint8_t) bitextu((unsigned int) src_in, 2, 28);
          *pIm2Col4 = (uint8_t) bitextu((unsigned int) src_in, 2, 30);
          pIm2Col++;
          pIm2Col2++;
          pIm2Col3++;
          pIm2Col4++;
          idx+=4;
        }
        pIm2Col-=dim_incr;
        pIm2Col2-=dim_incr;
        pIm2Col3-=dim_incr;
        pIm2Col4-=dim_incr;
        base_ptr+=dim_in_x;
        i_buff_y++;
      }while(i_buff_y < dim_in_y);
      for (i_buff_y; i_buff_y < dim_in_y + padding_y_bottom; i_buff_y++)
      {
        int i=0;
        do
        {
          *(v4u *) pIm2Col = (v4u) {0, 0, 0, 0};
          pIm2Col+=4;
          *(v4u *) pIm2Col2 = (v4u) {0, 0, 0, 0};
          pIm2Col2+=4;
          *(v4u *) pIm2Col3 = (v4u) {0, 0, 0, 0};
          pIm2Col3+=4;
          *(v4u *) pIm2Col4 = (v4u) {0, 0, 0, 0};
          pIm2Col4+=4;
          i++;
        }while(i<dim_kernel_x_size_padded);
        pIm2Col-=dim_incr;
        pIm2Col2-=dim_incr;
        pIm2Col3-=dim_incr;
        pIm2Col4-=dim_incr;
      }
      int l=0;
      do
      {
        pWt = pWeightBuffer + i_wt_ch;
        pWt2 = pWt + kernel_size;
        pWt3 = pWt2 + kernel_size;
        pWt4 = pWt3 + kernel_size;
        int sum = 0;
        int sum2 = 0;
        int sum3 = 0;
        int sum4 = 0;
        pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
        pIm2Col2 = pIm2Col + im2col_size;
        pIm2Col3 = pIm2Col2 + im2col_size;
        pIm2Col4 = pIm2Col3 + im2col_size;
        int j=0;
        do
        {
          v4s w = *(v4s *) pWt;
          v4u x = *(v4u *) pIm2Col;
          sum = SumDotp(x, w, sum);
          pWt += 4;
          pIm2Col += 4;
          v4s w2 = *(v4s *) pWt2;
          v4u x2 = *(v4u *) pIm2Col2;
          sum2 = SumDotp(x2, w2, sum2);
          pWt2 += 4;
          pIm2Col2 += 4;
          v4s w3 = *(v4s *) pWt3;
          v4u x3 = *(v4u *) pIm2Col3;
          sum3 = SumDotp(x3, w3, sum3);
          pWt3 += 4;
          pIm2Col3 += 4;
          v4s w4 = *(v4s *) pWt4;
          v4u x4 = *(v4u *) pIm2Col4;
          sum4 = SumDotp(x4, w4, sum4);
          pWt4 += 4;
          pIm2Col4 += 4;
          j++;
        }while(j<colCnt);
        for(int j=0; j<leftCnt; j++)
        {
          int8_t w = *(int8_t *) pWt++;
          uint8_t x = *(uint8_t *) pIm2Col++;
          sum += x * w;
          int8_t w2 = *(int8_t *) pWt2++;
          uint8_t x2 = *(uint8_t *) pIm2Col2++;
          sum2 += x2 * w2;
          int8_t w3 = *(int8_t *) pWt3++;
          uint8_t x3 = *(uint8_t *) pIm2Col3++;
          sum3 += x3 * w3;
          int8_t w4 = *(int8_t *) pWt4++;
          uint8_t x4 = *(uint8_t *) pIm2Col4++;
          sum4 += x4 * w4;
        }
        if (flag_batch_norm && flag_relu)
        {
          sum = pulp_nn_bn_quant_u2(sum, *k1, *lambda1, out_shift);
          sum2 = pulp_nn_bn_quant_u2(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
          sum3 = pulp_nn_bn_quant_u2(sum3, *(k1 + 2), *(lambda1 + 2), out_shift);
          sum4 = pulp_nn_bn_quant_u2(sum4, *(k1 + 3), *(lambda1 + 3), out_shift);
          sum = bitins(sum, n_mask2, sum2, mask2, off2);
          sum = bitins(sum, n_mask4, sum3, mask4, off4);
          *pOut = bitins(sum, n_mask6, sum4, mask6, off6);
        }
        else
        {
          if(flag_relu == 1)
          {
            sum = pulp_nn_quant_u2(sum, out_mult, out_shift);
            sum2 = pulp_nn_quant_u2(sum2, out_mult, out_shift);
            sum3 = pulp_nn_quant_u2(sum3, out_mult, out_shift);
            sum4 = pulp_nn_quant_u2(sum4, out_mult, out_shift);
            sum = bitins(sum, n_mask2, sum2, mask2, off2);
            sum = bitins(sum, n_mask4, sum3, mask4, off4);
            *pOut = bitins(sum, n_mask6, sum4, mask6, off6);
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
          }
        }
        pOut+=(dim_out_x * ch_out_r);
        l++;
      }while(l<dim_out_y);
      i_out_x++;
    }while((i_out_x * stride_x) < ((dim_out_x * stride_x) - padding_x_right));
    for (i_out_x; i_out_x < dim_out_x; i_out_x++)
    {
      uint8_t *pOut = pOutBuffer + i_out_ch + (i_out_x * ch_out_r);
      uint8_t *pIm2Col = pIm2ColBase;
      uint8_t *pIm2Col2 = pIm2Col + im2col_size;
      uint8_t *pIm2Col3 = pIm2Col2 + im2col_size;
      uint8_t *pIm2Col4 = pIm2Col3 + im2col_size;
      asm volatile ("":::"memory");
      i_buff_y = - padding_y_top;
      if(padding_y_top > 0)
      {
        do
        {
          int i=0;
          do
          {
            *(v4u *) pIm2Col = (v4u) {0, 0, 0, 0};
            pIm2Col+=4;
            *(v4u *) pIm2Col2 = (v4u) {0, 0, 0, 0};
            pIm2Col2+=4;
            *(v4u *) pIm2Col3 = (v4u) {0, 0, 0, 0};
            pIm2Col3+=4;
            *(v4u *) pIm2Col4 = (v4u) {0, 0, 0, 0};
            pIm2Col4+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
          pIm2Col-=dim_incr;
          pIm2Col2-=dim_incr;
          pIm2Col3-=dim_incr;
          pIm2Col4-=dim_incr;
          i_buff_y++;
        }while(i_buff_y < 0);
      }
      int base_ptr = pInBuffer + i_in_ch + (i_out_x * stride_x) - padding_x_left;
      do
      {
        int i = 0;
        int idx = 0;
        do
        {
          v4u src_in = *((v4u*) (base_ptr + idx));
          *pIm2Col = (uint8_t) bitextu((unsigned int) src_in, 2, 0);
          *pIm2Col2 = (uint8_t) bitextu((unsigned int) src_in, 2, 2);
          *pIm2Col3 = (uint8_t) bitextu((unsigned int) src_in, 2, 4);
          *pIm2Col4 = (uint8_t) bitextu((unsigned int) src_in, 2, 6);
          pIm2Col++;
          pIm2Col2++;
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col = (uint8_t) bitextu((unsigned int) src_in, 2, 8);
          *pIm2Col2 = (uint8_t) bitextu((unsigned int) src_in, 2, 10);
          *pIm2Col3 = (uint8_t) bitextu((unsigned int) src_in, 2, 12);
          *pIm2Col4 = (uint8_t) bitextu((unsigned int) src_in, 2, 14);
          pIm2Col++;
          pIm2Col2++;
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col = (uint8_t) bitextu((unsigned int) src_in, 2, 16);
          *pIm2Col2 = (uint8_t) bitextu((unsigned int) src_in, 2, 18);
          *pIm2Col3 = (uint8_t) bitextu((unsigned int) src_in, 2, 20);
          *pIm2Col4 = (uint8_t) bitextu((unsigned int) src_in, 2, 22);
          pIm2Col++;
          pIm2Col2++;
          pIm2Col3++;
          pIm2Col4++;
          *pIm2Col = (uint8_t) bitextu((unsigned int) src_in, 2, 24);
          *pIm2Col2 = (uint8_t) bitextu((unsigned int) src_in, 2, 26);
          *pIm2Col3 = (uint8_t) bitextu((unsigned int) src_in, 2, 28);
          *pIm2Col4 = (uint8_t) bitextu((unsigned int) src_in, 2, 30);
          pIm2Col++;
          pIm2Col2++;
          pIm2Col3++;
          pIm2Col4++;
          idx+=4;
          i++;
        }while(i<dim_kernel_x_size_padded);

        pIm2Col-=(dim_incr_pad_right + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right);
        pIm2Col2-=(dim_incr_pad_right + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right);
        pIm2Col3-=(dim_incr_pad_right + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right);
        pIm2Col4-=(dim_incr_pad_right + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right);
        base_ptr+=dim_in_x;
        for(int j=0; j<(1 + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right); j++)
        {
          *(uint8_t *) pIm2Col = 0;
          pIm2Col++;
          *(uint8_t *) pIm2Col2 = 0;
          pIm2Col2++;
          *(uint8_t *) pIm2Col3 = 0;
          pIm2Col3++;
          *(uint8_t *) pIm2Col4 = 0;
          pIm2Col4++;
        }
        i_buff_y++;
      }while(i_buff_y < dim_in_y);
      for (i_buff_y; i_buff_y < dim_in_y + padding_y_bottom; i_buff_y++)
      {
        int i=0;
        do
        {
          *(v4u *) pIm2Col = (v4u) {0, 0, 0, 0};
          pIm2Col+=4;
          *(v4u *) pIm2Col2 = (v4u) {0, 0, 0, 0};
          pIm2Col2+=4;
          *(v4u *) pIm2Col3 = (v4u) {0, 0, 0, 0};
          pIm2Col3+=4;
          *(v4u *) pIm2Col4 = (v4u) {0, 0, 0, 0};
          pIm2Col4+=4;
          i++;
        }while(i<dim_kernel_x_size_padded);
        pIm2Col-=dim_incr;
        pIm2Col2-=dim_incr;
        pIm2Col3-=dim_incr;
        pIm2Col4-=dim_incr;
      }

      int l=0;
      do
      {
        pWt = pWeightBuffer + i_wt_ch;
        pWt2 = pWt + kernel_size;
        pWt3 = pWt2 + kernel_size;
        pWt4 = pWt3 + kernel_size;
        int sum = 0;
        int sum2 = 0;
        int sum3 = 0;
        int sum4 = 0;
        pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
        pIm2Col2 = pIm2Col + im2col_size;
        pIm2Col3 = pIm2Col2 + im2col_size;
        pIm2Col4 = pIm2Col3 + im2col_size;
        int j=0;
        do
        {
          v4s w = *(v4s *) pWt;
          v4u x = *(v4u *) pIm2Col;
          sum = SumDotp(x, w, sum);
          pWt += 4;
          pIm2Col += 4;
          v4s w2 = *(v4s *) pWt2;
          v4u x2 = *(v4u *) pIm2Col2;
          sum2 = SumDotp(x2, w2, sum2);
          pWt2 += 4;
          pIm2Col2 += 4;
          v4s w3 = *(v4s *) pWt3;
          v4u x3 = *(v4u *) pIm2Col3;
          sum3 = SumDotp(x3, w3, sum3);
          pWt3 += 4;
          pIm2Col3 += 4;
          v4s w4 = *(v4s *) pWt4;
          v4u x4 = *(v4u *) pIm2Col4;
          sum4 = SumDotp(x4, w4, sum4);
          pWt4 += 4;
          pIm2Col4 += 4;
          j++;
        }while(j<colCnt);
        for(int j=0; j<leftCnt; j++)
        {
          int8_t w = *(int8_t *) pWt++;
          uint8_t x = *(uint8_t *) pIm2Col++;
          sum += x * w;
          int8_t w2 = *(int8_t *) pWt2++;
          uint8_t x2 = *(uint8_t *) pIm2Col2++;
          sum2 += x2 * w2;
          int8_t w3 = *(int8_t *) pWt3++;
          uint8_t x3 = *(uint8_t *) pIm2Col3++;
          sum3 += x3 * w3;
          int8_t w4 = *(int8_t *) pWt4++;
          uint8_t x4 = *(uint8_t *) pIm2Col4++;
          sum4 += x4 * w4;
        }
        if (flag_batch_norm && flag_relu)
        {
          sum = pulp_nn_bn_quant_u2(sum, *k1, *lambda1, out_shift);
          sum2 = pulp_nn_bn_quant_u2(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
          sum3 = pulp_nn_bn_quant_u2(sum3, *(k1 + 2), *(lambda1 + 2), out_shift);
          sum4 = pulp_nn_bn_quant_u2(sum4, *(k1 + 3), *(lambda1 + 3), out_shift);
          sum = bitins(sum, n_mask2, sum2, mask2, off2);
          sum = bitins(sum, n_mask4, sum3, mask4, off4);
          *pOut = bitins(sum, n_mask6, sum4, mask6, off6);
        }
        else
        {
          if(flag_relu == 1)
          {
            sum = pulp_nn_quant_u2(sum, out_mult, out_shift);
            sum2 = pulp_nn_quant_u2(sum2, out_mult, out_shift);
            sum3 = pulp_nn_quant_u2(sum3, out_mult, out_shift);
            sum4 = pulp_nn_quant_u2(sum4, out_mult, out_shift);
            sum = bitins(sum, n_mask2, sum2, mask2, off2);
            sum = bitins(sum, n_mask4, sum3, mask4, off4);
            *pOut = bitins(sum, n_mask6, sum4, mask6, off6);
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
          }
        }
        pOut+=(dim_out_x * ch_out_r);
        l++;
      }while(l<dim_out_y);
    }
    i_in_ch+=in_image_size;
    i_wt_ch+=(kernel_size << 2);
    k1+=4;
    lambda1+=4;
    i_out_ch++;
  }
  pi_cl_team_barrier(0);
}
