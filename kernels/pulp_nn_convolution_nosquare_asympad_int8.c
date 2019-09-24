/*
* pulp_nn_convolution_nosquare_asympad_int8.c
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

#define SumDotp(a, b, c)            __builtin_pulp_sdotsp4(a, b, c)
#define NN_ROUND(out_shift)         ((out_shift) ? (0x1 << (out_shift -1)) : (0))
#define MIN(a,b)                    ((a)<(b)?(a):(b))
#define CLIP8(x)                    __builtin_pulp_clip(x,-128, 127)

/*degree of freedom: if defined the 4x2 MatMul kernel is used. */
/* default: 2x2 kernel. */
/* 4x2 recommended for best performance */
#define SIZE4x2KERNEL

/**
* @brief INT8 zero-mem: necessary to add padding to the IFM
* @param[in,out]       pBuffer      pointer to buffer
* @param[in,out]       Size         size
* @return none.
*/

inline void pulp_zero_mem(int8_t * pBuffer, int size)
{
  v4s* pDst = (v4s *)pBuffer;
  int lfover = size &0x3;
  for (int i=0; i<(size>>2); i++)
  {
    *((v4s*) pBuffer) = (v4s){0,0,0,0};
    pBuffer+=4;
  }
  while(lfover)
  {
    *pBuffer++=0;
    lfover--;
  }
}

/**
* @brief INT8 image-like to columns function
* @param[in]       pInput            pointer to image-like input tensor
* @param[in,out]   pOutput           pointer to input columns
* @param[in]       blockSize         size of the input block to be transformed
* @return none.
*/
inline void pulp_nn_im2col_int8(int8_t * pInput, int8_t * pOutput, unsigned int blockSize)
{
  unsigned int blkCnt = blockSize >> 2u;
  unsigned int lfover = blockSize & 0x3;

  for (int i = 0; i<blkCnt; i++)
  {
    *((v4s*)pOutput) = *((v4s*) pInput);
    pInput+=4;
    pOutput+=4;
  }
  while(lfover)
  {
    *((int8_t*)pOutput) = *((int8_t*)pInput);
    pOutput++;
    pInput++;
    lfover--;
  }
}


/**
* @brief INT8 convolution function with asymmetric padding, nosquare input, nosquare convolution kernel
* @param[in]       Im_in            pointer to input tensor
* @param[in]       dim_im_in_x      input tensor width
* @param[in]       dim_im_in_y      input tensor height
* @param[in]       ch_im_in         number of input tensor channels
* @param[in]       wt               pointer to kernel weights
* @param[in]       ch_im_out        number of filters, i.e., output tensor channels
* @param[in]       dim_kernel_x     filter kernel width
* @param[in]       dim_kernel_y     filter kernel height
* @param[in]       padding_y_top    top padding size
* @param[in]       padding_y_bottom bottom padding size
* @param[in]       padding_x_left   left padding size
* @param[in]       padding_x_right  right padding size
* @param[in]       stride_x         convolution stride along width
* @param[in]       stride_y         convolution stride along height
* @param[in]       bias             pointer to bias
* @param[in]       bias_shift       amount of shift on bias
* @param[in]       out_shift        amount of shift on output
* @param[in,out]   Im_out           pointer to output tensor
* @param[in]       dim_im_out_x     output tensor width
* @param[in]       dim_im_out_y     output tensor height
* @param[in,out]   bufferC          pointer to buffer space for input (used for im2col)
* @param[in,out]   bufferB          pointer to buffer space for output (not used)
* @return          The function returns either
*/



void __attribute__ ((noinline)) pulp_nn_convolution_nosquare_asympad_int8(
  const int8_t * Im_in,
  const uint16_t dim_im_in_x,
  const uint16_t dim_im_in_y,
  const uint16_t ch_im_in,
  const int8_t * wt,
  const uint16_t ch_im_out,
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
  const uint16_t out_shift,
  int8_t * Im_out,
  const uint16_t dim_im_out_x,
  const uint16_t dim_im_out_y,
  int8_t * bufferC,
  int8_t * bufferB)
  {

    /* parallelization */
    int core_id = rt_core_id();
    int8_t * bufferA = bufferC  + (2*core_id*ch_im_in*dim_kernel_x*dim_kernel_y);

    // local vars
    int16_t i_out_y, i_out_x, i_ker_y, i_ker_x;
    int Log2Core = __builtin_pulp_fl1(NUM_CORES);

    /*chunks are built along the spatial dimension of the OFM */
    int chunck = (dim_im_out_y >> Log2Core) + ((dim_im_out_y & (NUM_CORES-1))!=0);

    /* defining the specific pixels computed by each core */
    int start_pixel, stop_pixel;
    start_pixel = MIN(chunck *  core_id, dim_im_out_y);
    stop_pixel = MIN(start_pixel+chunck, dim_im_out_y);
    int8_t *pBuffer = bufferA;
    int8_t  *pOut    = Im_out + start_pixel * ch_im_out * dim_im_out_x;

    /* start kernel: this first phase is devoted to building the im2col buffers */
    for (i_out_y = start_pixel; i_out_y < stop_pixel; i_out_y++)
    {
      for (i_out_x = 0; i_out_x < dim_im_out_x; i_out_x++)
      {
        if(i_out_y < padding_y_top)
        {
          for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y;i_ker_y++)
          {
            for (i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x;i_ker_x++)
            {
              if (i_ker_y < 0 || i_ker_y >= dim_im_in_y || i_ker_x < 0 || i_ker_x >= dim_im_in_x)
              {
                /* if padding needed, fill the im2col with zeros */
                pulp_zero_mem(pBuffer, ch_im_in);
              }
              else
              {
                /* 3d image like into 1d array transformation */
                pulp_nn_im2col_int8((int8_t *) Im_in + (i_ker_y * dim_im_in_x + i_ker_x) * ch_im_in,pBuffer, ch_im_in);
              }
              pBuffer += ch_im_in;
            }
          }
        }
        else if(i_out_y < dim_im_out_y - padding_y_bottom)
        {
          if(i_out_x < padding_x_left)
          {
            for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++)
            {
              for (i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++)
              {
                if (i_ker_x < 0 || i_ker_x >= dim_im_in_x)
                {
                  /* if padding needed, fill the im2col with zeros */
                  pulp_zero_mem(pBuffer, ch_im_in);
                }
                else
                {
                  /* 3d image like into 1d array transformation */
                  pulp_nn_im2col_int8((int8_t *) Im_in + (i_ker_y * dim_im_in_x + i_ker_x) * ch_im_in, pBuffer, ch_im_in);
                }
                pBuffer += ch_im_in;
              }
            }
          }
          else if(i_out_x < dim_im_out_x - padding_x_right)
          {
            for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++)
            {
              /* 3d image like into 1d array transformation */
              pulp_nn_im2col_int8((int8_t *) Im_in + (i_ker_y * dim_im_in_x + i_out_x * stride_x - padding_x_left) * ch_im_in,
              pBuffer, ch_im_in * dim_kernel_x);
              pBuffer += ch_im_in * dim_kernel_x;
            }
          }
          else
          {

            for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++)
            {
              for (i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++)
              {
                if (i_ker_x < 0 || i_ker_x >= dim_im_in_x)
                {
                  /* if padding needed, fill the im2col with zeros */
                  pulp_zero_mem(pBuffer, ch_im_in);
                }
                else
                {
                  /* 3d image like into 1d array transformation */
                  pulp_nn_im2col_int8((int8_t *) Im_in + (i_ker_y * dim_im_in_x + i_ker_x) * ch_im_in,pBuffer, ch_im_in);
                }
                pBuffer += ch_im_in;
              }
            }
          }
        }
        else
        {
          for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++)
          {
            for (i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x;i_ker_x++)
            {
              if (i_ker_y < 0 || i_ker_y >= dim_im_in_y || i_ker_x < 0 || i_ker_x >= dim_im_in_x)
              {
                /* if padding needed, fill the im2col with zeros */
                pulp_zero_mem(pBuffer, ch_im_in);
              }
              else
              {
                /* 3d image like into 1d array transformation */
                pulp_nn_im2col_int8((int8_t *) Im_in + (i_ker_y * dim_im_in_x + i_ker_x) * ch_im_in, pBuffer, ch_im_in);
              }
              pBuffer += ch_im_in;
            }
          }
        }

        /* when im2col buffers are built start the dot product computation */
        /* i.e. matrix multiplication kernel */
        if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel_x * dim_kernel_y)
        {
          pOut = pulp_nn_matmul_4x2_int8(wt, bufferA, ch_im_out, ch_im_in * dim_kernel_x * dim_kernel_y, bias_shift, out_shift, bias, pOut);
          pBuffer = bufferA;
        }
      }
    }

    /* check if there is left-over */
    if (pBuffer != bufferA)
    {
      const int8_t *pA = wt;
      int       i;
      for (i = 0; i < ch_im_out; i++)
      {
        int        sum = ((int)(bias[i]) << bias_shift) + NN_ROUND(out_shift);
        int8_t     *pB = bufferA;
        /* basically each time it process 4 entries */
        uint16_t  colCnt = ch_im_in * dim_kernel_x * dim_kernel_y >> 2;

        for (int j=0 ; j < colCnt; j++)
        {
          v4s inA = *((v4s*) pA);
          v4s inB = *((v4s*) pB);

          sum = SumDotp(inA, inB, sum);
          pA+=4;
          pB+=4;
        }
        colCnt = (ch_im_in * dim_kernel_y * dim_kernel_x) & 0x3;
        while (colCnt)
        {
          int8_t      inA1 = *pA++;
          int8_t      inB1 = *pB++;
          sum += inA1 * inB1;
          colCnt--;
        }
        *pOut = (int8_t) CLIP8(sum >> out_shift);
        pOut++;

      }

    }

    // final synch barrier
    rt_team_barrier();
  }
