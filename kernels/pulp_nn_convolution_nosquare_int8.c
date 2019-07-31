/*
* pulp_nn_convolution_nosquare_int8.c
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
#define NN_ROUND(out_shift)         (0x1 << (out_shift -1))
#define MIN(a,b)                    ((a)<(b)?(a):(b))
#define CLIP8(x)                    __builtin_pulp_clip(x,-128, 127)

#define SIZE4x2KERNEL

/**
* @brief INT8 zero-mem: necessary to add padding to the IFM
* @param[in,out]       pBuffer      pointer to buffer
* @param[in,out]       Size         size
* @return          void
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
* @return          void
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
* @brief INT8 convolution function, nosquare input, nosquare convolution kernel
* @param[in]       Im_in            pointer to input tensor
* @param[in]       dim_im_in_x      input tensor width
* @param[in]       dim_im_in_y      input tensor height
* @param[in]       ch_im_in         number of input tensor channels
* @param[in]       wt               pointer to kernel weights
* @param[in]       ch_im_out        number of filters, i.e., output tensor channels
* @param[in]       dim_kernel_x     filter kernel width
* @param[in]       dim_kernel_y     filter kernel height
* @param[in]       padding_x        width padding size
* @param[in]       padding_y        height padding size
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
void __attribute__ ((noinline)) pulp_nn_convolution_nosquare_int8(
  const int8_t * Im_in,
  const uint16_t dim_im_in_x,
  const uint16_t dim_im_in_y,
  const uint16_t ch_im_in,
  const int8_t * wt,
  const uint16_t ch_im_out,
  const uint16_t dim_kernel_x,
  const uint16_t dim_kernel_y,
  const uint16_t padding_x,
  const uint16_t padding_y,
  const uint16_t stride_x,
  const uint16_t stride_y,
  const int8_t * bias,
  const uint16_t bias_shift,
  const uint16_t out_shift,
  int8_t * Im_out,
  const uint16_t dim_im_out_x,
  const uint16_t dim_im_out_y,
  int8_t * bufferA,
  int8_t * bufferB)
  {


    int16_t   i_out_y, i_out_x, i_ker_y, i_ker_x;



    int8_t    *pBuffer = bufferA;
    int8_t     *pOut = Im_out;

    /* top part */
    for (i_out_y = 0; i_out_y < padding_y; i_out_y++)
    {
      for (i_out_x = 0; i_out_x < dim_im_out_x; i_out_x++)
      {
        /* im2col  */
        for (i_ker_y = i_out_y * stride_y - padding_y; i_ker_y < i_out_y * stride_y - padding_y + dim_kernel_y;
          i_ker_y++)
          {
            for (i_ker_x = i_out_x * stride_x - padding_x; i_ker_x < i_out_x * stride_x - padding_x + dim_kernel_x;
              i_ker_x++)
              {
                if (i_ker_y < 0 || i_ker_y >= dim_im_in_y || i_ker_x < 0 || i_ker_x >= dim_im_in_x)
                {
                  pulp_zero_mem(pBuffer, ch_im_in);
                }
                else
                {
                  pulp_nn_im2col_int8((int8_t *) Im_in + (i_ker_y * dim_im_in_x + i_ker_x) * ch_im_in,
                  pBuffer, ch_im_in);
                }
                pBuffer += ch_im_in;
              }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel_x * dim_kernel_y)
            {
              pOut =
              pulp_nn_matmul_4x2_int8(wt, bufferA, ch_im_out, ch_im_in * dim_kernel_x * dim_kernel_y,
                bias_shift, out_shift, bias, pOut);
                /* counter reset */
                pBuffer = bufferA;
              }
            }
          }

          /* middle part, split into left, mid and right */
          for (; i_out_y < dim_im_out_y - padding_y; i_out_y++)
          {

            /* left part */
            for (i_out_x = 0; i_out_x < padding_x; i_out_x++)
            {
              /* This part implements the im2col function */
              for (i_ker_y = i_out_y * stride_y - padding_y; i_ker_y < i_out_y * stride_y - padding_y + dim_kernel_y;
                i_ker_y++)
                {
                  for (i_ker_x = i_out_x * stride_x - padding_x; i_ker_x < i_out_x * stride_x - padding_x + dim_kernel_x;
                    i_ker_x++)
                    {
                      if (i_ker_x < 0 || i_ker_x >= dim_im_in_x)
                      {
                        pulp_zero_mem(pBuffer, ch_im_in);
                      } else
                      {
                        pulp_nn_im2col_int8((int8_t *) Im_in + (i_ker_y * dim_im_in_x + i_ker_x) * ch_im_in,
                        pBuffer, ch_im_in);
                      }
                      pBuffer += ch_im_in;
                    }
                  }

                  if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel_x * dim_kernel_y)
                  {
                    pOut =
                    pulp_nn_matmul_4x2_int8(wt, bufferA, ch_im_out, ch_im_in * dim_kernel_x * dim_kernel_y,
                      bias_shift, out_shift, bias, pOut);
                      /* counter reset */
                      pBuffer = bufferA;
                    }
                  }

                  /* mid part */
                  for (; i_out_x < dim_im_out_x - padding_x; i_out_x++)
                  {
                    /* im2col function */
                    for (i_ker_y = i_out_y * stride_y - padding_y; i_ker_y < i_out_y * stride_y - padding_y + dim_kernel_y;
                      i_ker_y++)
                      {
                        pulp_nn_im2col_int8((int8_t *) Im_in +
                        (i_ker_y * dim_im_in_x + i_out_x * stride_x - padding_x) * ch_im_in,
                        pBuffer, ch_im_in * dim_kernel_x);
                        pBuffer += ch_im_in * dim_kernel_x;
                      }

                      if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel_x * dim_kernel_y)
                      {

                        pOut =
                        pulp_nn_matmul_4x2_int8(wt, bufferA, ch_im_out, ch_im_in * dim_kernel_x * dim_kernel_y,
                          bias_shift, out_shift, bias, pOut);
                          /* counter reset */
                          pBuffer = bufferA;
                        }
                      }

                      /* right part */
                      for (; i_out_x < dim_im_out_x; i_out_x++)
                      {
                        
                        for (i_ker_y = i_out_y * stride_y - padding_y; i_ker_y < i_out_y * stride_y - padding_y + dim_kernel_y;
                          i_ker_y++)
                          {
                            for (i_ker_x = i_out_x * stride_x - padding_x; i_ker_x < i_out_x * stride_x - padding_x + dim_kernel_x;
                              i_ker_x++)
                              {
                                if (i_ker_x < 0 || i_ker_x >= dim_im_in_x)
                                {
                                 
                                  pulp_zero_mem(pBuffer, ch_im_in);
                                } else
                                {
                                  pulp_nn_im2col_int8((int8_t *) Im_in + (i_ker_y * dim_im_in_x + i_ker_x) * ch_im_in,
                                  pBuffer, ch_im_in);
                                }
                                pBuffer += ch_im_in;
                              }
                            }

                            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel_x * dim_kernel_y)
                            {
                              pOut =
                              pulp_nn_matmul_4x2_int8(wt, bufferA, ch_im_out, ch_im_in * dim_kernel_x * dim_kernel_y,
                                bias_shift, out_shift, bias, pOut);
                                /* counter reset */
                                pBuffer = bufferA;
                              }
                            }
                          }

                          for (; i_out_y < dim_im_out_y; i_out_y++)
                          {
                            for (i_out_x = 0; i_out_x < dim_im_out_x; i_out_x++)
                            {
                              /* This part implements the im2col function */
                              for (i_ker_y = i_out_y * stride_y - padding_y; i_ker_y < i_out_y * stride_y - padding_y + dim_kernel_y;
                                i_ker_y++)
                                {
                                  for (i_ker_x = i_out_x * stride_x - padding_x; i_ker_x < i_out_x * stride_x - padding_x + dim_kernel_x;
                                    i_ker_x++)
                                    {
                                      if (i_ker_y < 0 || i_ker_y >= dim_im_in_y || i_ker_x < 0 || i_ker_x >= dim_im_in_x)
                                      {
                                       
                                        pulp_zero_mem(pBuffer, ch_im_in);
                                      } else
                                      {
                                        pulp_nn_im2col_int8((int8_t *) Im_in + (i_ker_y * dim_im_in_x + i_ker_x) * ch_im_in,
                                        pBuffer, ch_im_in);
                                      }
                                      pBuffer += ch_im_in;
                                    }
                                  }

                                  if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel_x * dim_kernel_y)
                                  {
                                    pOut =
                                    pulp_nn_matmul_4x2_int8(wt, bufferA, ch_im_out, ch_im_in * dim_kernel_x * dim_kernel_y,
                                      bias_shift, out_shift, bias, pOut);
                                      /* counter reset */
                                      pBuffer = bufferA;
                                    }
                                  }
                                }

                                /* check if there is left-over for compute */
                                if (pBuffer != bufferA)
                                {
                                  const int8_t *pA = wt;
                                  int       i;
                                  for (i = 0; i < ch_im_out; i++)
                                  {
                                    int     sum = ((int)(bias[i]) << bias_shift) + NN_ROUND(out_shift);
                                    int8_t    *pB = bufferA;
                                    /* it processes 4 entries */
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
                              }
