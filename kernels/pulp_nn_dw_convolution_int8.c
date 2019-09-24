/*
* pulp_nn_dw_convolution_int8.c
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

#define SumDotp(a, b, c)          	__builtin_pulp_sdotsp4(a, b, c)
#define MIN(a,b)                    ((a)<(b)?(a):(b))
#define CLIP8(x)                    __builtin_pulp_clip(x,-128, 127)
#define NN_ROUND(out_shift) 		((out_shift) ? (0x1 << (out_shift -1)) : (0))
#define pack(x,y,z,t)      			__builtin_pulp_pack4(x,y,z,t)


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
    //printf("pOut: %X, Pin: %x\n", *((v4s*)pOutput), *((v4s*) pInput)  );
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
* @brief INT8 depthwise convolution function
* @param[in]       Im_in            pointer to input tensor
* @param[in]       dim_im_in        input tensor dimension
* @param[in]       ch_im_in         number of input tensor channels
* @param[in]       wt               pointer to kernel weights
* @param[in]       ch_im_out        number of filters, i.e., output tensor channels
* @param[in]       dim_kernel       filter kernel size
* @param[in]       padding          padding size
* @param[in]       stride           convolution stride
* @param[in]       bias             pointer to bias
* @param[in]       bias_shift       amount of shift on bias
* @param[in]       out_shift        amount of shift on output
* @param[in,out]   Im_out           pointer to output tensor
* @param[in]       dim_im_out       output tensor dimension
* @param[in,out]   bufferA          pointer to buffer space for input (used for im2col)
* @param[in,out]   bufferB          pointer to buffer space for output (not used)
* @return          The function returns either
*/
void pulp_nn_dw_convolution_int8    	(
  const int8_t * Im_in,
  const uint16_t dim_im_in,
  const uint16_t ch_im_in,
  const int8_t * wt,
  const uint16_t ch_im_out,
  const uint16_t dim_kernel,
  const uint16_t padding,
  const uint16_t stride,
  const int8_t * bias,
  const uint16_t bias_shift,
  const uint16_t out_shift,
  int8_t * Im_out,
  const uint16_t dim_im_out,
  int8_t * bufferC,
  int8_t * bufferB)
  {

    /* parallelization */
    int core_id = rt_core_id();
    int8_t * bufferA = bufferC  + (2*core_id*ch_im_in*dim_kernel*dim_kernel);

    // local vars
    int16_t i_out_y, i_out_x, i_ker_y, i_ker_x;
    int Log2Core = __builtin_pulp_fl1(NUM_CORES);

    /*chunks are built along the spatial dimension of the OFM */
    int chunck = (dim_im_out >> Log2Core) + ((dim_im_out & (NUM_CORES-1))!=0);

    /* defining the specific pixels computed by each core */
    int start_pixel, stop_pixel;
    start_pixel = MIN(chunck *  core_id, dim_im_out);
    stop_pixel = MIN(start_pixel+chunck, dim_im_out);

    int8_t     *colBuffer = bufferA;
    int8_t     *pBuffer = colBuffer;
    int8_t  *pOut    = Im_out + start_pixel * ch_im_out * dim_im_out;


    const int8_t *pBias = bias;
    uint16_t  rowCnt;
    uint16_t  row_shift;

    /* check if it is a depthwise */
    if (ch_im_in != ch_im_out)
    {
      return -1;
    }

    /* start kernel: this first phase is devoted to building the im2col buffers */
    for (i_out_y = start_pixel; i_out_y < stop_pixel; i_out_y++)
    {
      for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
      {
        /* image-like to columns transform*/
        for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
        {
          for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
          {
            if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
            {
              /* if padding needed, fill the im2col with zeros */
              pulp_zero_mem(pBuffer, ch_im_in);
            } else
            {
              /* 3d image like into 1d array transformation */
              pulp_nn_im2col_int8( (int8_t *) Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in, pBuffer, ch_im_in);
            }
            pBuffer += ch_im_in;
          }
        }

        rowCnt = ch_im_out >> 2;
        row_shift = 0;
        pBias = bias;

        while (rowCnt)
        {
          int     sum =  ((int)(*pBias++) << bias_shift) + NN_ROUND(out_shift);  //fix the bias
          int     sum2 = ((int)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
          int     sum3 = ((int)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
          int     sum4 = ((int)(*pBias++) << bias_shift) + NN_ROUND(out_shift);

          /* colCnt is the iterator: take into account the 8bit SIMD insns used */
          uint16_t  colCnt = (dim_kernel * dim_kernel) >> 2;

          /* fetch the IFM pixels: note that each channel is computed independently */
          const int8_t    *pB = colBuffer +  row_shift;

          /* fetch weights at right location */
          /* in the HWC format, for dw, the weights are already ordered to feed the GEMM */
          /* Co -- H -- W -- Ci but Ci == 1 --> Co -- H -- W  */
          int8_t *pA = wt + row_shift* dim_kernel * dim_kernel;
          int8_t *pA2 = pA  + dim_kernel* dim_kernel;
          int8_t *pA3 = pA2 + dim_kernel* dim_kernel;
          int8_t *pA4 = pA3 + dim_kernel* dim_kernel;
          row_shift += 4;

          /* support vectors to swap the pixels */
          /* on the fly HWC to CHW transformation (sort of) */
          int8_t pI1[4], pI2[4], pI3[4], pI4[4];
          int8_t left_p[4];
          int8_t left_w[4];

          while(colCnt)
          {
            /* need a strong optimization */
            *((v4s*)pI1) = *((v4s*) pB);
            pB += ch_im_in;
            *((v4s*)pI2) = *((v4s*) pB);
            pB += ch_im_in;
            *((v4s*)pI3) = *((v4s*) pB);
            pB += ch_im_in;
            *((v4s*)pI4) = *((v4s*) pB);
            pB += ch_im_in;

            /* on the fly HWC to CHW  --> high overhead */
            v4s i1 = pack(pI1[0], pI2[0], pI3[0], pI4[0]);
            v4s i2 = pack(pI1[1], pI2[1], pI3[1], pI4[1]);
            v4s i3 = pack(pI1[2], pI2[2], pI3[2], pI4[2]);
            v4s i4 = pack(pI1[3], pI2[3], pI3[3], pI4[3]);

            v4s w1 = *((v4s*) pA);
            pA +=4;
            v4s w2 = *((v4s*) pA2);
            pA2 +=4;
            v4s w3 = *((v4s*) pA3);
            pA3 +=4;
            v4s w4 = *((v4s*) pA4);
            pA4 +=4;

            sum  = SumDotp(i1,w1, sum );
            sum2 = SumDotp(i2,w2, sum2);
            sum3 = SumDotp(i3,w3, sum3);
            sum4 = SumDotp(i4,w4, sum4);

            colCnt--;
          }
          colCnt = (dim_kernel * dim_kernel) & 0x3;
          while(colCnt)
          {
            /* this loop, if optimized, does a mess at compiling time */
            int16_t A = (int16_t) *(pA++);
            int16_t B = (int16_t) *(pA2++);
            int16_t C = (int16_t) *(pA3++);
            int16_t D = (int16_t) *(pA4++);

            *((v4s*)left_p) = *((v4s*) pB);


            /* dummy variable to prevent the compiler doing a mess */
            int16_t a = left_p[0];
            int16_t b = left_p[1];
            int16_t c = left_p[2];
            int16_t d = left_p[3];

            /* bad stuff but needed to prevent the compiler doing a mess */
            asm volatile("": : :"memory");
            sum  += a * A;
            sum2 += b * B;
            sum3 += c * C;
            sum4 += d * D;



            pB+=ch_im_in;
            colCnt--;
          }

          *pOut++ = (int8_t) CLIP8(sum  >> out_shift);
          *pOut++ = (int8_t) CLIP8(sum2 >> out_shift);
          *pOut++ = (int8_t) CLIP8(sum3 >> out_shift);
          *pOut++ = (int8_t) CLIP8(sum4 >> out_shift);

          rowCnt--;
        }

        rowCnt = ch_im_out & 0x3;
        while(rowCnt)
        {

          int8_t 			*pB = colBuffer + row_shift;
          const int8_t 	*pA = wt + row_shift;
          int8_t     		sum = ((int)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
          uint16_t  		colCnt = (dim_kernel * dim_kernel);
          row_shift += 1;

          while (colCnt)
          {
            int8_t      A1 = *pA;
            int8_t      B1 = *pB;
            pA += ch_im_in;
            pB += ch_im_in;
            sum += A1 * B1;

            colCnt--;
          }
          *pOut++ = (int8_t) CLIP8(sum >> out_shift);
          rowCnt--;
        }

        /* clear counter and pointers */
        pBuffer = colBuffer;
      }
    }
    rt_team_barrier();
  }
