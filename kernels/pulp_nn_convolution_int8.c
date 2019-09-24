/*
* pulp_nn_convolution_int8.c
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
#define NN_ROUND(out_shift) 		   ((out_shift) ? (0x1 << (out_shift -1)) : (0))
#define MIN(a,b) 					         ((a)<(b)?(a):(b))
#define CLIP8(x)                   __builtin_pulp_clip(x,-128, 127)

/*degree of freedom: if defined the 4x2 MatMul kernel is used. */
/* default: 2x2 kernel. */
/* 4x2 recommended for best performance */
#define SIZE4x2KERNEL

#ifdef SIZE4x2KERNEL
int8_t __attribute__ ((noinline)) *pulp_nn_matmul_4x2_int8(
	int8_t * pWeight,
	int8_t * pInBuffer,
	uint16_t ch_im_out,
	uint16_t numCol_A,
	uint16_t bias_shift,
	uint16_t out_shift,
	int8_t * bias,
	int8_t * pOut);
	#else
int8_t __attribute__ ((noinline)) *pulp_nn_matmul_2x2_int8(
	int8_t * pWeight,
	int8_t * pInBuffer,
	uint16_t ch_im_out,
	uint16_t numCol_A,
	uint16_t bias_shift,
	uint16_t out_shift,
	int8_t * bias,
	int8_t * pOut);
	#endif

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
	* @brief INT8 convolution function
	* @param[in]       Im_in            pointer to the input feature map
	* @param[in]       dim_im_in      	input feature map spatial dimension
	* @param[in]       ch_im_in         number of IFM channels
	* @param[in]       wt               pointer to kernel weights
	* @param[in]       ch_im_out        number of filters, i.e., output tensor channels
	* @param[in]       dim_kernel     	filter kernel size
	* @param[in]       padding    		  padding size
	* @param[in]       stride         	convolution stride
	* @param[in]       bias             pointer to bias
	* @param[in]       bias_shift       amount of shift on bias
	* @param[in]       out_shift        amount of shift on output
	* @param[in,out]   Im_out           pointer to output feature map
	* @param[in]       dim_im_out     	output feature map spatial dimension
	* @param[in,out]   bufferC          pointer to buffer space for input (used for im2col)
	* @param[in,out]   bufferB          pointer to buffer space for output (not used)
	* @return          The function returns either.
	*/
	void __attribute__ ((noinline)) pulp_nn_convolution_int8(
		int8_t * Im_in,
		uint16_t dim_im_in,
		uint16_t ch_im_in,
		int8_t * wt,
		uint16_t ch_im_out,
		uint16_t dim_kernel,
		uint16_t padding,
		uint16_t stride,
		int8_t * bias,
		uint16_t bias_shift,
		uint16_t out_shift,
		int8_t * Im_out ,
		uint16_t dim_im_out ,
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
			start_pixel = chunck *  core_id;
			stop_pixel = MIN(start_pixel+chunck, dim_im_out);
			int8_t *pBuffer = bufferA;
			int8_t  *pOut    = Im_out + start_pixel * ch_im_out * dim_im_out;



			/* start kernel: this first phase is devoted to building the im2col buffers */
			for(int i_out_y=start_pixel; i_out_y<stop_pixel;i_out_y++)
			{
				for (int i_out_x=0; i_out_x < dim_im_out; i_out_x++)
				{
					/* top part */
					if(i_out_y < padding)
					{

						for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
						{
							for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
							{
								if ( i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
								{
									/* if padding needed, fill the im2col with zeros */
									pulp_zero_mem( pBuffer, ch_im_in);
								}
								else
								{
									/* 3d image like into 1d array transformation */
									pulp_nn_im2col_int8( (int8_t*) Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in, pBuffer, ch_im_in);
								}
								pBuffer += ch_im_in;
							}
						}
					}
					/* middle part */
					else if (i_out_y < dim_im_out - padding)
					{
						/*left side */
						if (i_out_x  < padding)
						{
							for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
							{
								for(i_ker_x = i_out_x * stride -padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
								{
									if (i_ker_x < 0 || i_ker_x > dim_im_in )
									{
										/* if padding needed, fill the im2col with zeros */
										pulp_zero_mem(pBuffer, ch_im_in);
									}
									else
									{
										/* 3d image like into 1d array transformation */
										pulp_nn_im2col_int8((int8_t*) Im_in +(i_ker_y * dim_im_in+ i_ker_x)* ch_im_in, pBuffer,ch_im_in);
									}
									pBuffer += ch_im_in;
								}
							}
						}
						/* center */
						else if ( i_out_x < dim_im_out - padding)
						{

							for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
							{
								/* 3d image like into 1d array transformation */
								pulp_nn_im2col_int8((int8_t*) Im_in + (i_ker_y * dim_im_in + i_out_x * stride - padding)*ch_im_in,pBuffer,ch_im_in * dim_kernel);
								pBuffer += ch_im_in * dim_kernel;
							}
						}
						/* right side */
						else
						{
							for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
							{
								for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
								{
									if (i_ker_x < 0 || i_ker_x >= dim_im_in)
									{
										/* if padding needed, fill the im2col with zeros */
										pulp_zero_mem ( pBuffer, ch_im_in);
									}
									else
									{
										/* 3d image like into 1d array transformation */
										pulp_nn_im2col_int8((int8_t *)Im_in+ (i_ker_y*dim_im_in+i_ker_x)* ch_im_in, pBuffer, ch_im_in);
									}
									pBuffer += ch_im_in;
								}
							}
						}
					}
					/* bottom part */
					else
					{
						/* This part implements the im2col function */
						for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
						{
							for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
							{
								if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
								{
									/* if padding needed, fill the im2col with zeros */
									pulp_zero_mem (pBuffer, ch_im_in);
								}
								else
								{
									/* 3d image like into 1d array transformation */
									pulp_nn_im2col_int8((int8_t *) Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in, pBuffer, ch_im_in);
								}
								pBuffer += ch_im_in;
							}
						}
					}

					/* when im2col buffers are built start the dot product computation */
					/* i.e. matrix multiplication kernel */
					if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
					{

					#ifdef SIZE4x2KERNEL
						pOut = pulp_nn_matmul_4x2_int8(
							wt,
							bufferA,
							ch_im_out,
							ch_im_in * dim_kernel * dim_kernel,
							bias_shift,
							out_shift,
							bias,
							pOut);
					#else
						pOut = pulp_nn_matmul_2x2_int8(
							wt,
							bufferA,
							ch_im_out,
							ch_im_in * dim_kernel * dim_kernel,
							bias_shift,
							out_shift,
							bias,
							pOut);
					#endif
						/* counter reset */
						pBuffer = bufferA;
					}
				}
			}

					/* if (pBuffer != bufferA) */
					/*   { */
					/*     int8_t *pA = wt; */
					/*     for (int i = 0; i < ch_im_out; i++) */
					/* 	{ */
					/* 	  int     sum = ((int)bias[i] << bias_shift) + NN_ROUND(out_shift); */
					/* 	  int8_t  *pB = bufferA; */

					/* 	  /\* each time it process 4 entries *\/ */
					/* 	  uint16_t  colCnt = (ch_im_in * dim_kernel * dim_kernel) >> 2; */

					/* 	  /\* accumulate over the vector *\/ */

					/* 	  for (int j=0; j < colCnt; j++) */
					/* 	    { */
					/* 	      sum  += __builtin_pulp_dotsp4( *((v4s *) pA), *((v4s *) pB) ); */
					/* 	      pB += 4; */
					/* 	      pA += 4; */
					/* 	    } */
					/* 	  colCnt = (ch_im_in * dim_kernel * dim_kernel) & 0x3; */
					/* 	  while (colCnt) */
					/* 	    { */
					/* 	      int8_t      inA1 = *pA++; */
					/* 	      int8_t      inB1 = *pB++; */
					/* 	      sum += inA1 * inB1; */
					/* 	      colCnt--; */
					/* 	    } */

					/* 	  *pOut = (int8_t)(sum);// CLIP8( (sum >> out_shift),8); */
					/* 	  pOut++; */
					/* 	} */
					/*   } */

				  // final synch barrier
					rt_team_barrier();
		}
