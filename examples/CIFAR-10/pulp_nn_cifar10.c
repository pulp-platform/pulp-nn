/*
 * pulp_nn_cifar10.c
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

#include "cifar10_inputs.h"
#include "cifar10_weights.h"
#include "cifar10_parameter.h"
#include "cifar10_data_allocation.h"
#include "../../include/pulp_nn.h"
#include "checksum_layer.h"
#include "stats.h"

#define CLIP8(x) 			__builtin_pulp_clip(x,-128, 127)

void pulp_nn_cifar10(rt_perf_t *perf)
{

	/* first layer: convolution + relu + max pooling */
	int8_t     *img_buffer2 = L1_buffer;  				// input of conv1
  	int8_t     *img_buffer1 = img_buffer2 + 8192;
  	int8_t     *weights_l1  = img_buffer1+ 32768;
  	int8_t     *bias_l1     = weights_l1 + 3200 ;
  	uint8_t    *cifar_input = image_data;

  	int mean_data[3] = INPUT_MEAN_SHIFT;
  	unsigned int scale_data[3] = INPUT_RIGHT_SHIFT;
  	int sumcheck;


  	if(rt_core_id()==0)
    {
    	for (int i=0;i<32*32*4; i+=4)
      	{
			img_buffer2[i] =   (int8_t) CLIP8( ((((int) *(cifar_input++) - mean_data[0])<<7) + (0x1<<(scale_data[0]-1)))
					 >> scale_data[0]);
			img_buffer2[i+1] = (int8_t) CLIP8( ((((int) *(cifar_input++) - mean_data[1])<<7) + (0x1<<(scale_data[1]-1)))
					  >> scale_data[1]);
			img_buffer2[i+2] = (int8_t) CLIP8( ((((int) *(cifar_input++) - mean_data[2])<<7) + (0x1<<(scale_data[2]-1)))
					 >> scale_data[2]);
			img_buffer2[i+3] = 0; // creating a ghost channel
      	}

      	/* transfer weights */
      	rt_dma_copy_t cp1;
      	rt_dma_memcpy(
		conv1_wt, // ext
		weights_l1, // loc
		CONV1_IN_CH * CONV1_KER_DIM * CONV1_KER_DIM * CONV1_OUT_CH, // size
		RT_DMA_DIR_EXT2LOC, // dir
		0, // merge
		&cp1 // copy
		);
    	rt_dma_wait(&cp1);

    	/*transfer bias */
    	for (int i=0; i<CONV1_OUT_CH; i++)
		{
	  		bias_l1[i] = conv1_bias[i];
		}
	}

	rt_team_barrier();

#ifdef PROFILING
	rt_perf_t perf2;
	rt_perf_init(&perf2);
	rt_perf_conf(&perf2, (1<<RT_PERF_CYCLES));
	rt_perf_reset(&perf2);
    rt_perf_stop(&perf2);
    rt_perf_start(&perf2);
	rt_team_barrier();
#endif

  	pulp_nn_convolution_int8(
				  						img_buffer2,
				  						CONV1_IN_DIM,
				  						CONV1_IN_CH,
				  						weights_l1,
				  						CONV1_OUT_CH,
				  						CONV1_KER_DIM,
				  						CONV1_PAD,
				  						CONV1_STRIDE,
				  						bias_l1,
				  						CONV1_BIAS_LSHIFT,
				  						CONV1_OUT_RSHIFT,
				  						img_buffer1,
				  						CONV1_OUT_DIM,
				  						col_buffer,
				  						NULL);

#ifdef CHECKLAYER
  	if(rt_core_id()==0){
  	sumcheck = checksum( img_buffer1, CONV1_OUT_DIM * CONV1_OUT_DIM * CONV1_OUT_CH );
  	printf("checksum of conv1 layer: %X \n", sumcheck );}
  	rt_team_barrier();
#endif

	// RELU 1
  	pulp_nn_relu_int8( img_buffer1, CONV1_OUT_DIM, CONV1_OUT_CH);

#ifdef CHECKLAYER
  	if(rt_core_id()==0){
  	sumcheck = checksum( img_buffer1, CONV1_OUT_DIM * CONV1_OUT_DIM * CONV1_OUT_CH );
  	printf("checksum of relu1 layer: %X \n", sumcheck );}
  	rt_team_barrier();
#endif

  	// MAX POOLING 1
  	pulp_nn_max_pooling_int8(
  								img_buffer1,
								CONV1_OUT_DIM,
								CONV1_OUT_CH,
								POOL1_KER_DIM,
								POOL1_PAD,
								POOL1_STRIDE,
								POOL1_OUT_DIM,
								col_buffer,
								img_buffer2
								);

#ifdef CHECKLAYER
  	if(rt_core_id()==0){
  	sumcheck = checksum( img_buffer2, POOL1_OUT_DIM * POOL1_OUT_DIM * CONV1_OUT_CH );
  	printf("checksum of mpool1 layer: %X \n", sumcheck );}
  	rt_team_barrier();
#endif

  	/* second layer */
  	img_buffer2 = L1_buffer;  //input of conv0
  	img_buffer1 = img_buffer2 + 8192;
  	weights_l1  = img_buffer1+ 4096;
  	bias_l1     = weights_l1 + 12800;

  	if(rt_core_id()==0)
    {
    	/* transfer weights */
      	rt_dma_copy_t cp1;
      	rt_dma_memcpy(
		conv2_wt, // ext
		weights_l1, // loc
		CONV2_IN_CH * CONV2_KER_DIM * CONV2_KER_DIM * CONV2_OUT_CH, // size
		RT_DMA_DIR_EXT2LOC, // dir
		0, // merge
		&cp1 // copy
		);
    	rt_dma_wait(&cp1);

    	/* transfer bias */
    	for (int i=0; i<CONV2_OUT_CH; i++)
		{
	  		bias_l1[i] = conv2_bias[i];
		}
	}

	rt_team_barrier();
    pulp_nn_convolution_int8(
				     				img_buffer2,
				     				CONV2_IN_DIM,
				     				CONV2_IN_CH,
				     				weights_l1,
				     				CONV2_OUT_CH,
				     				CONV2_KER_DIM,
				     				CONV2_PAD,
				     				CONV2_STRIDE,
				     				bias_l1,
				     				CONV2_BIAS_LSHIFT,
				     				CONV2_OUT_RSHIFT,
				     				img_buffer1,
				     				CONV2_OUT_DIM,
				     				col_buffer,
				     				NULL);
#ifdef CHECKLAYER
  	if(rt_core_id()==0){
  	sumcheck = checksum( img_buffer1, CONV2_OUT_DIM * CONV2_OUT_DIM * CONV2_OUT_CH );
  	printf("checksum of conv2 layer: %X \n", sumcheck );}
  	rt_team_barrier();
#endif

    pulp_nn_relu_int8( img_buffer1, CONV2_OUT_DIM, CONV2_OUT_CH);

#ifdef CHECKLAYER
  	if(rt_core_id()==0){
  	sumcheck = checksum( img_buffer1, CONV2_OUT_DIM * CONV2_OUT_DIM * CONV2_OUT_CH );
  	printf("checksum of relu2 layer: %X \n", sumcheck );}
  	rt_team_barrier();
#endif

	pulp_nn_max_pooling_int8(
								img_buffer1,
				 				CONV2_OUT_DIM,
				 				CONV2_OUT_CH,
				 				POOL2_KER_DIM,
				 				POOL2_PAD,
				 				POOL2_STRIDE,
				 				POOL2_OUT_DIM,
				 				NULL,
				 				img_buffer2
				 				);

#ifdef CHECKLAYER
  	if(rt_core_id()==0){
  	sumcheck = checksum( img_buffer2, POOL2_OUT_DIM * POOL2_OUT_DIM * CONV2_OUT_CH );
  	printf("checksum of mpool2 layer: %X \n", sumcheck );}
  	rt_team_barrier();
#endif

	/* third layer */
	img_buffer2 = L1_buffer;
   	img_buffer1 = img_buffer2 + 1024;
   	weights_l1  = img_buffer1+ 2048;
   	bias_l1     = weights_l1 +3200;

   	if(rt_core_id()==0)
    {
    	/* transfer weights */
      	rt_dma_copy_t cp1;
      	rt_dma_memcpy(
		conv3_wt, // ext
		weights_l1, // loc
		CONV3_IN_CH * CONV3_KER_DIM * CONV3_KER_DIM * CONV3_OUT_CH, // size
		RT_DMA_DIR_EXT2LOC, // dir
		0, // merge
		&cp1 // copy
		);
    	rt_dma_wait(&cp1);

    	/* transfer bias */
    	for (int i=0; i<CONV3_OUT_CH; i++)
		{
	  		bias_l1[i] = conv3_bias[i];
		}
	}

	rt_team_barrier();
   	pulp_nn_convolution_int8(
				   					img_buffer2,
				   					CONV3_IN_DIM,
				   					CONV3_IN_CH,
				   					weights_l1,
				   					CONV3_OUT_CH,
				   					CONV3_KER_DIM,
				   					CONV3_PAD,
				   					CONV3_STRIDE,
				   					bias_l1,
				   					CONV3_BIAS_LSHIFT,
				   					CONV3_OUT_RSHIFT,
				   					img_buffer1,
				   					CONV3_OUT_DIM,
				   					col_buffer,
				   					NULL);

#ifdef CHECKLAYER
  	if(rt_core_id()==0){
  	sumcheck = checksum( img_buffer1, CONV3_OUT_DIM * CONV3_OUT_DIM * CONV3_OUT_CH );
  	printf("checksum of conv3 layer: %X \n", sumcheck );}
  	rt_team_barrier();
#endif

   	pulp_nn_relu_int8( img_buffer1, CONV3_OUT_DIM, CONV3_OUT_CH);

#ifdef CHECKLAYER
  	if(rt_core_id()==0){
  	sumcheck = checksum( img_buffer1, CONV3_OUT_DIM * CONV3_OUT_DIM * CONV3_OUT_CH );
  	printf("checksum of relu3 layer: %X \n", sumcheck );}
  	rt_team_barrier();
#endif

   	pulp_nn_max_pooling_int8(
   								img_buffer1,
				   				CONV3_OUT_DIM,
				   				CONV3_OUT_CH,
				   				POOL3_KER_DIM,
				   				POOL3_PAD,
				   				POOL3_STRIDE,
				   				POOL3_OUT_DIM,
				   				NULL,
				   				img_buffer2
				   				);

#ifdef CHECKLAYER
  	if(rt_core_id()==0){
  	sumcheck = checksum( img_buffer2, POOL3_OUT_DIM * POOL3_OUT_DIM * CONV3_OUT_CH );
  	printf("checksum of mpool3 layer: %X \n", sumcheck );}
  	rt_team_barrier();
#endif

   	/* fully connected layer */
   	img_buffer2 = L1_buffer;
    img_buffer1 = img_buffer2 + 512;
    weights_l1  = img_buffer1 + 10;
    bias_l1     = weights_l1  + 5120;

    if(rt_core_id()==0)
    {
    	/* transfer weights */
      	rt_dma_copy_t cp1;
      	rt_dma_memcpy(
		ip1_wt, // ext
		weights_l1, // loc
		IP1_IN_DIM * IP1_OUT_DIM, // size
		RT_DMA_DIR_EXT2LOC, // dir
		0, // merge
		&cp1 // copy
		);
    	rt_dma_wait(&cp1);

    	/* transfer bias */
    	for (int i=0; i<IP1_OUT_DIM; i++)
		{
	  		bias_l1[i] = ip1_bias[i];
		}
	}

	rt_team_barrier();
    pulp_nn_linear_int8(img_buffer2,weights_l1,IP1_IN_DIM,IP1_OUT_DIM,IP1_BIAS_LSHIFT,IP1_OUT_RSHIFT,bias_l1,img_buffer1);

#ifdef PROFILING
	 rt_perf_stop(&perf2);
	 rt_perf_save(&perf2);
	 int cid = rt_core_id();
	 printf("[%d] : num_cycles: %d\n",cid,rt_perf_get(&perf2, RT_PERF_CYCLES) );
#endif
#ifdef CHECKLAYER
  	if(rt_core_id()==0){
  	sumcheck = checksum( img_buffer1, IP1_OUT_DIM);
  	printf("checksum of fully connected layer: %X \n", sumcheck );}
  	rt_team_barrier();
#endif

    /*RESULTS*/

    if(rt_core_id()==0)
    {
		/*transfer the result to L2 */
	 	for (int i = 0 ; i< IP1_OUT_DIM; i++)
	   	{
	    	output_data[i] = img_buffer1[i];
	     	printf("value %d : %d \n", i, output_data[i]);
	   	}
    }

}
