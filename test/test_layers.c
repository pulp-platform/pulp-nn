/*
 * test_layers.c
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
#include "stats.h"

#include "../include/pulp_nn.h"


/* INT-Q data representation  (select just one per time) */
//#define Q8
//#define Q4 //not available yet
//#define Q2 //not available yet
//#define Q1 //not available yet

/* define the layer to be tested (select just one per time) */
#if (TEST==1)
#define CONV_LAYER_TEST
#elif (TEST==2)
#define FULLY_LAYER_TEST
#elif (TEST==3)
#define MAX_POOLING_TEST
#else
#define RELU_TEST
#endif


/* layer parameters */
#ifdef CONV_LAYER_TEST
	#ifdef Q8
		#include "./layer_parameters/int8_conv_layer_parameters_corner_nosquare2.h"
	#endif
#endif

#ifdef MAX_POOLING_TEST
	#ifdef Q8
		#include "./layer_parameters/int8_maxpool_layer_parameters2.h"
	#endif
#endif

#ifdef FULLY_LAYER_TEST
	#ifdef Q8
		#include "./layer_parameters/int8_fully_layer_parameters.h"
	#endif
#endif

/* data allocation */
#include "test_data_allocation.h"


int test_layers(rt_perf_t *perf)
{

#ifdef Q4
	printf("Selected data type not supported yet\n");
	return(-1);
#endif
#ifdef Q2
	printf("Selected data type not supported yet\n");
	return(-1);
#endif
#ifdef Q1
	printf("Selected data type not supported yet\n");
	return(-1);
#endif

/*convolution layer test */
#ifdef CONV_LAYER_TEST

	if(rt_core_id()==0)
	{
		/* transfer layer weights and activations from L2 to L1 memory through DMA */
		rt_dma_copy_t cp1;
		rt_dma_copy_t cp2;

		printf("activations transfer from L2 to L1\n");

    	rt_dma_memcpy(
#ifdef Q8
			input_data_int8_L2, // ext
			input_data_int8_L1, // loc
			IFM_W*IFM_H*IFM_CH, // size
#endif
			RT_DMA_DIR_EXT2LOC, // dir
			0, // merge
			&cp1 // copy
			);
    	rt_dma_wait(&cp1);

		printf("weights transfer from L2 to L1\n");
    	rt_dma_memcpy(
#ifdef Q8
			conv_wt_int8_L2, // ext
			conv_wt_int8_L1, // loc
			IFM_CH*DIM_KER*DIM_KER*OFM_CH, // size
#endif
			RT_DMA_DIR_EXT2LOC, // dir
			0, // merge
			&cp2 // copy
			);
    	rt_dma_wait(&cp2);

#ifdef Q8
    	//BIAS
    	for (int k= 0; k < OFM_CH; k++)
		{
			conv_bias_int8_L1[k] = 0;
		}
#endif

		printf("end of transfer. Start Conv kernel parallel exec \n");
	}

	rt_team_barrier();

	/* Convolution layer execution */
#ifdef PROFILING
	/* These functions allow for a complete profiling of the kernel execution */
	/* Defined in "stats.h" header. See file for complete info                */
	INIT_PROFILING();
	START_PROFILING();
#endif

	rt_team_barrier();

#ifdef Q8
	pulp_nn_convolution_nosquare_asympad_int8(input_data_int8_L1, IFM_W, IFM_H, IFM_CH,conv_wt_int8_L1, OFM_CH, DIM_KER, DIM_KER, PADDING, PADDING, PADDING, PADDING,
															STRIDE, STRIDE, conv_bias_int8_L1, BIAS_SHIFT, QUANT_FACTOR, conv_int8_out_L1, OFM_W, OFM_H,col_buffer, NULL);
	//pulp_nn_convolution_int8(input_data_int8_L1, IFM_H, IFM_CH, conv_wt_int8_L1, OFM_CH, DIM_KER, PADDING,
	//										STRIDE, conv_bias_int8_L1, BIAS_SHIFT, QUANT_FACTOR, conv_int8_out_L1, OFM_H, col_buffer, NULL,NULL);
	//pulp_nn_dw_convolution_int8(input_data_int8_L1, IFM_W, IFM_CH, conv_wt_int8_L1,OFM_CH, DIM_KER,PADDING,STRIDE, conv_bias_int8_L1,BIAS_SHIFT, QUANT_FACTOR,
	//													conv_int8_out_L1, OFM_W,col_buffer, NULL);
#endif

#ifdef PROFILING
	STOP_PROFILING();
#endif

	rt_team_barrier();

#ifdef CHECKLAYER

	if(rt_core_id()==0)
	{
		rt_dma_copy_t cp3;
		rt_dma_memcpy(
#ifdef Q8
			conv_int8_out_L2, // ext
			conv_int8_out_L1, // loc
			OFM_W * OFM_H * OFM_CH, // size
#endif
			RT_DMA_DIR_LOC2EXT, // dir
			0, // merge
			&cp3 // copy
			);
	    rt_dma_wait(&cp3);

	    int errors=0;
	    for (int i=0; i< OFM_CH * OFM_W * OFM_H; i++)
	    {

	    	if(conv_int8_out_L2[i] != checksum_conv_int8[i])
	    	{
	    		printf("exp: %X, real: %X, index: %d\n",checksum_conv_int8[i], conv_int8_out_L2[i],i );
	    		errors ++;
	    	}
	    }

	    if(errors!=0)
	    	printf("check failed. number of errors: %d \n", errors);
	    else
	    	printf("check ok. The layer has been tested successfully. \n");
	}
#endif

#endif







#ifdef MAX_POOLING_TEST

	if(rt_core_id()==0)
	{
		/* transfer layer activations from L2 to L1 memory through DMA */
		rt_dma_copy_t cp1;

		printf("activations transfer from L2 to L1\n");
    	rt_dma_memcpy(
#ifdef Q8
			maxpool_data_int8_L2, // ext
			maxpool_data_int8_L1, // loc
			IFM_H_MP*IFM_H_MP*IFM_CH_MP, // size
#endif
			RT_DMA_DIR_EXT2LOC, // dir
			0, // merge
			&cp1 // copy
			);
    	rt_dma_wait(&cp1);
		printf("end of transfer. Start max Pooling kernel parallel exec \n");
	}
	rt_team_barrier();

	/* Max Pooling layer execution */
#ifdef PROFILING
	/* These functions allow for a complete profiling of the kernel execution */
	/* Defined in "stats.h" header. See file for complete info                */
	INIT_PROFILING();
	START_PROFILING();
#endif

	rt_team_barrier();

#ifdef Q8
	pulp_nn_max_pooling_int8_nosquare(maxpool_data_int8_L1,IFM_H_MP, IFM_H_MP, IFM_CH_MP, DIM_KER_MP, PADDING_MP,
								STRIDE_MP, OFM_H_MP, OFM_H_MP, NULL, maxpool_out_int8_L1);
#endif

#ifdef PROFILING
	STOP_PROFILING();
#endif


#ifdef CHECKLAYER

	if(rt_core_id()==0)
	{
		rt_dma_copy_t cp3;
		rt_dma_memcpy(
#ifdef Q8
			maxpool_out_int8_L2, // ext
			maxpool_out_int8_L1, // loc
			OFM_H_MP * OFM_H_MP * OFM_CH_MP, // size
#endif
			RT_DMA_DIR_LOC2EXT, // dir
			0, // merge
			&cp3 // copy
			);
	    rt_dma_wait(&cp3);

	    int errors=0;
	    for (int i=0; i< OFM_CH_MP * OFM_H_MP * OFM_H_MP; i++)
	    {

	    	if(maxpool_out_int8_L2[i] != checksum_maxpool_int8[i])
	    	{
	    		printf("exp: %X, real: %X, index: %d\n",checksum_maxpool_int8[i], maxpool_out_int8_L2[i],i );
	    		errors ++;
	    	}
	    }
	    if(errors!=0)
	    	printf("check failed. number of errors: %d \n", errors);
	    else
	    	printf("check ok. The layer has been tested successfully. \n");
	}

#endif
#endif



/*Fully-connected layer test */
#ifdef FULLY_LAYER_TEST
	if(rt_core_id()==0)
	{
		/* transfer layer weights and activations from L2 to L1 memory through DMA */
		rt_dma_copy_t cp1;
		rt_dma_copy_t cp2;

		printf("activations transfer from L2 to L1\n");
    	rt_dma_memcpy(
#ifdef Q8
			input_data_fully_int8_L2, // ext
			input_data_fully_int8_L1, // loc
			IFM_H_FC * IFM_H_FC *IFM_CH_FC, // size
#endif
			RT_DMA_DIR_EXT2LOC, // dir
			0, // merge
			&cp1 // copy
			);
    	rt_dma_wait(&cp1);

		printf("weights transfer from L2 to L1\n");
    	rt_dma_memcpy(
#ifdef Q8
			fully_wt_int8_L2, // ext
			fully_wt_int8_L1, // loc
			OUT_NEURONS * IFM_H_FC * IFM_H_FC *IFM_CH_FC, // size
#endif
			RT_DMA_DIR_EXT2LOC, // dir
			0, // merge
			&cp2 // copy
			);
    	rt_dma_wait(&cp2);

#ifdef Q8
    	//BIAS
    	for (int k= 0; k < OUT_NEURONS; k++)
		{
			fully_bias_int8_L1[k] = 0;
		}
#endif
	printf("end of transfer. Start Fully-connected kernel parallel exec \n");
	}
	rt_team_barrier();

	/* Execution of FC layer */

#ifdef PROFILING
	/* These functions allow for a complete profiling of the kernel execution */
	/* Defined in "stats.h" header. See file for complete info                */
	INIT_PROFILING();
	START_PROFILING();
#endif

	rt_team_barrier();

#ifdef Q8
	pulp_nn_linear_int8(input_data_fully_int8_L1, fully_wt_int8_L1, IFM_H_FC * IFM_H_FC * IFM_CH_FC,
									OUT_NEURONS, BIAS_SHIFT_FC, OUT_QF_FC, fully_bias_int8_L1, fully_int8_out_L1);
#endif

#ifdef PROFILING
	STOP_PROFILING();
#endif


#ifdef CHECKLAYER
	if(rt_core_id()==0)
	{
		rt_dma_copy_t cp3;
		rt_dma_memcpy(
		#ifdef Q8
			fully_int8_out_L2, // ext
			fully_int8_out_L1, // loc
			OUT_NEURONS, // size
		#endif
			RT_DMA_DIR_LOC2EXT, // dir
			0, // merge
			&cp3 // copy
			);
	    rt_dma_wait(&cp3);

	    int errors=0;
	    for (int i=0; i< OUT_NEURONS; i++)
	    {
	    	if(fully_int8_out_L2[i] != checksum_fully_int8[i])
	    	{
	    		printf("exp: %X, real: %X, index: %d\n",checksum_fully_int8[i], fully_int8_out_L2[i],i );
	    		errors ++;
	    	}
	    }

	    if(errors!=0)
	    	printf("check failed. number of errors: %d \n", errors);
	    else
	    	printf("check ok. The layer has been tested successfully. \n");
	}
#endif

#endif

return(0);
}
