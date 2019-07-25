#ifndef __PULP_NN_LIB__
#define __PULP_NN_LIB__
#endif

/* prototypes of multicore nn functions */

/* Convolution functions */
/* INT-8 convolution kernel (multicores) */
void __attribute__ ((noinline)) pulp_nn_convolution_int8(
								int8_t * Im_in,					// pointer to IFM
								uint16_t dim_im_in, 			// IFM spatial dim
								uint16_t ch_im_in,				// IFM number of channels
								int8_t * wt,					// pointer to filter weights
								uint16_t ch_im_out,				// OFM number of channels
								uint16_t dim_kernel,			// spatial dimension of filters
								uint16_t padding,				// amount of padding
								uint16_t stride,				// amount of stride
								int8_t * bias,					// pointer to bias
								uint16_t bias_shift,			// amount of bias shift
								uint16_t out_shift,				// amount of out shift
								int8_t * Im_out ,				// pointer to OFM
								uint16_t dim_im_out ,			// OFM spatial dimension
								int8_t * bufferC,				// auxiliary buffer (used for im2col)
								int8_t * bufferB);				// auxiliary buffer (not used)

/* INT-8 convolution nosquare kernel */
void __attribute__ ((noinline)) pulp_nn_convolution_nosquare_int8(const int8_t * Im_in,
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
                                                                  int8_t * bufferB);

void __attribute__ ((noinline)) pulp_nn_convolution_nosquare_asympad_int8(const int8_t * Im_in,
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
                                                                  int8_t * bufferB);
void pulp_nn_dw_convolution_int8    	(const int8_t * Im_in,
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
                                            int8_t * bufferB);
/* Fully connected functions */
/* INT-8 Fully-Connected kernel (multi-core) */
void pulp_nn_linear_int8	(
				    			int8_t *pIn,					// pointer to IFM
				     			int8_t *pWeights,				// pointer to filter weights
				     			uint16_t dim_vec,				// IFM size: ch_in x dim_in x dim_in
				     			uint16_t num_o_neurons,			// number of output neurons
				     			uint16_t bias_shift,			// amount of bias shift
				     			uint16_t out_shift,				// amount of out shift
				     			int8_t *bias,					// pointer to bias
				     			int8_t *pOut);					// pointer to OFM

/* Activation functions */
/* INT-8 ReLu function (multi-core) */
void pulp_nn_relu_int8(
                  				int8_t * data,					// pointer to FM
		              		    uint16_t dim_im_in,				// spatial dimension of FM
			            	    uint16_t ch_im_in);				// number of FM channels

/* Pooling functions */
/* INT-8 Max Pooling kernel (multi-core) */
void __attribute__ ((noinline))  pulp_nn_max_pooling_int8(
				   	            int8_t * Im_in,                 // pointer to the input feature map
				   	            uint16_t dim_im_in,             // spatial dimension of the input feature map
				   	            uint16_t ch_im_in,              // number of channels of the IFM
				   	            uint16_t dim_kernel,            // spatial dimension of the pooling filter
				   	            uint16_t padding,               // amount of padding
				   	            uint16_t stride,                // amount of stride
				   	            uint16_t dim_im_out,            // reduced spatial dimension of output
				   	            int8_t * bufferA,               // actually not used in this fx
				   	            int8_t * Im_out                 // pointer to the output
				   	            );

void __attribute__ ((noinline))  pulp_nn_max_pooling_int8_nosquare(
				   	            int8_t * Im_in,                         // pointer to the input feature map
				   	            uint16_t dim_im_in_x,                     // spatial dimension of the input feature map
				   	            uint16_t dim_im_in_y,
                        		uint16_t ch_im_in,                      // number of channels of the IFM
				   	            uint16_t dim_kernel,                    // spatial dimension of the pooling filter
				   	            uint16_t padding,                       // amount of padding
				   	            uint16_t stride,                        // amount of stride
				   	            uint16_t dim_im_out_x,                    // reduced spatial dimension of output
				   	            uint16_t dim_im_out_y,
                        		int8_t * bufferA,                       // actually not used in this fx
				   	            int8_t * Im_out                         // pointer to the output
				   	            );
