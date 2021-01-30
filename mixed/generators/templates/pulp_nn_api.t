%if config.api == "PULPNNConvolve":
void ${config.fn_name}(
	const uint8_t *pInBuffer,
	const uint16_t dim_in_x,
	const uint16_t dim_in_y,
	const uint16_t ch_in,
	const int8_t *pWeight,
	const uint16_t ch_out,
	const uint16_t dim_kernel_x,
	const uint16_t dim_kernel_y,
	const uint16_t padding_y_top,
	const uint16_t padding_y_bottom,
	const uint16_t padding_x_left,
	const uint16_t padding_x_right,
	const uint16_t stride_x,
	const uint16_t stride_y,
	const int8_t *bias,
	const uint16_t bias_shift,
	const int8_t out_shift,
	const uint16_t out_mult,
	uint8_t *pOutBuffer,
	const uint16_t dim_out_x,
	const uint16_t dim_out_y,
%if config.kernel.quantization == 'shift_clip' and config.kernel.act_prec == '32bit':
	int32_t *k,
	int32_t *lambda,
%elif config.kernel.quantization == 'shift_clip' and config.kernel.act_prec == '64bit':
	int64_t *k,
	int64_t *lambda,
%else:
	int16_t *pThr,
%endif
	uint8_t *pIm2ColBuffer,
	int flag_relu,
	int flag_batch_norm,
	unsigned int * memory_chan);
%elif config.api == "PULPNNConvolvePointwise":
void ${config.fn_name}(
	const uint8_t *pInBuffer,
	const uint16_t dim_in_x,
	const uint16_t dim_in_y,
	const uint16_t ch_in,
	const int8_t *pWeight,
	const uint16_t ch_out,
	const uint16_t dim_kernel_x,
	const uint16_t dim_kernel_y,
	const uint16_t padding_y_top,
	const uint16_t padding_y_bottom,
	const uint16_t padding_x_left,
	const uint16_t padding_x_right,
	const uint16_t stride_x,
	const uint16_t stride_y,
	const int8_t *bias,
	const uint16_t bias_shift,
	const int8_t out_shift,
	const uint16_t out_mult,
	uint8_t *pOutBuffer,
	const uint16_t dim_out_x,
	const uint16_t dim_out_y,
%if config.kernel.quantization == 'shift_clip' and config.kernel.act_prec == '32bit':
	int32_t *k,
	int32_t *lambda,
%elif config.kernel.quantization == 'shift_clip' and config.kernel.act_prec == '64bit':
	int64_t *k,
	int64_t *lambda,
%else:
	int16_t *pThr,
%endif
	uint8_t *pIm2ColBuffer,
	int flag_relu,
	int flag_batch_norm,
	unsigned int * memory_chan);
% elif config.api=="PULPNNMatMul":
uint8_t *${config.fn_name}(
	const int8_t * pWeight,
	uint8_t * pInBuffer,
	uint16_t ch_out,
	uint16_t num_col_im2col,
	uint16_t bias_shift,
	int8_t out_shift,
	uint16_t out_mult,
%if config.kernel.quantization == 'shift_clip' and config.kernel.act_prec == '32bit':
	int32_t *k,
	int32_t *lambda,
%elif config.kernel.quantization == 'shift_clip' and config.kernel.act_prec == '64bit':
	int64_t *k,
	int64_t *lambda,
%else:
	int16_t *pThr,
%endif
	const int8_t * bias,
	uint8_t * pOut,
	int flag_relu,
	int flag_batch_norm);
% elif config.api=="PULPNNConvolveDepthwise":
void ${config.fn_name}(
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
%if config.kernel.quantization == 'shift_clip' and config.kernel.act_prec == '32bit':
	int32_t *k,
	int32_t *lambda,
%elif config.kernel.quantization == 'shift_clip' and config.kernel.act_prec == '64bit':
	int64_t *k,
	int64_t *lambda,
%endif
	uint8_t * pIm2ColBuffer,
	int8_t * pWtBuffer,
	int flag_relu,
	int flag_batch_norm,
	unsigned int * memory_chan);
%elif config.api=="PULPNNLinearNoQuant" or config.api=="PULPNNLinearQuant":
void ${config.fn_name}(
	uint8_t *pInBuffer,
	int8_t *pWeights,
	uint16_t dim_vec,
	uint16_t num_o_neurons,
	int8_t *bias,
	uint16_t bias_shift,
	int8_t out_shift,
	uint16_t out_mult,
%if config.kernel.act_prec == '32bit':
	int32_t *k,
	int32_t *lambda,
%elif config.kernel.act_prec == '64bit':
	int64_t *k,
	int64_t *lambda,
%endif
%if config.kernel.quantization == 'thresholds':
	int16_t *pThr,
%endif
	uint8_t *pOutBuffer,
	int flag_relu,
	int flag_batch_norm,
	unsigned int * memory_chan);
%elif config.api=="PULPNNMaxPool":
void ${config.fn_name}(
	uint8_t * Im_in,
	uint16_t  dim_im_in_x,
	uint16_t  dim_im_in_y,
	uint16_t  ch_im_in,
	uint16_t  dim_kernel,
	uint16_t  padding_t,
	uint16_t  padding_b,
	uint16_t  padding_l,
	uint16_t  padding_r,
	uint16_t  stride,
	uint16_t  dim_im_out_x,
	uint16_t  dim_im_out_y,
	uint8_t * Im_out,
	unsigned int * memory_chan);
%elif config.api=="PULPNNAvgPool":
void ${config.fn_name}(
	uint8_t * Im_in,
	uint16_t dim_im_in,
	uint16_t ch_im_in,
	uint16_t dim_kernel,
	uint16_t padding,
	uint16_t stride,
	uint16_t dim_im_out,
	uint8_t * Im_out,
	unsigned int * memory_chan);
%elif config.api=="PULPNNAdd":
void ${config.fn_name}(
	uint8_t * Im_in_1,    
	uint8_t * Im_in_2,    
	uint16_t  ch_im_in,   
	uint16_t  dim_im_in_h,
	uint16_t  dim_im_in_w,
	uint8_t * Im_out,     
	uint16_t out_mult1,   
	uint16_t out_mult2,   
	uint16_t out_shift    
);
%endif
