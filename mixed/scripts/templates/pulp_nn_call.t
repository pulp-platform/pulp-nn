%if config.api == 'PULPNNConvolve':
#if (KERNEL == ${config.kernel.in_data_t}${config.kernel.out_data_t}${config.kernel.wt_data_t})
${config.fn_name}(IN_INT8_L1,
					DIM_IM_IN_X,
					DIM_IM_IN_Y,
					CH_IM_IN,
					WEIGHT_INT8_L1,
					CH_IM_OUT,
					DIM_KERNEL_X,
					DIM_KERNEL_Y,
					PADDING_Y_TOP,
					PADDING_Y_BOTTOM,
					PADDING_X_LEFT,
					PADDING_X_RIGHT,
					STRIDE_X,
					STRIDE_Y,
					BIAS_L1,
					BIAS_SHIFT,
          			OUT_SHIFT,
          			OUT_MULT,
					OUT_L1,
					DIM_IM_OUT_X,
					DIM_IM_OUT_Y,
%if config.kernel.quantization == 'shift_clip':
          			KAPPA_L1,
        			LAMBDA_L1,
%else:
%if config.kernel.out_data_t == 2:
          			THR_INT2_L1
%elif config.kernel.out_data_t == 4:
					THR_INT4_L1,
%endif
%endif
					IM2COL_L1,
%if config.layer.bn == True:
          			1,
%else:
					0,
%endif
%if config.layer.relu == True:
          			1,
%else:
					0,
%endif
        			NULL);
#endif
%elif config.api == 'PULPNNConvolvePointwise':
#if (KERNEL == ${config.kernel.in_data_t}${config.kernel.out_data_t}${config.kernel.wt_data_t})
${config.fn_name}(IN_INT8_L1,
					DIM_IM_IN_X,
					DIM_IM_IN_Y,
					CH_IM_IN,
					WEIGHT_INT8_L1,
					CH_IM_OUT,
					DIM_KERNEL_X,
					DIM_KERNEL_Y,
					PADDING_Y_TOP,
					PADDING_Y_BOTTOM,
					PADDING_X_LEFT,
					PADDING_X_RIGHT,
					STRIDE_X,
					STRIDE_Y,
					BIAS_L1,
					BIAS_SHIFT,
          			OUT_SHIFT,
          			OUT_MULT,
					OUT_L1,
					DIM_IM_OUT_X,
					DIM_IM_OUT_Y,
%if config.kernel.quantization == 'shift_clip':
          			KAPPA_L1,
        			LAMBDA_L1,
%else:
%if config.kernel.out_data_t == 2:
          			THR_INT2_L1
%elif config.kernel.out_data_t == 4:
					THR_INT4_L1,
%endif
%endif
					IM2COL_L1,
%if config.layer.bn == True:
          			1,
%else:
					0,
%endif
%if config.layer.relu == True:
          			1,
%else:
					0,
%endif
        			NULL);
#endif
%elif config.api == 'PULPNNConvolveDepthwise':
#if (KERNEL == ${config.kernel.in_data_t}${config.kernel.out_data_t}${config.kernel.wt_data_t})
${config.fn_name}(IN_INT8_L1_CHW,
					DIM_IM_IN_X,
					DIM_IM_IN_Y,
					CH_IM_IN,
					WEIGHT_INT8_L1_CHW,
					CH_IM_OUT,
					DIM_KERNEL_X,
					DIM_KERNEL_Y,
					PADDING_Y_TOP,
					PADDING_Y_BOTTOM,
					PADDING_X_LEFT,
					PADDING_X_RIGHT,
					STRIDE_X,
					STRIDE_Y,
					BIAS_L1,
					BIAS_SHIFT,
          			OUT_SHIFT,
          			OUT_MULT,
					OUT_L1,
					DIM_IM_OUT_X,
					DIM_IM_OUT_Y,
%if config.kernel.quantization == 'shift_clip':
          			KAPPA_L1,
          			LAMBDA_L1,
%endif
					IM2COL_L1,
					WTBUFF_L1,
%if config.layer.bn == True:
          			1,
%else:
					0,
%endif
%if config.layer.relu == True:
          			1,
%else:
					0,
%endif
          			NULL);
#endif
%elif config.api=="PULPNNMatMul":
#if (KERNEL == ${config.kernel.out_data_t}${config.kernel.wt_data_t})
OUT_L1 = ${config.fn_name}(
    				WEIGHT_INT8_L1,
    				IN_INT8_L1,
				    CH_IM_OUT,
				    (CH_IM_OUT * DIM_KERNEL_X * DIM_KERNEL_Y) << 1,
					BIAS_SHIFT,
					OUT_SHIFT,
					OUT_MULT,
				    KAPPA_L1,
				    LAMBDA_L1,
					BIAS_L1,
				    OUT_L1,
%if config.layer.bn == True:
          			1,
%else:
					0,
%endif
%if config.layer.relu == True:
          			1,
%else:
					0,
%endif);
#endif
%elif config.api == 'PULPNNLinearNoQuant':
#if (KERNEL == ${config.kernel.in_data_t}${config.kernel.wt_data_t})
${config.fn_name}(
				 	IN_INT8_L1,
					WEIGHT_INT8_L1,
					DIM_IM_IN_X*DIM_IM_IN_Y*CH_IM_IN,
					CH_IM_OUT,
					BIAS_L1,
					BIAS_SHIFT,
					OUT_SHIFT,
		      		OUT_MULT,
					KAPPA_L1,
					LAMBDA_L1,
					OUT_L1,
%if config.layer.bn == True:
          			1,
%else:
					0,
%endif
%if config.layer.relu == True:
          			1,
%else:
					0,
%endif
		      		NULL);
#endif
%elif config.api == 'PULPNNLinearQuant':
#if (KERNEL == ${config.kernel.in_data_t}${config.kernel.out_data_t}${config.kernel.wt_data_t})
${config.fn_name}(
				 	IN_INT8_L1,
					WEIGHT_INT8_L1,
					DIM_IM_IN_X*DIM_IM_IN_Y*CH_IM_IN,
					CH_IM_OUT,
					BIAS_L1,
					BIAS_SHIFT,
					OUT_SHIFT,
		      		OUT_MULT,
					KAPPA_L1,
					LAMBDA_L1,
					OUT_L1,
%if config.layer.bn == True:
          			1,
%else:
					0,
%endif
%if config.layer.relu == True:
          			1,
%else:
					0,
%endif
		      		NULL);
#endif
%elif config.api == 'PULPNNMaxPool':
#if (KERNEL == ${config.kernel.in_data_t})
${config.fn_name}(
					IN_INT8_L1,
					DIM_IM_IN_X,
					DIM_IM_IN_Y,
					CH_IM_IN,
					POOL_KERNEL,
					PADDING_Y_TOP,
					PADDING_Y_BOTTOM,
					PADDING_X_LEFT,
					PADDING_X_RIGHT,
					POOL_STRIDE,
					DIM_IM_OUT_X,
					DIM_IM_OUT_Y,
					OUT_L1,
				    NULL);
#endif
%elif config.api == 'PULPNNAvgPool':
#if (KERNEL == ${config.kernel.in_data_t})
${config.fn_name}(
					IN_INT8_L1,
					DIM_IM_IN_X,
					CH_IM_IN,
					POOL_KERNEL,
					PADDING_Y_TOP,
					POOL_STRIDE,
					DIM_IM_OUT_X,
					OUT_L1,
				    NULL);
#endif
%elif config.api == 'PULPNNAdd':
#if (KERNEL == ${config.in1_data_t}${config.in2_data_t})
${config.fn_name}(
					IN1_INT8_L1,
					IN2_INT8_L1,
					CH_IM_IN,
					DIM_IM_IN_X,
					DIM_IM_IN_Y,
					OUT_L1,
					OUT_MULT1,
					OUT_MULT2,
					OUT_SHIFT);
#endif
%endif
