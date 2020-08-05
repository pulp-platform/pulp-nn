PULP_APP = main

PULP_APP_FC_SRCS = main.c

PULP_APP_FC_SRCS += src/SupportFunctions/pulp_nn_utils.c

PULP_APP_FC_SRCS += src/DepthwiseConvolutions/pulp_nn_dw_u8_u8_i8.c

PULP_APP_FC_SRCS += src/DepthwiseConvolutions/pulp_nn_dw_u8_u8_i8.c



ifndef cores
cores=1
else
cores = $(cores)
endif

ifndef kernel
kernel=888
else
kernel = $(kernel)
endif

PULP_CFLAGS = -DPROFILING
PULP_CFLAGS += -DNUM_CORES=$(cores) -DKERNEL=$(kernel) -O3 -Iinclude -w -flto
PULP_LDFLAGS = -lm -flto

include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
