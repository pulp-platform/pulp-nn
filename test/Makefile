PULP_APP = main

PULP_APP_FC_SRCS  = main_test.c
PULP_APP_CL_OMP_SRCS = test_layers.c
PULP_APP_CL_OMP_SRCS += $(wildcard ./../kernels/*.c)


ifndef cores
cores=1
else
cores = $(cores)
endif

ifndef test
test= 1
else
test = $(test)  // 1:convolution, 2:fully_connected, 3:max_pooling, 4:relu (not available yet)
endif

ifndef Q
Q =8
else
Q = $(Q)
endif

ifeq ($(prof),1)
PULP_CFLAGS = -DPROFILING
endif

ifeq ($(check),1)
PULP_CFLAGS += -DCHECKLAYER
endif

PULP_CFLAGS += -DTEST=$(test) -DNUM_CORES=$(cores) -DQ$(Q) -O3  -Iinc -w
PULP_LDFLAGS = -lm 



include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
