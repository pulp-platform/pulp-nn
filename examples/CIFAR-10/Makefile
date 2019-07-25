PULP_APP = cifar10

PULP_APP_FC_SRCS  = cifar10_main.c
PULP_APP_CL_OMP_SRCS = pulp_nn_cifar10.c
PULP_APP_CL_OMP_SRCS += $(wildcard ./../../kernels/*.c)
#PULP_APP_CL_OMP_SRCS += $(wildcard ./../../kernels/activations/*.c)
#PULP_APP_CL_OMP_SRCS += $(wildcard ./../../kernels/fully_connected/*.c)
#PULP_APP_CL_OMP_SRCS += $(wildcard ./../../kernels/pooling/*.c)

ifeq ($(prof),1)
PULP_CFLAGS = -DPROFILING
endif

ifeq ($(check),1)
PULP_CFLAGS += -DCHECKLAYER
endif


PULP_CFLAGS += -DNUM_CORES=8 -O3  -Iinc -w
PULP_LDFLAGS = -lm -lg



include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
