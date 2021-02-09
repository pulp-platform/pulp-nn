APP = test

APP_SRCS = test.c

APP_SRCS += src/SupportFunctions/pulp_nn_utils.c

${config.make}

ifndef cores
cores=1
else
cores = $(cores)
endif

%if config.kernel.type == 'maxpool' or config.kernel.type == 'avgpool':
ifndef kernel
kernel=8
%elif config.kernel.type == 'matmul' or config.kernel.type == 'linear_no_quant' or config.kernel.type == 'add':
ifndef kernel
kernel=88
%elif config.kernel.type == 'convolution' or config.kernel.type == 'pointwise' or config.kernel.type == 'depthwise' or config.kernel.type == 'linear_quant':
ifndef kernel
kernel=888
%endif
else
kernel = $(kernel)
endif

ifeq ($(perf), 1)
APP_CFLAGS += -DVERBOSE_PERF
endif

ifeq ($(check), 1)
APP_CFLAGS += -DVERBOSE_CHECK
endif

APP_CFLAGS += -O3 -Iinclude -w #-flto
APP_CFLAGS += -DNUM_CORES=$(cores) -DKERNEL=$(kernel)

APP_LDFLAGS += -lm #-flto

%if config.kernel.extentions == 'XpulpNN':
PULP_ARCH_CFLAGS = -march=rv32imXpulpnn -D__riscv__
PULP_ARCH_LDFLAGS =  -march=rv32imXpulpnn -D__riscv__

disopt = --disassembler-options="march=rv32imcXpulpnn" -d
%endif

include $(RULES_DIR)/pmsis_rules.mk
