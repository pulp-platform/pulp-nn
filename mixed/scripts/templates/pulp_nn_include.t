%if config.api == 'PULPNNMatMul':
#if (KERNEL == ${config.kernel.out_data_t}${config.kernel.wt_data_t})
#define OUTPUT ${config.kernel.out_data_t}
#define WEIGHTS ${config.kernel.wt_data_t}
#include "GoldenModelMatMul/golden_8_${config.kernel.out_data_t}_${config.kernel.wt_data_t}.h"
#include "DataAllocationMatMul/data_allocation_8_${config.kernel.out_data_t}_${config.kernel.wt_data_t}.h"
#endif
%elif config.api == 'PULPNNConvolve':
#if (KERNEL == ${config.kernel.in_data_t}${config.kernel.out_data_t}${config.kernel.wt_data_t})
#define INPUT ${config.kernel.in_data_t}
#define OUTPUT ${config.kernel.out_data_t}
#define WEIGHTS ${config.kernel.wt_data_t}
#include "GoldenModelConvolution/golden_${config.kernel.in_data_t}_${config.kernel.out_data_t}_${config.kernel.wt_data_t}.h"
#include "DataAllocationConvolution/data_allocation_${config.kernel.in_data_t}_${config.kernel.out_data_t}_${config.kernel.wt_data_t}.h"
#endif
%elif config.api == 'PULPNNConvolvePointwise':
#if (KERNEL == ${config.kernel.in_data_t}${config.kernel.out_data_t}${config.kernel.wt_data_t})
#define INPUT ${config.kernel.in_data_t}
#define OUTPUT ${config.kernel.out_data_t}
#define WEIGHTS ${config.kernel.wt_data_t}
#include "GoldenModelPointwise/golden_${config.kernel.in_data_t}_${config.kernel.out_data_t}_${config.kernel.wt_data_t}.h"
#include "DataAllocationPointwise/data_allocation_${config.kernel.in_data_t}_${config.kernel.out_data_t}_${config.kernel.wt_data_t}.h"
#endif
%elif config.api == 'PULPNNConvolveDepthwise':
#if (KERNEL == ${config.kernel.in_data_t}${config.kernel.out_data_t}${config.kernel.wt_data_t})
#define INPUT ${config.kernel.in_data_t}
#define OUTPUT ${config.kernel.out_data_t}
#define WEIGHTS ${config.kernel.wt_data_t}
#include "GoldenModelDepthwise/golden_${config.kernel.in_data_t}_${config.kernel.out_data_t}_${config.kernel.wt_data_t}.h"
#include "DataAllocationDepthwise/data_allocation_${config.kernel.in_data_t}_${config.kernel.out_data_t}_${config.kernel.wt_data_t}.h"
#endif
%elif config.api == 'PULPNNLinearNoQuant':
#if (KERNEL == ${config.kernel.in_data_t}${config.kernel.wt_data_t})
#define INPUT ${config.kernel.in_data_t}
#define WEIGHTS ${config.kernel.wt_data_t}
#include "GoldenModelLinearNoQuant/golden_${config.kernel.in_data_t}_32_${config.kernel.wt_data_t}.h"
#include "DataAllocationLinearNoQuant/data_allocation_${config.kernel.in_data_t}_32_${config.kernel.wt_data_t}.h"
#endif
%elif config.api == 'PULPNNLinearQuant':
#if (KERNEL == ${config.kernel.in_data_t}${config.kernel.out_data_t}${config.kernel.wt_data_t})
#define INPUT ${config.kernel.in_data_t}
#define OUTPUT ${config.kernel.out_data_t}
#define WEIGHTS ${config.kernel.wt_data_t}
#include "GoldenModelLinearQuant/golden_${config.kernel.in_data_t}_${config.kernel.out_data_t}_${config.kernel.wt_data_t}.h"
#include "DataAllocationLinearQuant/data_allocation_${config.kernel.in_data_t}_${config.kernel.out_data_t}_${config.kernel.wt_data_t}.h"
#endif
%elif config.api == 'PULPNNMaxPool':
#if (KERNEL == ${config.kernel.in_data_t})
#define INPUT ${config.kernel.in_data_t}
#define OUTPUT ${config.kernel.in_data_t}
#include "GoldenModelMaxPool/golden_${config.kernel.in_data_t}.h"
#include "DataAllocationMaxPool/data_allocation_${config.kernel.in_data_t}.h"
#endif
%elif config.api == 'PULPNNAvgPool':
#if (KERNEL == ${config.kernel.in_data_t})
#define INPUT ${config.kernel.in_data_t}
#define OUTPUT ${config.kernel.in_data_t}
#include "GoldenModelAvgPool/golden_${config.kernel.in_data_t}.h"
#include "DataAllocationAvgPool/data_allocation_${config.kernel.in_data_t}.h"
#endif
%elif config.api == 'PULPNNAdd':
#if (KERNEL == ${config.in1_data_t}${config.in2_data_t})
#define INPUT1 ${config.in1_data_t}
#define INPUT2 ${config.in2_data_t}
#define OUTPUT ${config.max_precision}
#include "GoldenModelAdd/golden_${config.in1_data_t}_${config.in2_data_t}.h"
#include "DataAllocationAdd/data_allocation_${config.in1_data_t}_${config.in2_data_t}.h"
#endif
%endif
