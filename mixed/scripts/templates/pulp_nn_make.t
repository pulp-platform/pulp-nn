%if config.api == "PULPNNConvolve":
APP_SRCS += src/Convolution/${config.filename}
%elif config.api == "PULPNNConvolvePointwise":
APP_SRCS += src/Pointwise/${config.filename}
%elif config.api == "PULPNNMatMul":
APP_SRCS += src/MatrixMultiplication/${config.filename}
%elif config.api == "PULPNNConvolveDepthwise":
APP_SRCS += src/Depthwise/${config.filename}
%elif config.api == "PULPNNLinearNoQuant":
APP_SRCS += src/LinearNoQuant/${config.filename}
%elif config.api == "PULPNNLinearQuant":
APP_SRCS += src/LinearQuant/${config.filename}
%elif config.api == "PULPNNMaxPool":
APP_SRCS += src/Pooling/MaxPool/${config.filename}
%elif config.api == "PULPNNAvgPool":
APP_SRCS += src/Pooling/AvgPool/${config.filename}
%elif config.api == "PULPNNAdd":
APP_SRCS += src/Add/${config.filename}
%endif
