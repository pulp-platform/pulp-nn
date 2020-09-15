# PULP-NN: Enabling QNN inference on PULP

PULP_NN is a multicore computing library for QNN inference on Parallel-Ultra-Low-Power (PULP) Clusters of RISC-V based processors. 
It adopts the Height-Width-Channel (HWC) layout to store NN weights and activations and the implementation of the convolution-based kernels as a Matrix Multiplication operation, as proposed by ARM's CMSIS-NN open source library.
It fully exploits the Xpulp ISA extension and the cluster's parallelism to achieve high performance and high energy efficiency on PULP-based devices.

The PULP-NN library is described and evaluated fully in Garofalo et al. [\[arXiv:1908.11263\]](https://arxiv.org/abs/1908.11263). If you intend to use or reference PULP-NN for an academic publication, please consider citing it:
```
Garofalo Angelo, Rusci Manuele, Conti Francesco, Rossi Davide and Benini Luca 2020PULP-NN: accelerating quantized neural networks on parallel ultra-low-power RISC-V processorsPhil. Trans. R. Soc. A.37820190155
http://doi.org/10.1098/rsta.2019.0155
```

The version of this paper is available at the branch [paper_version](https://github.com/pulp-platform/pulp-nn/tree/paper_version) of this repository.

An updated an evolved version of the library is now available on [master](https://github.com/pulp-platform/pulp-nn/tree/master) and it is composed by the kernels of previous version (``8bit`` directory) and mixed- and sub-byte precision (``mixed`` directory) ones.
The latter is explained in detail in Bruschi et al. [\[arXiv:2007.07759\]](https://arxiv.org/abs/2007.07759). If you intend to use or reference PULP-NN Mixed for an academic publication, please consider citing it:
```
@inproceedings{10.1145/3387902.3394038,
author = {Bruschi, Nazareno and Garofalo, Angelo and Conti, Francesco and Tagliavini, Giuseppe and Rossi, Davide},
title = {Enabling Mixed-Precision Quantized Neural Networks in Extreme-Edge Devices},
year = {2020},
isbn = {9781450379564},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3387902.3394038},
doi = {10.1145/3387902.3394038},
booktitle = {Proceedings of the 17th ACM International Conference on Computing Frontiers},
pages = {217–220},
numpages = {4},
keywords = {embedded systems, quantized neural network, low power architectures},
location = {Catania, Sicily, Italy},
series = {CF ’20}
}
```

## 8bit or mixed

Both the sub-directories are structured as will explain below but the ``8bit`` directory contains kernels which are already tested in QNN inference applications and is completed. It is ready to use if you want to write your applications.
``mixed`` directory instead, is a work-in-progress library in which every kernel is already tested "on banch" but not in a real QNN inference application. It contains a ``scripts`` directory in which you can generate tests to stimulate the kernels and study the behaviours.
To start with PULP-NN Mixed, please refer to the ``README`` in ``mixed`` directory.

## Structure of the library

The library is organized as follow:
+ The ``32bit`` and ``64bit`` directories refer to the precision of the batch normalization paramenters;
+ To use the library the header file under the ``include`` directory should be inserted in your QNN inference code. They are ``pulp_nn_kernels.h`` and ``pulp_nn_utils.h``, which contains every kernel and useful function of PULP-NN library;
+ The directory ``src`` contains every computational kernel and useful function;

## Convolutions

To  efficiently  execute  the  convolution  on  MCUs, it  is  decomposed  into  two  phases:  the im2colstep  loads the  3D  input  features  of  the  current  convolution  into  a  1D vector, while the dot product step is implemented as a Matrix Multiplication (MatMul).

PULP-NN contains different convolution based kernels:
+ The Standard convolution supports squared and non-squared input feature maps, squared and non-squared filters and also asymmetric padding;
+ Pointwise convolution kernels, in which the im2col step is unnecessary and it has been removed to further speed-up the computation, and one in which the parallelization has been optimized to compute small spatial size layers;
+ The library is also provided with an efficient Depthwise convolution;
+ The linear kernels. One for 8-bit quantized outputs and one for a 32-bit not quantized outputs.

The inner kernel of the convolution and linear layers consists of an efficient Matrix Multiplication (MatMul) kernel, which exploits the SIMD sum of dot products ISA instructions. This allows to achieve high performance and high operation efficiency.
It is known that the MatMul kernel easily blows up the memory if the memory access patterns are not regular. The HWC data layout avoids such a performance degradation.
The activations and the weights are stored in contiguous memory cells firstly along the channels and then along the spatial dimensions. Such a structure allows to access the two operands of the matrix multiplication in the same memory order, regularizing so the memory access patterns.

Being the core of the convolution computation, the Matrix Multiplication needs to be highly optimized. We explored different MatMul structures to maximize the data reuse at the register file level and thus the throughput. The 4x2 sized MatMul kernels revealed to be the best solution to speedup the convolution. It works on two activation output of four consecutive channels in parallel.

Depthwise convolution instead requires different data layout to minimize the performance dropping due to the channel-wise computation. PULP-NN Depthwise convolution exploits CHW data layout for input activations (and weights) and provides HWC outputs activations layout to feed the subsequent layer, typically a Standard or a Pointwise convolution one.
To maximize the data reusing and speed-up the computation, Depthwise kernel performs an im2col step, in which not only kxk input activations are loaded and reordered but H rows.

## Pooling

The average and the maximum pooling functions are split into two phases, as proposed by CMSIS-NN.
First we perform the pooling along the x spatial dimension, with in situ updates, to save in terms of memory footprint, and then along the y spatial dimension.

## Activations and Quantization

PULP-NN supports an efficient implementation of the Batch Normalization and Rectified-Linear-Unit (ReLu) activation functions and a staircase quantization function, fusing them into the computational kernels, which are handlable with flags when you are calling the kernel.

## Parallelism

All the kernels provided by the PULP-NN library are efficiently parallelized on a cluster of eight RI5CY processors.
The data-parallel execution exploits fully the PULP cluster, achieving almost linear speedup for all the kernels.

+ Convolution: to take advantage of the HWC data layout, the data chunks to be assigned to each core are built along the H spatial dimension of the Output Feature Map;

+ Pointwise small spatial dimension: is a combination of the chunk building along the H and W spatial dimensions. Each cores computes half Ouput Feature Map along W and just the same of Standard convolution along H one;

+ Depthwise: each core computes a balanced number of channels;

+ Linear: each core computes a balanced number of output neurons;

## Getting Started with PULP-NN

For both library are strictly required to have:

+ [pulp-sdk](https://github.com/pulp-platform/pulp-sdk) on Linux-based machine. If you have not done yet, please refer to its ``README`` for the right installation on your machine;

+ [pulp-toolchain](https://github.com/pulp-platform/pulp-riscv-gnu-toolchain) on the same machine. If you have not done yet, please refer to its ``README`` for the right installation on your machine;

The PULP-NN kernels can be run either on the OpenPULP RTL platform either on any embodiment of the PULP cluster.
To use the QNN kernels to run Neural Networks on PULP, it is necessary to compile the kernels and the utils before running the application.
The ``pulp_nn_kernels.h`` header must be included in the main file of the application code.

If you want to approach with PULP-NN mixed you could need other requirements such as python and please refer to its ``README`` for more details.

## Support and Contribution

+ **Nazareno Bruschi**, *University of Bologna*, [email](mailto:nazareno.bruschi@unibo.it)
+ **Angelo Garofalo**, *University of Bologna*, [email](mailto:angelo.garofalo@unibo.it)
+ **Alessio Burrello**, *University of Bologna*, [email](mailto:alessio.burrello@unibo.it)
+ **Francesco Conti**, *University of Bologna*, [email](mailto:francesco.conti@unibo.it)
+ **Giuseppe Tagliavini**, *University of Bologna*, [email](mailto:giuseppe.tagliavini@unibo.it)
+ **Manuele Rusci**, *University of Bologna*, [email](mailto:manuele.rusci@unibo.it)
+ **Davide Rossi**, *University of Bologna*, [email](mailto:davide.rossi@unibo.it)
+ **Luca Benini**, *University of Bologna and ETH Zurich*, [email](mailto:luca.benini@unibo.it)

## Current limitations

+ Refer to PULP-NN Mixed ``README``;
