# PULP-NN Mixed

# In order to use the library in an existing project, you can copy the sources and the headers that are already generated in src and inc folders. Note that, the current version of batch norm activations (32bit or 64bit) is defined in script/setup.py. Before coping the library, check if is which you would like.

# If you want to test the library sources, you can generate the whole setup (pulp-sdk based) and golden models (python 3 and torch are required) using

# (from directory radix)

# > cd scripts
# > python3 pulp_nn_examples_generator.py

# In order to select the kernels to test, open scripts/setup.py and follow the instructions. You can test either a single kernel per type or all set of kernels per type (pointwise convolution, depthwise convolution, linear with 32-bit of outputs precision and linear with sub-byte of outputs precision)

# Then, you can run the simulation on your favorite target architecture using

# (from directory radix)
# > cd test
# > make clean all run cores=NUM_CORES kernel=KERNEL platform=PLATFORM

# Where, NUM_CORES is the number of cores (by default is set to 1) that you want to use and KERNEL is the precision configuration of the kernel (by default is set to 888 or 88) that you want to test (every permutation is already included).

# example: make clean all run cores=8 kernel=888 (and you have selected pointwise in scripts/pulp_nn_examples_generator.py) you will see the results of the 8-bit of inputs, 8-bit of output and 8-bit of weights (in this order) pointwise kernel results, computed in a cluster execution with 8 cores on. Note that, for linear kernels with 32-bit of outputs precision KERNEL can be 88, 84, 82 and so on, for the inputs and weights precision.

# You could modify the kernel sources which are been generated or on the templates used for that, which are in scripts/templates. Then, you can regenerate them using

# (from directory radix)
# > cd scripts
# > python3 pulp_nn_kernels_generator.py