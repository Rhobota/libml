LIBML
=====

Compilation Instructions
------------------------

Run:

         make



OpenMP
------

Some classes in this library use Eigen, a very fast C++ linear algebra library.
Eigen supports parallelization using OpenMP, and it will do so by default. If
you do not want Eigen to parallelize its operations, you should set the macro
EIGEN_DONT_PARALLELIZE before compiling this library, or remove -fopenmp from
this library's Makefile. Also, see the file: source/ml/Eigen.h

When OpenMP is enabled, you can control how many threads will be created by
setting the OMP_NUM_THREADS environment variable before running your executable.
For example, if main.exec was compiled with OpenMP enabled, you can run it like
this:

    #> OMP_NUM_THREADS=4 ./main.exec



GPU
---

Many of this library's algorithms will run on the GPU if it is compiled and used
properly. The GPU code is written in CUDA. Some of the CUDA code is templatized
so that it runs as fast as possible on the GPU (which doesn't handle branch
prediction very well). While templates make the code run faster, it makes
compilation time much longer. You can turn on/off this templatization (e.g. it's
useful to turn it off while developing and testing the code); for details see
the file: source/ml/conv2d/gpu_common.ipp

Also, see source/ml/conv2d/gpu_common.ipp and source/ml/cuda_stuff.ipp, as they
both have some constants that you may need to edit for your particular GPU.

At this time, multi-GPU is not implemented internally within this library.
However, you can certainly call cudaSetDevice() in your application code
followed by calls into this library to make use of multiple GPUs on the
application side of things. (There are no calls to cudaSetDevice() inside
this library, so calls to this library will not mess up such an endeavour.)

