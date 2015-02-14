LIBML
=====

Compilation Instructions
------------------------

Run:

         make

Some classes in this library use Eigen, a very fast C++ linear algebra library.
Eigen supports parallelization using OpenMP, and it will do so by default. If
you do not want Eigen to parallelize its operations, you should set the macro
EIGEN_DONT_PARALLELIZE before compiling this library, or remove -fopenmp from
this library's Makefile.

When OpenMP is enabled, you can control how many threads will be created by
setting the OMP_NUM_THREADS environment variable before running your executable.
For example, if main.exec was compiled with OpenMP enabled, you can run it like
this:

    #> OMP_NUM_THREADS=4 ./main.exec

