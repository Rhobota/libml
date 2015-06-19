#include <cuda.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cassert>


#define SHARED_MEM_AVAIL_PER_SM 49152


#define cuda_assert(expression) \
    do { \
        cudaError_t err; \
        if ((err = (expression)) != cudaSuccess) \
        { \
            std::cout << "Cuda error: " << cudaGetErrorString(err) << std::endl; \
            assert(false); \
        } \
    } while (false)


#define cublas_assert(expression, what) \
    do { \
        if ((expression) != CUBLAS_STATUS_SUCCESS) \
        { \
            std::cout << "cuBLAS error! " << what << std::endl; \
            assert(false); \
        } \
    } while (false)


namespace ml
{
namespace    // <-- un-named namespaces act like everything inside is statically scoped
{


fml* s_cudaMalloc(u32 size)
{
    fml* ptr = NULL;
    cuda_assert( cudaMalloc((void**)(&ptr), size*sizeof(fml)) );
    return ptr;
}


void s_cudaFree(fml*& buf)
{
    if (buf)
    {
        cuda_assert( cudaFree(buf) );
        buf = NULL;
    }
}


void s_cudaCopyHostToDevice(fml* dest, const fml* source, u32 size)
{
    cuda_assert( cudaMemcpy(dest, source, size*sizeof(fml), cudaMemcpyHostToDevice) );
}


void s_cudaCopyDeviceToHost(fml* dest, const fml* source, u32 size)
{
    cuda_assert( cudaMemcpy(dest, source, size*sizeof(fml), cudaMemcpyDeviceToHost) );
}


void s_createCublasContext(void*& ptr)
{
    cublasHandle_t* cublasHandle = new cublasHandle_t;
    cublas_assert( cublasCreate(cublasHandle), "s_createCublasContext" );
    ptr = cublasHandle;
}


void s_destroyCublasContext(void*& ptr)
{
    if (ptr)
    {
        cublasHandle_t* cublasHandle = (cublasHandle_t*)ptr;
        cublas_assert( cublasDestroy(*cublasHandle), "s_destroyCublasContext" );
        delete cublasHandle;
        ptr = NULL;
    }
}


}  // <-- end of un-named namespace
}  // <-- end of namespace ml
