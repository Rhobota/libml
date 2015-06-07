
#ifdef ENABLE_DEVICE_FUNCTIONS

#include <cuda.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#endif


#include <cassert>


namespace ml2
{
namespace    // <-- un-named namespaces act like everything inside is statically scoped
{


#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
fml s_max(fml a, fml b)
{
    if (a > b)
        return a;
    return b;
}

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
fml s_min(fml a, fml b)
{
    if (a < b)
        return a;
    return b;
}


#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
fml logistic_function(fml z)
{
    return (FML(1.0) / (FML(1.0) + std::exp(-z)));
}

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
fml derivative_of_logistic_function(fml z)
{
    fml y = logistic_function(z);
    fml slope = (y * (FML(1.0) - y));
    slope = s_max(slope, FML(1e-5));    // <-- Experimental
    return slope;
}

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
fml inverse_of_logistic_function(fml y)
{
    if (y < FML(0.0001)) y = FML(0.0001);
    if (y > FML(0.9999)) y = FML(0.9999);
    return -std::log((FML(1.0) / y) - FML(1.0));
}

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
fml logistic_function_min()
{
    return FML(0.0);
}

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
fml logistic_function_max()
{
    return FML(1.0);
}


#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
fml hyperbolic_function(fml z)
{
    // Recommended by: "Efficient BackProp" (LeCun et al.)
    return FML(1.7159) * std::tanh(FML(2.0)/FML(3.0) * z);
}

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
fml derivative_of_hyperbolic_function(fml z)
{
    fml s = FML(1.0) / std::cosh(FML(2.0)/FML(3.0) * z);
    return FML(1.14393) * s * s;
}

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
fml inverse_of_hyperbolic_function(fml y)
{
    if (y < FML(-1.71589)) y = FML(-1.71589);
    if (y > FML(1.71589)) y = FML(1.71589);
    return FML(1.5) * FML(atanh)(FML(0.582785) * y);
}

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
fml hyperbolic_function_min()
{
    return FML(-1.7159);
}

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
fml hyperbolic_function_max()
{
    return FML(1.7159);
}


class tExpFunc
{
    public:

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
        fml operator()(fml val) const { return s_min(std::exp(val), FML(1e30)); }
};


class tLogisticFunc
{
    public:

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
        fml operator()(fml val) const { return logistic_function(val); }
};


class tDirLogisticFunc
{
    public:

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
        fml operator()(fml val) const { return derivative_of_logistic_function(val); }
};


class tHyperbolicFunc
{
    public:

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
        fml operator()(fml val) const { return hyperbolic_function(val); }
};


class tDirHyperbolicFunc
{
    public:

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
        fml operator()(fml val) const { return derivative_of_hyperbolic_function(val); }
};


class t_RMSPROP_update
{
    public:

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
        fml operator()(fml accum, fml accum_avg) const
        {
            return (accum_avg > FML(0.0)) ? (accum / std::sqrt(accum_avg)) : FML(0.0);
        }
};


#ifdef ENABLE_DEVICE_FUNCTIONS

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


class tFillColumnsWithFunc
{
    public:

        tFillColumnsWithFunc(fml* vect, u32 vectSize)
            : m_vect(vect), m_vectSize(vectSize) { }

        __host__ __device__
        fml operator()(const ssize_t& index)
        {
            return m_vect[(index % m_vectSize)];
        }

    private:

        fml* m_vect;
        u32  m_vectSize;
};


class tColumnIndexFunc : public thrust::unary_function<u32,u32>
{
    public:

        tColumnIndexFunc(u32 numRows)
            : m_numRows(numRows) { }

        __host__ __device__
        u32 operator()(u32 index)
        {
            return (index / m_numRows);
        }

    private:

        u32 m_numRows;
};


class tDivInputColsByVectorValues
{
    public:

        tDivInputColsByVectorValues(fml* input, fml* vect, u32 numInputRows)
            : m_input(input), m_vect(vect), m_numInputRows(numInputRows) { }

        __host__ __device__
        fml operator()(const ssize_t& index)
        {
            fml denom = m_vect[(index / m_numInputRows)];

            if (denom > FML(0.0))
                return m_input[index] / denom;
            else
                return FML(1.0) / ((fml) m_numInputRows);
        }

    private:

        fml* m_input;
        fml* m_vect;
        u32  m_numInputRows;
};


class tNoOp
{
    public:

        __host__ __device__
        fml operator()(const fml& val)
        {
            return val;
        }
};


template<class T>
class tMultWithUniOperator
{
    public:

        __host__ __device__
        fml operator()(const fml& a, const fml& b)
        {
            return a * m_uniOp(b);
        }

    private:

        T m_uniOp;
};


class tSubWithScalarMult
{
    public:

        tSubWithScalarMult(fml mult)
            : m_mult(mult) { }

        __host__ __device__
        fml operator()(const fml& a, const fml& b)
        {
            return a - m_mult*b;
        }

    private:

        fml m_mult;
};


class tVelUpdateFunc
{
    public:

        tVelUpdateFunc(fml viscosity, fml mult)
            : m_viscosity(viscosity), m_mult(mult) { }

        __host__ __device__
        fml operator()(const fml& vel, const fml& dw_accum)
        {
            return vel*m_viscosity - m_mult*dw_accum;
        }

    private:

        fml m_viscosity, m_mult;
};


class tMultBy
{
    public:

        tMultBy(fml val)
            : m_val(val) { }

        __host__ __device__
        fml operator()(const fml& val)
        {
            return m_val * val;
        }

    private:

        fml m_val;
};


class t_RMSPROP_avg_update
{
    public:

        __host__ __device__
        fml operator()(const fml& dw_accum_avg, const fml& dw_accum)
        {
            return dw_accum_avg*FML(0.9) + dw_accum*dw_accum*FML(0.1);
        }
};


class t_RMSPROP_update_with_alpha
{
    public:

        t_RMSPROP_update_with_alpha(fml alpha)
            : m_alpha(alpha) { }

        __host__ __device__
        fml operator()(fml accum, fml accum_avg) const
        {
            return m_alpha * m_func(accum, accum_avg);
        }

    private:

        fml m_alpha;
        t_RMSPROP_update m_func;
};

#endif   // #ifdef ENABLE_DEVICE_FUNCTIONS


#ifdef ENABLE_CONVOLVE_CPU

void s_conv2d(const fml* inputPtr,  u32 inputRows,   u32 inputCols,   u32 inputComponents,
              const fml* kernelPtr, u32 kernelRows,  u32 kernelCols,
                                    u32 kernelStepY, u32 kernelStepX,
                                    u32 numKernels,
              const fml* kernelBiases, fml scaleFactor,
              fml* outputPtr)
{
    // TODO: templatize -- including templatizing the block() calls
    // TODO: re-structure to remove as many ifs from the inner loops as possible

    assert(inputPtr && inputRows > 0 && inputCols > 0 && inputComponents > 0);
    assert(kernelPtr && (kernelRows % 2) == 1 && (kernelCols % 2) == 1);
    assert(kernelStepY > 0 && kernelStepX > 0 && numKernels > 0);
    assert(kernelBiases);
    assert(outputPtr);

    u32 kernelRadiusY = kernelRows / 2;
    u32 kernelRadiusX = kernelCols / 2;

    MapRowMajorConst input(inputPtr, inputRows, inputCols * inputComponents);

    for (u32 r = 0; r < inputRows; r += kernelStepY)
    {
        u32 y, ky;
        if (r < kernelRadiusY)
            y = 0, ky = kernelRadiusY - r;
        else
            y = r - kernelRadiusY, ky = 0;

        u32 h = r + kernelRadiusY + 1;  // not really "+ 1", but "+ (kernelRows%2)", except we know that is always 1, so we're taking a shortcut here
        if (h > inputRows)
            h = inputRows;
        h -= y;

        for (u32 c = 0; c < inputCols; c += kernelStepX)
        {
            u32 x, kx;
            if (c < kernelRadiusX)
                x = 0, kx = kernelRadiusX - c;
            else
                x = c - kernelRadiusX, kx = 0;

            u32 w = c + kernelRadiusX + 1;  // not really "+ 1", but "+ (kernelCols%2)", except we know that is always 1, so we're taking a shortcut here
            if (w > inputCols)
                w = inputCols;
            w -= x;

            for (u32 i = 0; i < numKernels; i++)
            {
                MapRowMajorConst kernel(kernelPtr + i * kernelRows * kernelCols * inputComponents,
                                        kernelRows,
                                        kernelCols * inputComponents);
                fml val = input.block(y, x*inputComponents, h, w*inputComponents)
                               .cwiseProduct(kernel.block(ky, kx*inputComponents, h, w*inputComponents))
                               .sum();
                *outputPtr++ = (val + kernelBiases[i]) * scaleFactor;
            }
        }
    }
}

#endif  // ENABLE_CONVOLVE_CPU


}  // <-- end of un-named namespace
}  // <-- end of namespace ml2
