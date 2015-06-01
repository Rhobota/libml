
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


static
void s_cudaFree(fml*& buf)
{
    if (buf)
    {
        cuda_assert( cudaFree(buf) );
        buf = NULL;
    }
}


static
fml* s_cudaMalloc(u32 size)
{
    fml* ptr = NULL;
    cuda_assert( cudaMalloc((void**)(&ptr), size*sizeof(fml)) );
    return ptr;
}


static
void s_createCublasContext(void*& ptr)
{
    cublasHandle_t* cublasHandle = new cublasHandle_t;
    cublas_assert( cublasCreate(cublasHandle), "s_createCublasContext" );
    ptr = cublasHandle;
}


static
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


#endif   // #ifdef ENABLE_DEVICE_FUNCTIONS


}  // <-- end of un-named namespace
