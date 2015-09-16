
#ifdef ENABLE_DEVICE_FUNCTIONS

#include "cuda_stuff.ipp"

#endif


#include <cassert>


namespace ml
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


#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
fml relu_function(fml z)
{
    if (z > FML(0.0))
        return z;
    return z * FML(0.001);
}

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
fml derivative_of_relu_function(fml z)
{
    if (z > FML(0.0))
        return FML(1.0);
    return FML(0.001);
}

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
fml inverse_of_relu_function(fml y)
{
    // This function isn't invertible. But this is close...
    // I want to throw an exception here, but you can't throw
    // exceptions in GPU code.
    return relu_function(y);
}

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
fml relu_function_min()
{
    return FML(0.0);
}

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
fml relu_function_max()
{
    return INFINITY;
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


class tReLUFunc
{
    public:

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
        fml operator()(fml val) const { return relu_function(val); }
};


class tDirReLUFunc
{
    public:

#ifdef ENABLE_DEVICE_FUNCTIONS
__host__ __device__
#endif
        fml operator()(fml val) const { return derivative_of_relu_function(val); }
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


class tMultWithDirLogisticFunc
{
    public:

        __host__ __device__
        fml operator()(const fml& a, const fml& b)
        {
            return a * derivative_of_logistic_function(b);
        }
};


class tMultWithDirHyperbolicFunc
{
    public:

        __host__ __device__
        fml operator()(const fml& a, const fml& b)
        {
            return a * derivative_of_hyperbolic_function(b);
        }
};


class tMultWithDirReLUFunc
{
    public:

        __host__ __device__
        fml operator()(const fml& a, const fml& b)
        {
            return a * derivative_of_relu_function(b);
        }
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


}  // <-- end of un-named namespace
}  // <-- end of namespace ml
