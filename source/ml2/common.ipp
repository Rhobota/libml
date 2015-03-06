
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


}  // <-- end of un-named namespace
