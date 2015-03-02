#if __linux__
#pragma GCC optimize 3
#endif

#include <ml2/common.h>


namespace ml2
{


fml logistic_function(fml z)
{
    return (FML(1.0) / (FML(1.0) + std::exp(-z)));
}

fml derivative_of_logistic_function(fml z)
{
    fml y = logistic_function(z);
    fml slope = (y * (FML(1.0) - y));
    slope = std::max(slope, FML(1e-5));    // <-- Experimental
    return slope;
}

fml inverse_of_logistic_function(fml y)
{
    if (y < FML(0.0001)) y = FML(0.0001);
    if (y > FML(0.9999)) y = FML(0.9999);
    return -std::log((FML(1.0) / y) - FML(1.0));
}

fml logistic_function_min()
{
    return FML(0.0);
}

fml logistic_function_max()
{
    return FML(1.0);
}


fml hyperbolic_function(fml z)
{
    // Recommended by: "Efficient BackProp" (LeCun et al.)
    return FML(1.7159) * std::tanh(FML(2.0)/FML(3.0) * z);
}

fml derivative_of_hyperbolic_function(fml z)
{
    fml s = FML(1.0) / std::cosh(FML(2.0)/FML(3.0) * z);
    return FML(1.14393) * s * s;
}

fml inverse_of_hyperbolic_function(fml y)
{
    if (y < FML(-1.71589)) y = FML(-1.71589);
    if (y > FML(1.71589)) y = FML(1.71589);
    return FML(1.5) * FML(atanh)(FML(0.582785) * y);
}

fml hyperbolic_function_min()
{
    return FML(-1.7159);
}

fml hyperbolic_function_max()
{
    return FML(1.7159);
}


}   // namespace ml2
