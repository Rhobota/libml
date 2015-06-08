#include <ml2/common.h>

#include "../../source/ml/Eigen.h"

#define ENABLE_CONVOLVE_CPU
#include "../../source/ml2/common_nn.ipp"

#include <rho/tTest.h>
#include <rho/tCrashReporter.h>

using namespace rho;
using ml2::fml;
using ml2::s_conv2d;


/*
 * Note: The "correctOutput" vectors in this file were calculated using Octave's
 * conv2() function.
 *
 * E.g.
 *
 * > i = rand(5,5)                                  # the input matrix
 * > k = rand(3,3)                                  # the convolution kernel
 * > b = rand()                                     # the bias term
 * > c = conv2(i, fliplr(flipud(k)), "same") + b    # the convolve step
 *
 *
 * Note: Matlab and Octave also have a xcorr2() function that implements
 * convolution like I do, but I'm not using that above because I want to draw
 * explicit attention to this fact that conv2() and xcorr2() are different in
 * Matlab and Octave!
 */


static
void s_checkOutput(const tTest& t, fml* output, fml* correctOutput, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        if (fabs(output[i] - correctOutput[i]) >= 0.0001)
        {
            std::cerr << "Fail at element " << i << ": output = " << output[i] << "  correctOutput = " << correctOutput[i] << std::endl;
            t.fail();
        }
    }
}


void test0(const tTest& t)
{
    fml input[] = {  FML(1.0),  FML(2.0),
                     FML(3.0),  FML(4.0),  };
    u32 inputRows = 2;
    u32 inputCols = 2;
    u32 inputComponents = 1;

    fml kernel[] = {  FML(1.0),  FML(2.0),  FML(3.0),
                      FML(4.0),  FML(5.0),  FML(6.0),
                      FML(7.0),  FML(8.0),  FML(9.0),  };
    u32 kernelRows = 3;
    u32 kernelCols = 3;

    fml bias = -FML(1.0);

    fml correctOutput[] = {  FML(76.0),  FML(66.0),
                             FML(46.0),  FML(36.0),  };
    fml output[] = {  FML(0.0),  FML(0.0),
                      FML(0.0),  FML(0.0),  };

    s_conv2d(input, inputRows, inputCols, inputComponents,
             kernel, kernelRows, kernelCols,
                     1, 1,
                     1,
             &bias, FML(1.0),
             output);

    s_checkOutput(t, output, correctOutput, inputRows*inputCols);
}


fml kKernel33_1[] = {
    FML(0.54221),   FML(0.28064),   FML(0.41314),
    FML(0.15373),   FML(0.72719),   FML(0.16746),
    FML(0.83394),   FML(0.73315),   FML(0.70754),
};
fml kBias33_1     = FML(0.21668);

fml kKernel55_1[] = {
    FML(0.809372),   FML(0.973671),   FML(0.838673),   FML(0.742363),   FML(0.992151),
    FML(0.469994),   FML(0.373717),   FML(0.011637),   FML(0.543847),   FML(0.242291),
    FML(0.434365),   FML(0.227910),   FML(0.584053),   FML(0.557097),   FML(0.900669),
    FML(0.526409),   FML(0.503185),   FML(0.944718),   FML(0.814531),   FML(0.748724),
    FML(0.357251),   FML(0.774389),   FML(0.911320),   FML(0.022793),   FML(0.265590),
};
fml kBias55_1     = FML(0.41873);

fml kKernel57_1[] = {
    FML(0.055275),   FML(0.074646),   FML(0.953215),   FML(0.880510),   FML(0.847759),   FML(0.389875),   FML(0.193407),
    FML(0.682900),   FML(0.049138),   FML(0.850288),   FML(0.280178),   FML(0.496713),   FML(0.445732),   FML(0.472153),
    FML(0.865554),   FML(0.579464),   FML(0.508697),   FML(0.665348),   FML(0.204075),   FML(0.212285),   FML(0.370633),
    FML(0.323998),   FML(0.568641),   FML(0.417675),   FML(0.313431),   FML(0.043598),   FML(0.900668),   FML(0.895766),
    FML(0.972583),   FML(0.541725),   FML(0.764313),   FML(0.426405),   FML(0.700672),   FML(0.250786),   FML(0.499891),
};
fml kBias57_1     = FML(0.81093);


#define TEST_ALL_1_COMPONENT_1_COUNT_KERNELS \
    fml* output = new fml[inputRows*inputCols]; \
    for (u32 i = 0; i < inputRows*inputCols; i++) \
        output[i] = FML(0.0); \
 \
    { \
        s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel33_1, 3, 3, \
                              1, 1, \
                              1, \
                 &kBias33_1, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k33, inputRows*inputCols); \
    } \
 \
    { \
        s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel55_1, 5, 5, \
                              1, 1, \
                              1, \
                 &kBias55_1, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k55, inputRows*inputCols); \
    } \
 \
    { \
        s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel57_1, 5, 7, \
                              1, 1, \
                              1, \
                 &kBias57_1, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k57, inputRows*inputCols); \
    } \
 \
    delete [] output; \


void test1(const tTest& t)
{
    fml input[] = {  FML(0.22451)  };
    u32 inputRows = 1;
    u32 inputCols = 1;

    fml correctOutput_k33[] = {  FML(0.37994)  };

    fml correctOutput_k55[] = {  FML(0.54986)  };

    fml correctOutput_k57[] = {  FML(0.96030)  };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test2(const tTest& t)
{
    fml input[] = {  FML(0.0084837),   FML(0.6833903),  };
    u32 inputRows = 1;
    u32 inputCols = 2;

    fml correctOutput_k33[] = {  FML(0.33729),   FML(0.71494),  };

    fml correctOutput_k55[] = {  FML(0.80440),   FML(0.81980),  };

    fml correctOutput_k57[] = {  FML(0.95603),   FML(1.26993),  };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test3(const tTest& t)
{
    fml input[] =
    {  FML(0.82231),   FML(0.66422),   FML(0.31948),
    };
    u32 inputRows = 1;
    u32 inputCols = 3;

    fml correctOutput_k33[] =
    {  FML(0.92589),   FML(0.87961),   FML(0.55111),
    };

    fml correctOutput_k55[] =
    {  FML(1.5568),   FML(1.1721),   FML(1.1139),
    };

    fml correctOutput_k57[] =
    {  FML(1.5614),   FML(1.7364),   FML(1.8379),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test4(const tTest& t)
{
    fml input[] =
    {  FML(0.2843171),   FML(0.0054868),   FML(0.7031313),   FML(0.3080383),
    };
    u32 inputRows = 1;
    u32 inputCols = 4;

    fml correctOutput_k33[] =
    {  FML(0.42435),   FML(0.38213),   FML(0.78042),   FML(0.54878),
    };

    fml correctOutput_k55[] =
    {  FML(1.22113),   FML(1.15589),   FML(1.12575),   FML(0.76127),
    };

    fml correctOutput_k57[] =
    {  FML(1.2646),   FML(1.1681),   FML(1.5092),   FML(1.6228),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test5(const tTest& t)
{
    fml input[] =
    {  FML(0.52358),   FML(0.71060),   FML(0.38927),   FML(0.13739),   FML(0.11871),
    };
    u32 inputRows = 1;
    u32 inputCols = 5;

    fml correctOutput_k33[] =
    {  FML(0.71642),   FML(0.87910),   FML(0.63201),   FML(0.39631),   FML(0.32413),
    };

    fml correctOutput_k55[] =
    {  FML(1.47100),   FML(1.29369),   FML(1.21892),   FML(0.96248),   FML(0.68846),
    };

    fml correctOutput_k57[] =
    {  FML(1.4379),   FML(1.7027),   FML(1.7880),   FML(1.9895),   FML(1.8004),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test6(const tTest& t)
{
    fml input[] =
    {  FML(0.52680),   FML(0.64388),   FML(0.17254),   FML(0.70199),   FML(0.63416),   FML(0.99946),
    };
    u32 inputRows = 1;
    u32 inputCols = 6;

    fml correctOutput_k33[] =
    {  FML(0.70759),   FML(0.79478),   FML(0.55869),   FML(0.85988),   FML(0.95312),   FML(1.04096),
    };

    fml correctOutput_k55[] =
    {  FML(1.2405),   FML(1.6432),   FML(1.8573),   FML(2.4012),   FML(1.5808),   FML(1.4519),
    };

    fml correctOutput_k57[] =
    {  FML(1.5896),   FML(1.9266),   FML(2.2068),   FML(2.5364),   FML(2.4512),   FML(2.3546),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test7(const tTest& t)
{
    fml input[] =
    {  FML(0.57096),   FML(0.62612),   FML(0.24641),   FML(0.64417),   FML(0.65714),   FML(0.20753),   FML(0.65333),
    };
    u32 inputRows = 1;
    u32 inputCols = 7;

    fml correctOutput_k33[] =
    {  FML(0.73672),   FML(0.80102),   FML(0.59999),   FML(0.83304),   FML(0.82833),   FML(0.57802),   FML(0.72368),
    };

    fml correctOutput_k55[] =
    {  FML(1.3229),   FML(1.6320),   FML(1.9041),   FML(1.6761),   FML(1.7604),   FML(1.3335),   FML(1.1330),
    };

    fml correctOutput_k57[] =
    {  FML(1.6096),   FML(1.9485),   FML(1.9721),   FML(2.6422),   FML(2.4416),   FML(2.0032),   FML(2.2895),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test8(const tTest& t)
{
    fml input[] =
     {  FML(0.087327),   FML(0.327156),   FML(0.295933),   FML(0.545469),   FML(0.899346),   FML(0.196497),   FML(0.698366),  FML(0.825638),
     };
    u32 inputRows = 1;
    u32 inputCols = 8;

    fml correctOutput_k33[] =
    {  FML(0.33497),   FML(0.51757),   FML(0.57352),   FML(0.80944),   FML(0.98744),   FML(0.61478),   FML(0.89299),   FML(0.92444),
    };

    fml correctOutput_k55[] =
    {  FML(0.91853),   FML(1.28586),   FML(1.81795),   FML(1.62486),   FML(1.93532),   FML(2.10808),   FML(1.72200),   FML(1.14546),
    };

    fml correctOutput_k57[] =
    {  FML(1.2008),   FML(1.5825),   FML(1.5999),   FML(2.0736),   FML(2.6358),   FML(2.2892),   FML(2.5373),   FML(2.6078),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test9(const tTest& t)
{
    fml input[] =
    {  FML(0.29033),
   FML(0.56729),
    };
    u32 inputRows = 2;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(0.84371),
   FML(0.71069),
    };

    fml correctOutput_k55[] =
    {  FML(1.12423),
   FML(0.75344),
    };

    fml correctOutput_k57[] =
    {  FML(1.1819),
   FML(1.2697),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test10(const tTest& t)
{
    fml input[] =
    {  FML(0.70124),   FML(0.89188),
   FML(0.83127),   FML(0.20847),
    };
    u32 inputRows = 2;
    u32 inputCols = 2;

    fml correctOutput_k33[] =
    {  FML(1.6329),   FML(1.8191),
   FML(1.4213),   FML(1.1266),
    };

    fml correctOutput_k55[] =
    {  FML(2.2803),   FML(1.7147),
   FML(1.5136),   FML(1.0024),
    };

    fml correctOutput_k57[] =
    {  FML(1.7291),   FML(2.1736),
   FML(2.0460),   FML(2.2186),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test11(const tTest& t)
{
    fml input[] =
    {  FML(0.994467),   FML(0.605887),   FML(0.027617),
   FML(0.477909),   FML(0.583280),   FML(0.449822),
    };
    u32 inputRows = 2;
    u32 inputCols = 3;

    fml correctOutput_k33[] =
    {  FML(1.80437),   FML(1.95922),   FML(1.14611),
   FML(1.19129),   FML(1.51029),   FML(0.96973),
    };

    fml correctOutput_k55[] =
    {  FML(2.6253),   FML(2.1725),   FML(1.9749),
   FML(1.7757),   FML(1.5126),   FML(1.7161),
    };

    fml correctOutput_k57[] =
    {  FML(2.1825),   FML(2.1276),   FML(2.3701),
   FML(1.9353),   FML(2.5630),   FML(2.2556),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test12(const tTest& t)
{
    fml input[] =
    {  FML(0.502066),   FML(0.957044),   FML(0.056785),   FML(0.901603),
   FML(0.698238),   FML(0.852240),   FML(0.397751),   FML(0.431815),
    };
    u32 inputRows = 2;
    u32 inputCols = 4;

    fml correctOutput_k33[] =
    {  FML(1.85694),   FML(2.48785),   FML(1.86393),   FML(1.52933),
   FML(1.40344),   FML(1.57464),   FML(1.61660),   FML(0.87566),
    };

    fml correctOutput_k55[] =
    {  FML(2.9479),   FML(3.7396),   FML(2.9143),   FML(2.4307),
   FML(2.1996),   FML(2.1342),   FML(2.4738),   FML(1.6133),
    };

    fml correctOutput_k57[] =
    {  FML(2.6876),   FML(2.8711),   FML(2.7070),   FML(3.4412),
   FML(2.7609),   FML(3.0311),   FML(3.3040),   FML(3.0896),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test13(const tTest& t)
{
    fml input[] =
    {  FML(0.95928),   FML(0.93373),   FML(0.40276),   FML(0.66430),   FML(0.56894),
   FML(0.67245),   FML(0.14751),   FML(0.71125),   FML(0.17486),   FML(0.79590),
    };
    u32 inputRows = 2;
    u32 inputCols = 5;

    fml correctOutput_k33[] =
    {  FML(1.6680),   FML(2.2828),   FML(1.5325),   FML(2.1414),   FML(1.4619),
   FML(1.3854),   FML(1.4950),   FML(1.6796),   FML(1.2263),   FML(1.3422),
    };

    fml correctOutput_k55[] =
    {  FML(3.1499),   FML(3.1934),   FML(4.0044),   FML(2.8701),   FML(2.2917),
   FML(2.1508),   FML(1.9612),   FML(3.2777),   FML(2.0969),   FML(1.6766),
    };

    fml correctOutput_k57[] =
    {  FML(2.9859),   FML(3.5828),   FML(3.7575),   FML(3.6336),   FML(3.3437),
   FML(2.7300),   FML(3.5705),   FML(3.4910),   FML(3.6312),   FML(3.3509),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test14(const tTest& t)
{
    fml input[] =
    {  FML(0.336546),   FML(0.164736),   FML(0.124773),   FML(0.687435),   FML(0.309419),   FML(0.279011),
   FML(0.209603),   FML(0.848987),   FML(0.719823),   FML(0.241466),   FML(0.401619),   FML(0.076185),
    };
    u32 inputRows = 2;
    u32 inputCols = 6;

    fml correctOutput_k33[] =
    {  FML(1.24336),   FML(1.71564),   FML(1.85444),   FML(1.84905),   FML(1.14381),   FML(0.85792),
   FML(0.67378),   FML(1.26708),   FML(1.31943),   FML(0.95860),   FML(1.13346),   FML(0.57990),
    };

    fml correctOutput_k55[] =
    {  FML(2.2479),   FML(2.9549),   FML(3.0519),   FML(2.7653),   FML(1.9076),   FML(1.3520),
   FML(1.7862),   FML(1.9430),   FML(2.2899),   FML(1.7529),   FML(1.5343),   FML(1.1016),
    };

    fml correctOutput_k57[] =
    {  FML(2.3171),   FML(2.3401),   FML(2.6220),   FML(2.8541),   FML(2.5528),   FML(2.2225),
   FML(1.9223),   FML(2.6762),   FML(2.8086),   FML(2.9239),   FML(3.2969),   FML(2.2891),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test15(const tTest& t)
{
    fml input[] =
     {  FML(0.9284847),   FML(0.2235027),   FML(0.1045408),   FML(0.1017261),   FML(0.0991668),   FML(0.5384337),  FML(0.5551741),
    FML(0.9498428),   FML(0.2002578),   FML(0.0038149),   FML(0.2791154),   FML(0.0997967),   FML(0.7981017),  FML(0.8047618),
     };
    u32 inputRows = 2;
    u32 inputCols = 7;

    fml correctOutput_k33[] =
    {  FML(1.76735),   FML(1.48108),   FML(0.71138),   FML(0.60176),   FML(1.26521),   FML(1.95418),   FML(1.95875),
   FML(1.29384),   FML(1.11832),   FML(0.48954),   FML(0.56315),   FML(0.77125),   FML(1.38140),   FML(1.37234),
    };

    fml correctOutput_k55[] =
    {  FML(2.2430),   FML(1.7900),   FML(1.9865),   FML(2.1891),   FML(2.8346),   FML(2.7159),   FML(2.1232),
   FML(1.2462),   FML(1.4368),   FML(1.7250),   FML(1.7737),   FML(2.2274),   FML(1.8702),   FML(1.3683),
    };

    fml correctOutput_k57[] =
    {  FML(2.0941),   FML(2.3121),   FML(3.2155),   FML(4.1600),   FML(2.3850),   FML(1.9692),   FML(2.3325),
   FML(2.0538),   FML(2.5206),   FML(2.4536),   FML(3.7386),   FML(2.3155),   FML(2.3093),   FML(2.7395),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test16(const tTest& t)
{
    fml input[] =
     {  FML(0.207039),   FML(0.113207),   FML(0.905827),   FML(0.109260),   FML(0.720840),   FML(0.178480),   FML(0.660971),  FML(0.069806),
    FML(0.532035),   FML(0.299957),   FML(0.281805),   FML(0.336894),   FML(0.437951),   FML(0.730484),   FML(0.208357),  FML(0.820568),
     };
    u32 inputRows = 2;
    u32 inputCols = 8;

    fml correctOutput_k33[] =
    {  FML(0.98849),   FML(1.34550),   FML(1.60620),   FML(1.34796),   FML(1.90643),   FML(1.61617),   FML(2.07897),   FML(1.14441),
   FML(0.75868),   FML(1.08205),   FML(0.88487),   FML(1.39795),   FML(1.04455),   FML(1.56411),   FML(0.92902),   FML(1.22340),
    };

    fml correctOutput_k55[] =
    {  FML(2.3765),   FML(2.1679),   FML(3.0732),   FML(2.8221),   FML(3.4355),   FML(3.0377),   FML(2.6608),   FML(1.9522),
   FML(1.4338),   FML(1.7734),   FML(1.8491),   FML(2.5402),   FML(2.2005),   FML(2.6459),   FML(1.8054),   FML(1.5945),
    };

    fml correctOutput_k57[] =
    {  FML(1.9400),   FML(2.4910),   FML(3.4123),   FML(3.4491),   FML(3.6616),   FML(3.7363),   FML(2.6324),   FML(2.8224),
   FML(1.9804),   FML(2.6192),   FML(2.7118),   FML(3.8319),   FML(3.0270),   FML(3.8219),   FML(2.5156),   FML(3.3478),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test17(const tTest& t)
{
    fml input[] =
    {  FML(0.93911),
   FML(0.83836),
   FML(0.15584),
    };
    u32 inputRows = 3;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(1.51423),
   FML(1.20414),
   FML(0.56529),
    };

    fml correctOutput_k55[] =
    {  FML(1.9013),
   FML(1.0665),
   FML(1.3071),
    };

    fml correctOutput_k57[] =
    {  FML(1.7650),
   FML(1.6807),
   FML(1.9764),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test18(const tTest& t)
{
    fml input[] =
    {  FML(0.57724),   FML(0.78659),
   FML(0.48628),   FML(0.56567),
   FML(0.94784),   FML(0.82385),
    };
    u32 inputRows = 3;
    u32 inputCols = 2;

    fml correctOutput_k33[] =
    {  FML(1.5249),   FML(1.6977),
   FML(2.4298),   FML(2.6310),
   FML(1.4141),   FML(1.3839),
    };

    fml correctOutput_k55[] =
    {  FML(2.9968),   FML(3.2736),
   FML(3.0189),   FML(2.3401),
   FML(2.8126),   FML(2.5260),
    };

    fml correctOutput_k57[] =
    {  FML(2.5140),   FML(3.0841),
   FML(2.1353),   FML(2.8000),
   FML(3.2020),   FML(3.6560),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test19(const tTest& t)
{
    fml input[] =
    {  FML(0.46614),   FML(0.63596),   FML(0.85341),
   FML(0.89549),   FML(0.42656),   FML(0.28634),
   FML(0.73900),   FML(0.78633),   FML(0.47165),
    };
    u32 inputRows = 3;
    u32 inputCols = 3;

    fml correctOutput_k33[] =
    {  FML(1.62048),   FML(2.15582),   FML(1.50069),
   FML(2.43101),   FML(3.02276),   FML(2.07634),
   FML(1.31329),   FML(1.70463),   FML(0.99219),
    };

    fml correctOutput_k55[] =
    {  FML(4.0384),   FML(3.7583),   FML(3.5239),
   FML(3.6871),   FML(3.1761),   FML(2.7691),
   FML(3.7347),   FML(3.4253),   FML(3.4902),
    };

    fml correctOutput_k57[] =
    {  FML(2.9735),   FML(3.3961),   FML(3.9519),
   FML(3.0722),   FML(3.1828),   FML(3.4366),
   FML(3.4359),   FML(4.5572),   FML(3.8323),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test20(const tTest& t)
{
    fml input[] =
    {  FML(0.130624),   FML(0.042344),   FML(0.451024),   FML(0.171283),
   FML(0.466452),   FML(0.752286),   FML(0.805553),   FML(0.788656),
   FML(0.512106),   FML(0.928868),   FML(0.934330),   FML(0.012693),
    };
    u32 inputRows = 3;
    u32 inputCols = 4;

    fml correctOutput_k33[] =
    {  FML(1.1930),   FML(1.8536),   FML(2.3558),   FML(1.6606),
   FML(1.7687),   FML(3.0085),   FML(2.7391),   FML(1.9951),
   FML(1.1863),   FML(1.9242),   FML(2.0008),   FML(1.0277),
    };

    fml correctOutput_k55[] =
    {  FML(3.3174),   FML(4.3385),   FML(4.6255),   FML(3.2533),
   FML(3.9096),   FML(4.3653),   FML(3.5082),   FML(2.5513),
   FML(3.2750),   FML(3.0896),   FML(2.9967),   FML(2.3234),
    };

    fml correctOutput_k57[] =
    {  FML(3.7866),   FML(3.6553),   FML(3.5042),   FML(4.1753),
   FML(3.1315),   FML(2.8611),   FML(3.3873),   FML(4.1962),
   FML(3.1399),   FML(3.8528),   FML(4.0769),   FML(4.1287),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test21(const tTest& t)
{
    fml input[] =
    {  FML(0.014151),   FML(0.286512),   FML(0.772909),   FML(0.062187),   FML(0.311414),
   FML(0.243021),   FML(0.525094),   FML(0.635263),   FML(0.599077),   FML(0.591995),
   FML(0.427144),   FML(0.701374),   FML(0.612650),   FML(0.944313),   FML(0.953119),
    };
    u32 inputRows = 3;
    u32 inputCols = 5;

    fml correctOutput_k33[] =
    {  FML(0.82465),   FML(1.59374),   FML(2.16069),   FML(1.82071),   FML(1.38631),
   FML(1.41308),   FML(2.45356),   FML(2.95983),   FML(3.29190),   FML(2.34666),
   FML(0.92989),   FML(1.43656),   FML(1.63865),   FML(1.91432),   FML(1.54591),
    };

    fml correctOutput_k55[] =
    {  FML(2.9836),   FML(3.8949),   FML(4.7089),   FML(4.1807),   FML(3.9644),
   FML(3.2021),   FML(4.2019),   FML(4.7533),   FML(4.4108),   FML(3.2652),
   FML(3.0445),   FML(3.5947),   FML(4.5873),   FML(4.0264),   FML(2.9334),
    };

    fml correctOutput_k57[] =
    {  FML(3.5731),   FML(4.4268),   FML(4.6032),   FML(4.6777),   FML(4.8550),
   FML(3.5190),   FML(4.5404),   FML(4.1475),   FML(4.1518),   FML(3.9747),
   FML(3.1819),   FML(4.3949),   FML(4.5446),   FML(4.9930),   FML(4.3596),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test22(const tTest& t)
{
    fml input[] =
    {  FML(0.368099),   FML(0.314494),   FML(0.123600),   FML(0.718517),   FML(0.038167),   FML(0.635907),
   FML(0.987620),   FML(0.532174),   FML(0.285569),   FML(0.938679),   FML(0.029324),   FML(0.120869),
   FML(0.347388),   FML(0.077237),   FML(0.545377),   FML(0.961921),   FML(0.564045),   FML(0.535741),
    };
    u32 inputRows = 3;
    u32 inputCols = 6;

    fml correctOutput_k33[] =
    {  FML(1.63763),   FML(1.93849),   FML(1.79254),   FML(1.71165),   FML(1.35120),   FML(0.79804),
   FML(1.56655),   FML(1.87443),   FML(2.31025),   FML(2.79164),   FML(2.66033),   FML(1.37139),
   FML(0.97926),   FML(1.22041),   FML(1.54273),   FML(1.52486),   FML(1.43157),   FML(0.74280),
    };

    fml correctOutput_k55[] =
    {  FML(2.9637),   FML(3.9447),   FML(3.8538),   FML(4.5039),   FML(3.2275),   FML(3.0027),
   FML(2.5539),   FML(3.7539),   FML(4.3199),   FML(3.8500),   FML(3.2544),   FML(2.5594),
   FML(2.1908),   FML(3.8980),   FML(4.4127),   FML(3.9849),   FML(3.3576),   FML(2.8836),
    };

    fml correctOutput_k57[] =
    {  FML(3.6632),   FML(4.0666),   FML(4.2689),   FML(4.8780),   FML(4.1384),   FML(4.1643),
   FML(4.1040),   FML(4.3013),   FML(4.5114),   FML(4.8441),   FML(4.0541),   FML(3.1549),
   FML(3.4195),   FML(4.1466),   FML(4.2295),   FML(4.6521),   FML(4.6958),   FML(3.4405),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test23(const tTest& t)
{
    fml input[] =
     {  FML(0.1517139),   FML(0.7654372),   FML(0.3616887),   FML(0.4911390),   FML(0.3479146),   FML(0.5811299),  FML(0.7580701),
    FML(0.9793864),   FML(0.6448141),   FML(0.8979641),   FML(0.3243579),   FML(0.7350552),   FML(0.4455876),  FML(0.6888704),
    FML(0.3094254),   FML(0.6165461),   FML(0.3668912),   FML(0.6573423),   FML(0.3628784),   FML(0.0068594),  FML(0.9648418),
     };
    u32 inputRows = 3;
    u32 inputCols = 7;

    fml correctOutput_k33[] =
    {  FML(1.6294),   FML(2.7820),   FML(2.1052),   FML(2.1944),   FML(1.7672),   FML(2.2468),   FML(1.7339),
   FML(2.0588),   FML(2.4027),   FML(2.9908),   FML(2.2360),   FML(2.2988),   FML(2.4243),   FML(2.0271),
   FML(1.0862),   FML(1.8570),   FML(1.4240),   FML(1.6935),   FML(1.1490),   FML(1.2472),   FML(1.3543),
    };

    fml correctOutput_k55[] =
    {  FML(3.7758),   FML(4.6049),   FML(4.9939),   FML(4.9813),   FML(5.3338),   FML(3.5398),   FML(3.4212),
   FML(3.7336),   FML(3.7214),   FML(4.6981),   FML(4.1653),   FML(4.6327),   FML(3.4721),   FML(2.7374),
   FML(2.9073),   FML(4.1317),   FML(4.5484),   FML(4.7310),   FML(5.0200),   FML(3.9219),   FML(3.1446),
    };

    fml correctOutput_k57[] =
    {  FML(3.7458),   FML(4.4080),   FML(5.4517),   FML(6.5170),   FML(5.8715),   FML(4.7483),   FML(4.4153),
   FML(3.7639),   FML(4.4246),   FML(5.0344),   FML(6.2773),   FML(6.1287),   FML(4.3819),   FML(3.9862),
   FML(3.6307),   FML(5.0625),   FML(5.0327),   FML(6.4479),   FML(5.4658),   FML(5.1593),   FML(4.3397),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test24(const tTest& t)
{
    fml input[] =
     {  FML(0.819852),   FML(0.639513),   FML(0.177243),   FML(0.090697),   FML(0.201485),   FML(0.229824),   FML(0.676013),  FML(0.536134),
    FML(0.868819),   FML(0.157813),   FML(0.089042),   FML(0.514470),   FML(0.115112),   FML(0.461106),   FML(0.820230),  FML(0.815445),
    FML(0.992130),   FML(0.994394),   FML(0.301495),   FML(0.351057),   FML(0.610951),   FML(0.283036),   FML(0.235369),  FML(0.054359),
     };
    u32 inputRows = 3;
    u32 inputCols = 8;

    fml correctOutput_k33[] =
    {  FML(1.66859),   FML(1.74069),   FML(1.01997),   FML(0.87651),   FML(1.25531),   FML(1.54238),   FML(2.39622),   FML(1.99234),
   FML(2.80014),   FML(2.94688),   FML(2.12450),   FML(1.76964),   FML(1.59830),   FML(2.04361),   FML(2.00347),   FML(1.68890),
   FML(1.41369),   FML(1.69496),   FML(0.97069),   FML(0.86084),   FML(1.26408),   FML(1.08653),   FML(1.25756),   FML(0.96598),
    };

    fml correctOutput_k55[] =
    {  FML(3.4364),   FML(3.9781),   FML(3.9517),   FML(3.1598),   FML(3.7805),   FML(4.4170),   FML(3.6370),   FML(2.7445),
   FML(3.4675),   FML(3.6012),   FML(4.0497),   FML(3.5370),   FML(3.3652),   FML(3.7393),   FML(2.7179),   FML(1.9679),
   FML(3.2794),   FML(3.7656),   FML(4.4764),   FML(3.1513),   FML(3.2832),   FML(3.4858),   FML(2.9476),   FML(2.4537),
    };

    fml correctOutput_k57[] =
    {  FML(3.7493),   FML(4.5533),   FML(4.9695),   FML(6.6073),   FML(5.8223),   FML(4.0579),   FML(3.4268),   FML(3.6674),
   FML(3.2403),   FML(4.2305),   FML(4.5244),   FML(5.3717),   FML(4.2603),   FML(3.5283),   FML(3.6481),   FML(3.4857),
   FML(3.8232),   FML(5.0198),   FML(4.1517),   FML(5.1088),   FML(4.9953),   FML(4.1111),   FML(4.4533),   FML(3.8316),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test25(const tTest& t)
{
    fml input[] =
    {  FML(0.69678),
   FML(0.24998),
   FML(0.90032),
   FML(0.74011),
    };
    u32 inputRows = 4;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(0.90664),
   FML(1.25408),
   FML(1.48415),
   FML(1.00755),
    };

    fml correctOutput_k55[] =
    {  FML(1.8823),
   FML(2.0979),
   FML(2.2310),
   FML(1.0711),
    };

    fml correctOutput_k57[] =
    {  FML(1.7368),
   FML(1.7703),
   FML(2.3255),
   FML(1.7757),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test26(const tTest& t)
{
    fml input[] =
    {  FML(0.338906),   FML(0.839243),
   FML(0.539805),   FML(0.716136),
   FML(0.088687),   FML(0.831425),
   FML(0.419003),   FML(0.317084),
    };
    u32 inputRows = 4;
    u32 inputCols = 2;

    fml correctOutput_k33[] =
    {  FML(1.50612),   FML(1.85426),
   FML(1.82427),   FML(1.92323),
   FML(1.39930),   FML(1.91047),
   FML(0.94286),   FML(0.79310),
    };

    fml correctOutput_k55[] =
    {  FML(2.2773),   FML(2.7607),
   FML(2.7434),   FML(2.5400),
   FML(2.8908),   FML(2.6788),
   FML(2.2776),   FML(1.8684),
    };

    fml correctOutput_k57[] =
    {  FML(2.0285),   FML(2.4139),
   FML(2.2929),   FML(2.8384),
   FML(2.7016),   FML(3.4053),
   FML(2.6747),   FML(2.6885),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test27(const tTest& t)
{
    fml input[] =
    {  FML(0.31782),   FML(0.69340),   FML(0.25303),
   FML(0.66754),   FML(0.62294),   FML(0.32094),
   FML(0.11051),   FML(0.92770),   FML(0.77245),
   FML(0.12994),   FML(0.46619),   FML(0.49688),
    };
    u32 inputRows = 4;
    u32 inputCols = 3;

    fml correctOutput_k33[] =
    {  FML(1.49407),   FML(2.05261),   FML(1.26207),
   FML(1.91949),   FML(2.61633),   FML(2.33277),
   FML(1.32220),   FML(2.50870),   FML(2.10190),
   FML(0.80352),   FML(1.29827),   FML(1.36946),
    };

    fml correctOutput_k55[] =
    {  FML(2.9239),   FML(3.1715),   FML(3.2925),
   FML(3.5862),   FML(3.4759),   FML(3.5645),
   FML(4.0270),   FML(3.8367),   FML(3.5966),
   FML(3.2355),   FML(2.8803),   FML(2.6952),
    };

    fml correctOutput_k57[] =
    {  FML(2.6339),   FML(2.9949),   FML(3.3548),
   FML(3.2740),   FML(3.2375),   FML(3.7353),
   FML(3.3522),   FML(3.8936),   FML(3.8451),
   FML(3.1751),   FML(3.4831),   FML(3.3908),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test28(const tTest& t)
{
    fml input[] =
    {  FML(0.59977),   FML(0.73296),   FML(0.74394),   FML(0.25231),
   FML(0.85450),   FML(0.66155),   FML(0.30469),   FML(0.48705),
   FML(0.46408),   FML(0.80144),   FML(0.97425),   FML(0.81620),
   FML(0.27202),   FML(0.16620),   FML(0.77191),   FML(0.84459),
    };
    u32 inputRows = 4;
    u32 inputCols = 4;

    fml correctOutput_k33[] =
    {  FML(1.87012),   FML(2.37966),   FML(2.03228),   FML(1.12570),
   FML(2.32727),   FML(3.38231),   FML(3.29207),   FML(2.50274),
   FML(1.51851),   FML(2.70368),   FML(3.13256),   FML(2.52481),
   FML(0.90367),   FML(1.38767),   FML(1.99017),   FML(1.70683),
    };

    fml correctOutput_k55[] =
    {  FML(4.1216),   FML(4.6217),   FML(4.5815),   FML(3.8003),
   FML(4.4240),   FML(5.3060),   FML(5.2730),   FML(4.7955),
   FML(5.2128),   FML(6.4375),   FML(6.3983),   FML(4.6991),
   FML(3.5528),   FML(4.7751),   FML(4.4218),   FML(3.1510),
    };

    fml correctOutput_k57[] =
    {  FML(4.0302),   FML(4.4091),   FML(4.8080),   FML(5.2123),
   FML(5.3797),   FML(5.3027),   FML(5.0661),   FML(6.1353),
   FML(5.7590),   FML(6.1389),   FML(5.3494),   FML(5.5162),
   FML(4.3767),   FML(4.7088),   FML(4.4973),   FML(4.3266),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test29(const tTest& t)
{
    fml input[] =
    {  FML(0.256924),   FML(0.912201),   FML(0.236915),   FML(0.999137),   FML(0.427073),
   FML(0.160530),   FML(0.939540),   FML(0.774395),   FML(0.894989),   FML(0.477652),
   FML(0.359979),   FML(0.046701),   FML(0.856326),   FML(0.939925),   FML(0.813082),
   FML(0.463890),   FML(0.233237),   FML(0.424757),   FML(0.021674),   FML(0.548801),
    };
    u32 inputRows = 4;
    u32 inputCols = 5;

    fml correctOutput_k33[] =
    {  FML(1.33872),   FML(2.32980),   FML(2.68101),   FML(2.69109),   FML(1.77739),
   FML(1.23668),   FML(2.48777),   FML(3.37979),   FML(3.63035),   FML(2.74315),
   FML(1.42461),   FML(1.97841),   FML(2.62173),   FML(2.79479),   FML(1.99219),
   FML(0.71339),   FML(1.09081),   FML(1.21901),   FML(1.45365),   FML(1.35692),
    };

    fml correctOutput_k55[] =
    {  FML(3.3437),   FML(4.9017),   FML(5.3759),   FML(5.3642),   FML(4.0828),
   FML(3.8500),   FML(4.9328),   FML(6.4710),   FML(5.0035),   FML(4.0883),
   FML(4.2007),   FML(5.5589),   FML(7.0315),   FML(5.5653),   FML(4.4595),
   FML(3.0397),   FML(4.1525),   FML(5.1651),   FML(4.3500),   FML(3.5907),
    };

    fml correctOutput_k57[] =
    {  FML(4.0499),   FML(5.1338),   FML(4.5300),   FML(5.6458),   FML(5.3729),
   FML(4.9147),   FML(6.0901),   FML(5.7519),   FML(5.6218),   FML(6.3068),
   FML(4.7122),   FML(5.6272),   FML(6.7591),   FML(5.7197),   FML(5.8359),
   FML(3.6278),   FML(5.1223),   FML(5.2629),   FML(5.3437),   FML(4.1195),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test30(const tTest& t)
{
    fml input[] =
    {  FML(0.61992),   FML(0.41954),   FML(0.34088),   FML(0.95166),   FML(0.11010),   FML(0.19466),
   FML(0.60851),   FML(0.76563),   FML(0.14351),   FML(0.74863),   FML(0.49072),   FML(0.98324),
   FML(0.31447),   FML(0.86521),   FML(0.83952),   FML(0.38108),   FML(0.33436),   FML(0.45006),
   FML(0.56596),   FML(0.81475),   FML(0.69872),   FML(0.86022),   FML(0.45513),   FML(0.25696),
    };
    u32 inputRows = 4;
    u32 inputCols = 6;

    fml correctOutput_k33[] =
    {  FML(1.72558),   FML(1.84447),   FML(1.96182),   FML(1.99529),   FML(2.15540),   FML(1.50526),
   FML(1.97742),   FML(2.97628),   FML(2.88707),   FML(2.57877),   FML(2.36196),   FML(1.73025),
   FML(2.06874),   FML(3.20256),   FML(3.58905),   FML(2.70487),   FML(2.77648),   FML(1.70532),
   FML(1.21039),   FML(1.77333),   FML(1.85626),   FML(1.72613),   FML(1.20932),   FML(0.78111),
    };

    fml correctOutput_k55[] =
    {  FML(3.1568),   FML(4.7113),   FML(5.0753),   FML(5.2232),   FML(3.5832),   FML(3.3460),
   FML(3.9982),   FML(5.7812),   FML(6.3619),   FML(6.3648),   FML(4.7616),   FML(3.6087),
   FML(5.1898),   FML(6.7228),   FML(7.2774),   FML(6.6281),   FML(5.0832),   FML(3.4353),
   FML(3.7312),   FML(4.9480),   FML(5.3176),   FML(5.6621),   FML(4.0994),   FML(3.2638),
    };

    fml correctOutput_k57[] =
    {  FML(3.8994),   FML(4.7923),   FML(5.7317),   FML(6.2209),   FML(4.8532),   FML(4.2987),
   FML(5.3145),   FML(6.0848),   FML(6.9631),   FML(7.5631),   FML(6.9486),   FML(5.0729),
   FML(5.3157),   FML(6.6376),   FML(7.2758),   FML(6.7529),   FML(6.7390),   FML(4.4079),
   FML(4.2785),   FML(5.1520),   FML(5.8406),   FML(5.9788),   FML(6.0195),   FML(4.7164),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test31(const tTest& t)
{
    fml input[] =
    {  FML(0.626555),   FML(0.876820),   FML(0.285900),   FML(0.780896),   FML(0.834180),   FML(0.777396),   FML(0.617987),
   FML(0.683797),   FML(0.885153),   FML(0.892566),   FML(0.974930),   FML(0.861094),   FML(0.543050),   FML(0.619226),
   FML(0.827029),   FML(0.064779),   FML(0.983145),   FML(0.192497),   FML(0.023953),   FML(0.634787),   FML(0.297135),
   FML(0.732353),   FML(0.798278),   FML(0.119266),   FML(0.276429),   FML(0.037861),   FML(0.135628),   FML(0.422648),
    };
    u32 inputRows = 4;
    u32 inputCols = 7;

    fml correctOutput_k33[] =
    {  FML(1.94674),   FML(2.84920),   FML(2.77249),   FML(3.03655),   FML(2.90208),   FML(2.56808),   FML(1.69244),
   FML(2.05241),   FML(3.25165),   FML(2.95437),   FML(2.90381),   FML(2.68959),   FML(2.46904),   FML(2.09262),
   FML(2.48826),   FML(2.82387),   FML(3.05576),   FML(1.95405),   FML(1.71887),   FML(2.03691),   FML(1.42154),
   FML(1.14178),   FML(1.80252),   FML(0.86299),   FML(1.03937),   FML(0.68278),   FML(0.70580),   FML(0.97246),
    };

    fml correctOutput_k55[] =
    {  FML(4.5822),   FML(5.3465),   FML(6.5838),   FML(6.5589),   FML(5.4932),   FML(4.3829),   FML(3.4016),
   FML(4.9560),   FML(5.8490),   FML(6.5861),   FML(5.5867),   FML(5.1833),   FML(3.9409),   FML(3.0844),
   FML(5.4204),   FML(6.1128),   FML(7.0585),   FML(6.6365),   FML(6.3103),   FML(5.3591),   FML(3.8001),
   FML(3.7978),   FML(5.2965),   FML(5.4533),   FML(5.2455),   FML(5.3747),   FML(3.6731),   FML(2.7099),
    };

    fml correctOutput_k57[] =
    {  FML(4.4278),   FML(5.9066),   FML(6.1526),   FML(7.5727),   FML(6.3027),   FML(5.5367),   FML(4.8153),
   FML(5.4642),   FML(5.7937),   FML(7.5503),   FML(8.7647),   FML(7.8103),   FML(5.8404),   FML(5.2579),
   FML(5.3208),   FML(6.5606),   FML(7.6445),   FML(8.4670),   FML(7.2965),   FML(6.4682),   FML(4.5462),
   FML(4.2709),   FML(5.8530),   FML(5.5606),   FML(7.1127),   FML(5.0698),   FML(4.2428),   FML(3.3590),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test32(const tTest& t)
{
    fml input[] =
     {  FML(0.067732),   FML(0.662511),   FML(0.938444),   FML(0.517251),   FML(0.511806),   FML(0.485878),   FML(0.303237),  FML(0.623806),
    FML(0.110061),   FML(0.400040),   FML(0.622888),   FML(0.189843),   FML(0.416249),   FML(0.238959),   FML(0.182893),  FML(0.398467),
    FML(0.458002),   FML(0.747057),   FML(0.536994),   FML(0.240431),   FML(0.058904),   FML(0.669714),   FML(0.382004),  FML(0.390300),
    FML(0.375530),   FML(0.584177),   FML(0.668169),   FML(0.603765),   FML(0.546609),   FML(0.924854),   FML(0.083083),  FML(0.137673),
     };
    u32 inputRows = 4;
    u32 inputCols = 8;

    fml correctOutput_k33[] =
    {  FML(0.74061),   FML(1.69180),   FML(2.01217),   FML(1.77594),   FML(1.38230),   FML(1.35119),   FML(1.23164),   FML(1.16158),
   FML(1.52078),   FML(2.54877),   FML(2.78602),   FML(2.05141),   FML(1.93094),   FML(1.83461),   FML(2.17413),   FML(1.47876),
   FML(1.55964),   FML(2.56376),   FML(2.63665),   FML(2.43353),   FML(2.28572),   FML(2.33771),   FML(1.93789),   FML(0.94044),
   FML(1.02476),   FML(1.49095),   FML(1.54858),   FML(1.23296),   FML(1.28544),   FML(1.36488),   FML(1.07392),   FML(0.64623),
    };

    fml correctOutput_k55[] =
    {  FML(3.1458),   FML(4.0042),   FML(4.4630),   FML(4.2476),   FML(3.4836),   FML(3.6113),   FML(3.0098),   FML(2.5482),
   FML(3.8316),   FML(4.4166),   FML(4.8165),   FML(5.3311),   FML(4.7899),   FML(4.8650),   FML(3.7848),   FML(2.5796),
   FML(4.7664),   FML(5.4743),   FML(6.1038),   FML(7.4469),   FML(6.3426),   FML(5.7415),   FML(4.0897),   FML(3.0803),
   FML(3.1143),   FML(3.3851),   FML(4.1345),   FML(4.6994),   FML(3.9525),   FML(3.2516),   FML(2.5056),   FML(2.0875),
    };

    fml correctOutput_k57[] =
    {  FML(3.1387),   FML(3.6542),   FML(4.7941),   FML(5.1383),   FML(6.1387),   FML(4.9749),   FML(3.8181),   FML(3.4553),
   FML(4.0929),   FML(4.7834),   FML(6.6139),   FML(7.3474),   FML(7.6312),   FML(6.4142),   FML(4.9773),   FML(4.3105),
   FML(4.4410),   FML(5.6663),   FML(7.2451),   FML(7.3850),   FML(6.1952),   FML(5.8875),   FML(4.5807),   FML(3.9509),
   FML(3.1135),   FML(3.9979),   FML(4.9298),   FML(5.2146),   FML(5.0437),   FML(4.6249),   FML(4.0097),   FML(3.0272),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test33(const tTest& t)
{
    fml input[] =
    {  FML(0.11413),
   FML(0.78066),
   FML(0.17618),
   FML(0.48251),
   FML(0.57367),
    };
    u32 inputRows = 5;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(0.87202),
   FML(0.94557),
   FML(0.91764),
   FML(1.03759),
   FML(0.76926),
    };

    fml correctOutput_k55[] =
    {  FML(1.38345),
   FML(1.48217),
   FML(1.60507),
   FML(1.89927),
   FML(0.90715),
    };

    fml correctOutput_k57[] =
    {  FML(1.2067),
   FML(1.6233),
   FML(1.6432),
   FML(2.0485),
   FML(1.4829),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test34(const tTest& t)
{
    fml input[] =
    {  FML(0.45632),   FML(0.39425),
   FML(0.18853),   FML(0.12748),
   FML(0.47297),   FML(0.34893),
   FML(0.51018),   FML(0.38762),
   FML(0.78791),   FML(0.74799),
    };
    u32 inputRows = 5;
    u32 inputCols = 2;

    fml correctOutput_k33[] =
    {  FML(0.84294),   FML(0.82421),
   FML(1.25971),   FML(1.34667),
   FML(1.37291),   FML(1.39077),
   FML(2.03636),   FML(2.13681),
   FML(1.21822),   FML(1.26714),
    };

    fml correctOutput_k55[] =
    {  FML(1.6258),   FML(1.6525),
   FML(2.0244),   FML(2.0272),
   FML(3.1690),   FML(3.4919),
   FML(2.7343),   FML(2.3358),
   FML(2.1681),   FML(1.9835),
    };

    fml correctOutput_k57[] =
    {  FML(1.7058),   FML(1.9343),
   FML(1.9387),   FML(2.3522),
   FML(3.0859),   FML(3.5176),
   FML(2.0889),   FML(2.6838),
   FML(2.5355),   FML(3.0099),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test35(const tTest& t)
{
    fml input[] =
    {  FML(0.904452),   FML(0.508572),   FML(0.409249),
   FML(0.650574),   FML(0.035667),   FML(0.737553),
   FML(0.776809),   FML(0.632391),   FML(0.482608),
   FML(0.879195),   FML(0.691875),   FML(0.685167),
   FML(0.108347),   FML(0.959249),   FML(0.365523),
    };
    u32 inputRows = 5;
    u32 inputCols = 3;

    fml correctOutput_k33[] =
    {  FML(1.46175),   FML(1.88461),   FML(1.16294),
   FML(2.17664),   FML(2.72126),   FML(2.03031),
   FML(2.21888),   FML(3.26948),   FML(1.97048),
   FML(2.20929),   FML(2.82000),   FML(2.36755),
   FML(0.98869),   FML(1.94605),   FML(1.19738),
    };

    fml correctOutput_k55[] =
    {  FML(3.6453),   FML(3.3006),   FML(3.4308),
   FML(4.4784),   FML(4.2736),   FML(4.4175),
   FML(5.5320),   FML(6.1101),   FML(5.6532),
   FML(4.8659),   FML(4.4344),   FML(4.0304),
   FML(3.4979),   FML(3.5616),   FML(3.2269),
    };

    fml correctOutput_k57[] =
    {  FML(3.3685),   FML(3.2095),   FML(3.5920),
   FML(3.8333),   FML(4.4217),   FML(4.4427),
   FML(5.2079),   FML(5.6736),   FML(5.0368),
   FML(3.7250),   FML(4.5744),   FML(4.1481),
   FML(3.4600),   FML(4.5670),   FML(3.5141),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test36(const tTest& t)
{
    fml input[] =
    {  FML(0.102043),   FML(0.644812),   FML(0.630665),   FML(0.632529),
   FML(0.254845),   FML(0.090823),   FML(0.259359),   FML(0.470102),
   FML(0.050791),   FML(0.013101),   FML(0.311491),   FML(0.226057),
   FML(0.575180),   FML(0.731398),   FML(0.934227),   FML(0.306934),
   FML(0.845803),   FML(0.991242),   FML(0.370971),   FML(0.968637),
    };
    u32 inputRows = 5;
    u32 inputCols = 4;

    fml correctOutput_k33[] =
    {  FML(0.64997),   FML(1.26950),   FML(1.47885),   FML(1.33454),
   FML(0.75876),   FML(1.13454),   FML(1.68515),   FML(1.54337),
   FML(1.30404),   FML(2.23388),   FML(2.31135),   FML(1.70562),
   FML(2.09853),   FML(2.84786),   FML(3.03174),   FML(1.83535),
   FML(1.46132),   FML(2.03275),   FML(1.58660),   FML(1.57078),
    };

    fml correctOutput_k55[] =
    {  FML(2.0438),   FML(2.6353),   FML(2.4558),   FML(2.2863),
   FML(2.4373),   FML(3.3153),   FML(3.7457),   FML(2.9796),
   FML(4.7719),   FML(6.6974),   FML(6.1063),   FML(5.1082),
   FML(4.5094),   FML(5.3899),   FML(4.6519),   FML(3.5977),
   FML(2.7912),   FML(3.5914),   FML(2.9727),   FML(2.6995),
    };

    fml correctOutput_k57[] =
    {  FML(2.3392),   FML(2.4438),   FML(2.3601),   FML(2.7954),
   FML(3.8019),   FML(3.7419),   FML(3.9433),   FML(4.3691),
   FML(5.4171),   FML(5.3425),   FML(6.2016),   FML(6.0588),
   FML(3.9248),   FML(4.4269),   FML(4.1997),   FML(4.7538),
   FML(3.3205),   FML(3.8890),   FML(3.7963),   FML(4.7594),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test37(const tTest& t)
{
    fml input[] =
    {  FML(0.353102),   FML(0.792772),   FML(0.187947),   FML(0.849217),   FML(0.191495),
   FML(0.790903),   FML(0.078293),   FML(0.208784),   FML(0.747796),   FML(0.028499),
   FML(0.483566),   FML(0.966369),   FML(0.763947),   FML(0.753788),   FML(0.922198),
   FML(0.839690),   FML(0.381426),   FML(0.205740),   FML(0.997533),   FML(0.778221),
   FML(0.312394),   FML(0.618948),   FML(0.020907),   FML(0.847944),   FML(0.994713),
    };
    u32 inputRows = 5;
    u32 inputCols = 5;

    fml correctOutput_k33[] =
    {  FML(1.24145),   FML(1.74362),   FML(1.36489),   FML(1.63770),   FML(1.13099),
   FML(2.26982),   FML(2.57403),   FML(3.23852),   FML(3.05889),   FML(2.17128),
   FML(1.86995),   FML(2.78421),   FML(2.63171),   FML(2.82507),   FML(2.81906),
   FML(2.09308),   FML(2.23570),   FML(2.77321),   FML(3.45368),   FML(3.03987),
   FML(0.94074),   FML(1.36563),   FML(1.14571),   FML(1.71610),   FML(1.82966),
    };

    fml correctOutput_k55[] =
    {  FML(2.8688),   FML(4.5064),   FML(4.6710),   FML(3.9293),   FML(3.0159),
   FML(4.2376),   FML(5.4941),   FML(6.0962),   FML(5.0886),   FML(4.3123),
   FML(4.6634),   FML(7.5088),   FML(8.5155),   FML(6.6980),   FML(5.8945),
   FML(3.7667),   FML(5.7625),   FML(6.8839),   FML(5.7601),   FML(4.1398),
   FML(3.1128),   FML(4.8949),   FML(6.7942),   FML(5.2755),   FML(3.8063),
    };

    fml correctOutput_k57[] =
    {  FML(4.1232),   FML(4.8376),   FML(4.4497),   FML(5.4445),   FML(4.9539),
   FML(5.3846),   FML(6.1928),   FML(6.3676),   FML(6.3763),   FML(5.6535),
   FML(6.0532),   FML(8.3666),   FML(7.6713),   FML(7.6160),   FML(7.4173),
   FML(5.0648),   FML(7.1439),   FML(6.1803),   FML(6.0847),   FML(5.5185),
   FML(4.1399),   FML(6.1014),   FML(5.5990),   FML(5.9235),   FML(5.4297),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test38(const tTest& t)
{
    fml input[] =
    {  FML(0.066333),   FML(0.341324),   FML(0.139016),   FML(0.134913),   FML(0.072594),   FML(0.722927),
   FML(0.633567),   FML(0.084770),   FML(0.651344),   FML(0.930865),   FML(0.822862),   FML(0.614186),
   FML(0.811559),   FML(0.990176),   FML(0.953959),   FML(0.028994),   FML(0.409567),   FML(0.979954),
   FML(0.297410),   FML(0.801987),   FML(0.542301),   FML(0.249281),   FML(0.090030),   FML(0.903561),
   FML(0.543789),   FML(0.031032),   FML(0.311690),   FML(0.489969),   FML(0.758451),   FML(0.221111),
    };
    u32 inputRows = 5;
    u32 inputCols = 6;

    fml correctOutput_k33[] =
    {  FML(0.84655),   FML(1.54972),   FML(1.59968),   FML(2.15616),   FML(2.22539),   FML(1.89005),
   FML(2.14681),   FML(2.75168),   FML(2.68471),   FML(2.38133),   FML(2.47101),   FML(2.09206),
   FML(1.97096),   FML(3.07734),   FML(2.92357),   FML(2.10607),   FML(2.58567),   FML(2.34831),
   FML(1.62473),   FML(2.74522),   FML(2.19372),   FML(2.34678),   FML(2.12841),   FML(2.17927),
   FML(1.03212),   FML(0.98542),   FML(1.22019),   FML(1.14910),   FML(1.41430),   FML(0.79646),
    };

    fml correctOutput_k55[] =
    {  FML(2.9436),   FML(4.0188),   FML(5.1484),   FML(5.1394),   FML(3.8028),   FML(3.6210),
   FML(4.3635),   FML(5.1332),   FML(6.0127),   FML(5.8595),   FML(4.4376),   FML(3.5976),
   FML(4.8811),   FML(5.2673),   FML(5.7144),   FML(6.8357),   FML(5.8828),   FML(4.6294),
   FML(4.3196),   FML(5.3409),   FML(6.3573),   FML(7.3058),   FML(6.1508),   FML(4.1715),
   FML(3.9675),   FML(4.0091),   FML(5.2522),   FML(5.2924),   FML(3.9437),   FML(2.3390),
    };

    fml correctOutput_k57[] =
    {  FML(3.9206),   FML(4.9737),   FML(5.5742),   FML(5.4012),   FML(4.8859),   FML(4.4783),
   FML(4.1789),   FML(4.2813),   FML(7.0962),   FML(6.9020),   FML(6.1244),   FML(5.0275),
   FML(4.5604),   FML(6.3266),   FML(7.6864),   FML(8.6956),   FML(6.7515),   FML(6.4804),
   FML(4.4932),   FML(6.4644),   FML(7.2184),   FML(7.3761),   FML(6.3648),   FML(5.5438),
   FML(4.1995),   FML(5.0022),   FML(5.3494),   FML(4.9897),   FML(4.4227),   FML(3.9180),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test39(const tTest& t)
{
    fml input[] =
    {  FML(0.55151),   FML(0.76868),   FML(0.52278),   FML(0.82389),   FML(0.81103),   FML(0.51768),   FML(0.30668),
   FML(0.93350),   FML(0.54588),   FML(0.64591),   FML(0.99656),   FML(0.37232),   FML(0.42777),   FML(0.54744),
   FML(0.52793),   FML(0.27563),   FML(0.14945),   FML(0.76214),   FML(0.13469),   FML(0.60170),   FML(0.18632),
   FML(0.11638),   FML(0.61790),   FML(0.44682),   FML(0.13066),   FML(0.38627),   FML(0.57836),   FML(0.13707),
   FML(0.22107),   FML(0.12014),   FML(0.66569),   FML(0.47100),   FML(0.82189),   FML(0.56782),   FML(0.27775),
    };
    u32 inputRows = 5;
    u32 inputCols = 7;

    fml correctOutput_k33[] =
    {  FML(1.81707),   FML(2.58367),   FML(2.48685),   FML(2.56468),   FML(2.42649),   FML(1.78061),   FML(1.27737),
   FML(2.04135),   FML(2.34414),   FML(2.71974),   FML(2.73145),   FML(2.76052),   FML(2.07369),   FML(1.68568),
   FML(1.65676),   FML(2.31571),   FML(2.31965),   FML(2.34185),   FML(2.15548),   FML(2.09737),   FML(1.41306),
   FML(0.91391),   FML(1.92752),   FML(2.08623),   FML(2.27762),   FML(2.71125),   FML(2.33669),   FML(1.46097),
   FML(0.68550),   FML(0.87062),   FML(1.31251),   FML(1.23768),   FML(1.40004),   FML(1.23084),   FML(0.85802),
    };

    fml correctOutput_k55[] =
    {  FML(3.9772),   FML(5.1502),   FML(5.3864),   FML(5.5201),   FML(4.6529),   FML(3.9258),   FML(2.6804),
   FML(3.4749),   FML(4.8232),   FML(5.7258),   FML(5.4593),   FML(4.6756),   FML(4.4196),   FML(2.8364),
   FML(4.3601),   FML(6.3299),   FML(7.8737),   FML(7.9665),   FML(7.5566),   FML(6.5140),   FML(4.0924),
   FML(4.0597),   FML(5.5938),   FML(6.8345),   FML(6.5598),   FML(6.7433),   FML(4.8530),   FML(3.2465),
   FML(2.4557),   FML(3.2720),   FML(3.9095),   FML(4.3809),   FML(4.0804),   FML(2.9750),   FML(2.3178),
    };

    fml correctOutput_k57[] =
    {  FML(4.3786),   FML(4.8895),   FML(5.5537),   FML(6.4642),   FML(5.9548),   FML(4.6981),   FML(4.6998),
   FML(4.8630),   FML(5.8364),   FML(6.7017),   FML(7.8379),   FML(6.7642),   FML(5.7957),   FML(5.0697),
   FML(5.4879),   FML(7.4575),   FML(8.3180),   FML(9.7462),   FML(8.1619),   FML(7.2017),   FML(6.0665),
   FML(4.6943),   FML(6.2450),   FML(6.7233),   FML(6.5096),   FML(6.4175),   FML(4.7205),   FML(4.5410),
   FML(2.8030),   FML(3.4742),   FML(4.3393),   FML(4.4040),   FML(4.7278),   FML(4.2641),   FML(3.5967),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test40(const tTest& t)
{
    fml input[] =
     {  FML(0.218157),   FML(0.147869),   FML(0.023657),   FML(0.919928),   FML(0.706861),   FML(0.100598),   FML(0.111541),  FML(0.784131),
    FML(0.500598),   FML(0.306171),   FML(0.706288),   FML(0.490869),   FML(0.725474),   FML(0.097113),   FML(0.418351),  FML(0.879794),
    FML(0.596648),   FML(0.094601),   FML(0.638145),   FML(0.855039),   FML(0.285503),   FML(0.793443),   FML(0.360042),  FML(0.984837),
    FML(0.841410),   FML(0.664857),   FML(0.353879),   FML(0.765312),   FML(0.049259),   FML(0.223321),   FML(0.098071),  FML(0.065647),
    FML(0.093581),   FML(0.070960),   FML(0.017007),   FML(0.579227),   FML(0.311523),   FML(0.238023),   FML(0.141500),  FML(0.435166),
     };
    u32 inputRows = 5;
    u32 inputCols = 8;

    fml correctOutput_k33[] =
    {  FML(0.98372),   FML(1.50337),   FML(1.53111),   FML(2.46982),   FML(1.89891),   FML(1.38938),   FML(1.45475),   FML(1.79793),
   FML(1.25866),   FML(1.82255),   FML(2.47815),   FML(2.72778),   FML(3.05845),   FML(2.00101),   FML(2.71542),   FML(2.22359),
   FML(2.02067),   FML(2.57271),   FML(2.76086),   FML(2.69586),   FML(2.03083),   FML(1.76549),   FML(1.60352),   FML(1.59185),
   FML(1.26522),   FML(1.64457),   FML(1.76949),   FML(2.19904),   FML(2.15893),   FML(1.46371),   FML(1.88159),   FML(1.18814),
   FML(0.80743),   FML(1.07453),   FML(1.11295),   FML(1.11968),   FML(1.09317),   FML(0.59126),   FML(0.60478),   FML(0.62648),
    };

    fml correctOutput_k55[] =
    {  FML(2.6163),   FML(3.6704),   FML(4.7013),   FML(4.6784),   FML(4.1685),   FML(5.0406),   FML(3.8603),   FML(3.4984),
   FML(3.6009),   FML(4.8846),   FML(5.9182),   FML(5.2636),   FML(4.8837),   FML(5.7365),   FML(4.2613),   FML(2.9130),
   FML(3.7471),   FML(5.9001),   FML(6.3028),   FML(6.4438),   FML(5.8497),   FML(7.2794),   FML(4.4077),   FML(3.3006),
   FML(3.3191),   FML(4.9807),   FML(5.6564),   FML(5.1113),   FML(5.1696),   FML(5.4550),   FML(4.0130),   FML(2.9268),
   FML(2.1891),   FML(3.6951),   FML(4.1932),   FML(4.1284),   FML(4.1329),   FML(4.6809),   FML(3.1135),   FML(2.7696),
    };

    fml correctOutput_k57[] =
    {  FML(3.4865),   FML(4.2108),   FML(4.5341),   FML(5.2584),   FML(6.6796),   FML(5.7719),   FML(5.5201),   FML(4.2029),
   FML(4.9456),   FML(5.4926),   FML(6.3385),   FML(7.0600),   FML(7.9440),   FML(6.4484),   FML(5.2888),   FML(4.3701),
   FML(4.7035),   FML(5.5955),   FML(6.3674),   FML(7.9153),   FML(7.7772),   FML(7.3795),   FML(6.2129),   FML(5.2321),
   FML(4.4048),   FML(5.8410),   FML(5.6128),   FML(7.0711),   FML(6.5186),   FML(5.8171),   FML(5.3724),   FML(3.5451),
   FML(3.2107),   FML(4.1711),   FML(4.0847),   FML(4.8088),   FML(5.0684),   FML(3.8202),   FML(4.5487),   FML(3.0118),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test41(const tTest& t)
{
    fml input[] =
    {  FML(0.82139),
   FML(0.86681),
   FML(0.29727),
   FML(0.26126),
   FML(0.13910),
   FML(0.59101),
    };
    u32 inputRows = 6;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(1.44948),
   FML(1.29547),
   FML(0.86765),
   FML(0.59207),
   FML(0.82445),
   FML(0.68550),
    };

    fml correctOutput_k55[] =
    {  FML(1.98826),
   FML(1.45347),
   FML(1.66489),
   FML(1.97176),
   FML(1.31066),
   FML(0.98464),
    };

    fml correctOutput_k57[] =
    {  FML(1.7559),
   FML(1.8224),
   FML(2.1160),
   FML(2.1269),
   FML(1.4237),
   FML(1.4732),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test42(const tTest& t)
{
    fml input[] =
    {  FML(0.54738),   FML(0.56549),
   FML(0.35689),   FML(0.52208),
   FML(0.94704),   FML(0.60132),
   FML(0.49247),   FML(0.45762),
   FML(0.12302),   FML(0.40398),
   FML(0.91248),   FML(0.66396),
    };
    u32 inputRows = 6;
    u32 inputCols = 2;

    fml correctOutput_k33[] =
    {  FML(1.3405),   FML(1.3924),
   FML(2.0707),   FML(2.3373),
   FML(2.0067),   FML(1.8858),
   FML(1.5417),   FML(1.7062),
   FML(1.8398),   FML(2.1725),
   FML(1.1928),   FML(1.0199),
    };

    fml correctOutput_k55[] =
    {  FML(2.6926),   FML(2.8279),
   FML(3.0756),   FML(2.8592),
   FML(3.4331),   FML(3.2760),
   FML(3.2782),   FML(3.6998),
   FML(3.6137),   FML(3.3849),
   FML(2.2954),   FML(1.9285),
    };

    fml correctOutput_k57[] =
    {  FML(2.2503),   FML(2.7586),
   FML(2.4429),   FML(3.1193),
   FML(3.3943),   FML(3.7776),
   FML(3.4633),   FML(4.2981),
   FML(2.9991),   FML(3.7107),
   FML(2.6102),   FML(2.8070),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test43(const tTest& t)
{
    fml input[] =
    {  FML(0.968609),   FML(0.088877),   FML(0.690500),
   FML(0.190788),   FML(0.752156),   FML(0.426541),
   FML(0.930695),   FML(0.053554),   FML(0.917558),
   FML(0.549668),   FML(0.129983),   FML(0.415485),
   FML(0.035858),   FML(0.637935),   FML(0.604345),
   FML(0.753428),   FML(0.630288),   FML(0.149932),
    };
    u32 inputRows = 6;
    u32 inputCols = 3;

    fml correctOutput_k33[] =
    {  FML(1.60798),   FML(1.55819),   FML(1.67244),
   FML(1.51015),   FML(3.16442),   FML(1.60182),
   FML(1.76169),   FML(1.89077),   FML(1.83269),
   FML(1.39913),   FML(2.28923),   FML(1.80041),
   FML(1.55587),   FML(2.48995),   FML(1.57685),
   FML(1.14373),   FML(1.26411),   FML(0.93811),
    };

    fml correctOutput_k55[] =
    {  FML(3.8612),   FML(3.0205),   FML(3.3550),
   FML(3.7844),   FML(3.6975),   FML(3.4780),
   FML(5.0406),   FML(4.3763),   FML(5.0013),
   FML(4.3370),   FML(5.1063),   FML(4.4471),
   FML(4.5852),   FML(4.2980),   FML(3.6748),
   FML(2.8084),   FML(2.3442),   FML(2.1594),
    };

    fml correctOutput_k57[] =
    {  FML(2.7614),   FML(3.2148),   FML(3.3695),
   FML(3.3553),   FML(3.8991),   FML(3.3180),
   FML(4.6165),   FML(4.6903),   FML(4.7832),
   FML(4.3435),   FML(4.9206),   FML(4.3572),
   FML(3.1183),   FML(4.3181),   FML(3.4794),
   FML(2.8251),   FML(3.1442),   FML(2.9122),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test44(const tTest& t)
{
    fml input[] =
    {  FML(0.918216),   FML(0.084256),   FML(0.070449),   FML(0.630948),
   FML(0.423633),   FML(0.940319),   FML(0.826283),   FML(0.882588),
   FML(0.240336),   FML(0.945249),   FML(0.182619),   FML(0.628442),
   FML(0.211733),   FML(0.663051),   FML(0.306901),   FML(0.922462),
   FML(0.623959),   FML(0.059512),   FML(0.892999),   FML(0.938640),
   FML(0.521968),   FML(0.041437),   FML(0.281564),   FML(0.821435),
    };
    u32 inputRows = 6;
    u32 inputCols = 4;

    fml correctOutput_k33[] =
    {  FML(1.87440),   FML(2.05821),   FML(2.40094),   FML(2.02246),
   FML(1.81970),   FML(2.67722),   FML(2.80283),   FML(1.81381),
   FML(1.68148),   FML(2.68638),   FML(3.13703),   FML(2.32970),
   FML(1.43922),   FML(2.44963),   FML(2.88812),   FML(2.64291),
   FML(1.42574),   FML(1.59798),   FML(2.68132),   FML(2.29885),
   FML(0.80289),   FML(1.09816),   FML(1.23603),   FML(1.60492),
    };

    fml correctOutput_k55[] =
    {  FML(3.1393),   FML(4.9387),   FML(4.4237),   FML(3.6363),
   FML(3.4311),   FML(5.5084),   FML(4.7747),   FML(4.1026),
   FML(4.6463),   FML(6.6495),   FML(6.1787),   FML(5.6081),
   FML(5.4804),   FML(7.3655),   FML(7.2322),   FML(6.4899),
   FML(3.8802),   FML(5.3478),   FML(5.5656),   FML(4.0428),
   FML(2.2306),   FML(4.3108),   FML(3.8725),   FML(2.9624),
    };

    fml correctOutput_k57[] =
    {  FML(4.5212),   FML(3.6573),   FML(3.8635),   FML(4.5572),
   FML(4.3525),   FML(4.9902),   FML(4.7293),   FML(5.6796),
   FML(6.0345),   FML(6.9193),   FML(5.8725),   FML(6.6276),
   FML(6.6483),   FML(6.7181),   FML(6.9565),   FML(6.3473),
   FML(5.0972),   FML(4.9232),   FML(5.1576),   FML(4.5399),
   FML(3.6231),   FML(4.1505),   FML(3.9872),   FML(4.5938),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test45(const tTest& t)
{
    fml input[] =
    {  FML(0.9909976),   FML(0.4855031),   FML(0.5158173),   FML(0.4691882),   FML(0.1588093),
   FML(0.0909718),   FML(0.9613793),   FML(0.3934839),   FML(0.6976583),   FML(0.1980127),
   FML(0.5905871),   FML(0.6251398),   FML(0.0458525),   FML(0.6968696),   FML(0.2811529),
   FML(0.9814237),   FML(0.0062722),   FML(0.0456875),   FML(0.5994561),   FML(0.8310949),
   FML(0.7056721),   FML(0.3305334),   FML(0.3351699),   FML(0.7513894),   FML(0.0280001),
   FML(0.2782521),   FML(0.1849388),   FML(0.3630352),   FML(0.8348799),   FML(0.4009003),
    };
    u32 inputRows = 6;
    u32 inputCols = 5;

    fml correctOutput_k33[] =
    {  FML(1.76553),   FML(1.86756),   FML(2.32881),   FML(1.64349),   FML(1.13127),
   FML(1.79782),   FML(2.86562),   FML(2.41729),   FML(2.04270),   FML(1.55417),
   FML(1.89752),   FML(2.10681),   FML(1.84562),   FML(2.33414),   FML(2.07133),
   FML(2.10665),   FML(1.96234),   FML(2.04399),   FML(1.98558),   FML(2.01709),
   FML(1.39806),   FML(1.79892),   FML(1.91203),   FML(2.55415),   FML(1.90098),
   FML(0.78459),   FML(1.06860),   FML(1.23263),   FML(1.35091),   FML(1.05183),
    };

    fml correctOutput_k55[] =
    {  FML(3.4608),   FML(4.6481),   FML(4.1128),   FML(3.5344),   FML(2.4000),
   FML(3.7705),   FML(4.9916),   FML(4.4994),   FML(3.7989),   FML(3.1622),
   FML(5.1819),   FML(6.2611),   FML(7.0448),   FML(5.7182),   FML(4.0551),
   FML(4.1160),   FML(5.6293),   FML(7.1549),   FML(5.8028),   FML(4.2322),
   FML(3.0330),   FML(5.2333),   FML(5.9158),   FML(4.3112),   FML(2.9485),
   FML(2.1535),   FML(3.7658),   FML(4.6028),   FML(2.7666),   FML(2.7572),
    };

    fml correctOutput_k57[] =
    {  FML(3.9522),   FML(4.1294),   FML(4.2662),   FML(4.9670),   FML(4.0484),
   FML(3.9919),   FML(5.9923),   FML(4.7881),   FML(6.1285),   FML(4.6093),
   FML(6.0615),   FML(7.5743),   FML(7.1822),   FML(6.1233),   FML(5.5253),
   FML(5.6731),   FML(6.5500),   FML(6.5299),   FML(6.3762),   FML(5.5290),
   FML(4.6518),   FML(5.9261),   FML(4.6323),   FML(5.1806),   FML(4.1317),
   FML(3.2896),   FML(4.0569),   FML(3.3103),   FML(4.3191),   FML(4.0687),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test46(const tTest& t)
{
    fml input[] =
    {  FML(0.682243),   FML(0.670315),   FML(0.521830),   FML(0.475541),   FML(0.846684),   FML(0.450586),
   FML(0.865259),   FML(0.442760),   FML(0.564415),   FML(0.462996),   FML(0.670062),   FML(0.522524),
   FML(0.806532),   FML(0.372429),   FML(0.884768),   FML(0.334396),   FML(0.099789),   FML(0.639624),
   FML(0.122529),   FML(0.860213),   FML(0.760660),   FML(0.159789),   FML(0.524314),   FML(0.475026),
   FML(0.343215),   FML(0.257472),   FML(0.289738),   FML(0.217647),   FML(0.496525),   FML(0.972593),
   FML(0.243732),   FML(0.225299),   FML(0.464495),   FML(0.855626),   FML(0.670193),   FML(0.212605),
    };
    u32 inputRows = 6;
    u32 inputCols = 6;

    fml correctOutput_k33[] =
    {  FML(1.77268),   FML(2.34191),   FML(1.88945),   FML(2.06872),   FML(2.22801),   FML(1.61638),
   FML(2.24324),   FML(3.11146),   FML(2.67492),   FML(2.57215),   FML(2.34882),   FML(1.83735),
   FML(1.98976),   FML(2.85729),   FML(2.95117),   FML(2.44784),   FML(1.95648),   FML(1.99261),
   FML(1.26384),   FML(2.57578),   FML(2.09834),   FML(1.90492),   FML(2.40932),   FML(2.00345),
   FML(1.23726),   FML(1.82438),   FML(2.38312),   FML(2.66538),   FML(2.55943),   FML(2.13264),
   FML(0.63434),   FML(0.87383),   FML(1.04321),   FML(1.44583),   FML(1.53035),   FML(1.01649),
    };

    fml correctOutput_k55[] =
    {  FML(4.2398),   FML(4.4177),   FML(5.7074),   FML(5.1707),   FML(3.7686),   FML(2.9356),
   FML(4.2391),   FML(4.9351),   FML(6.4685),   FML(5.6367),   FML(4.2631),   FML(3.3494),
   FML(5.6508),   FML(6.6258),   FML(7.9368),   FML(7.6032),   FML(6.0639),   FML(5.1669),
   FML(4.7968),   FML(5.9237),   FML(7.0367),   FML(7.7212),   FML(7.0712),   FML(4.8540),
   FML(4.2694),   FML(4.9537),   FML(5.9499),   FML(6.9847),   FML(5.1893),   FML(3.3638),
   FML(2.8152),   FML(3.5408),   FML(4.6406),   FML(4.8559),   FML(3.6412),   FML(2.4047),
    };

    fml correctOutput_k57[] =
    {  FML(3.8962),   FML(5.1962),   FML(5.9617),   FML(6.0883),   FML(5.0691),   FML(4.5491),
   FML(5.0407),   FML(5.8247),   FML(7.0175),   FML(7.6055),   FML(6.4078),   FML(5.6308),
   FML(5.5446),   FML(7.2979),   FML(8.5467),   FML(8.9656),   FML(7.2708),   FML(6.5385),
   FML(5.0795),   FML(7.2930),   FML(8.0984),   FML(8.6110),   FML(7.0886),   FML(6.4420),
   FML(4.8166),   FML(6.0481),   FML(6.1818),   FML(5.5810),   FML(4.9341),   FML(4.9095),
   FML(3.0554),   FML(4.1318),   FML(4.8077),   FML(4.9409),   FML(4.3180),   FML(4.0664),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test47(const tTest& t)
{
    fml input[] =
    {  FML(0.905583),   FML(0.822392),   FML(0.996569),   FML(0.048491),   FML(0.448022),   FML(0.350277),   FML(0.448746),
   FML(0.389162),   FML(0.262717),   FML(0.526197),   FML(0.692746),   FML(0.343583),   FML(0.823290),   FML(0.233678),
   FML(0.728669),   FML(0.341599),   FML(0.048829),   FML(0.059609),   FML(0.671381),   FML(0.430677),   FML(0.476285),
   FML(0.499816),   FML(0.035547),   FML(0.163406),   FML(0.173261),   FML(0.384791),   FML(0.472989),   FML(0.334188),
   FML(0.281602),   FML(0.900084),   FML(0.333120),   FML(0.790379),   FML(0.394382),   FML(0.875341),   FML(0.203495),
   FML(0.603674),   FML(0.039732),   FML(0.680315),   FML(0.140310),   FML(0.150587),   FML(0.601497),   FML(0.131536),
    };
    u32 inputRows = 6;
    u32 inputCols = 7;

    fml correctOutput_k33[] =
    {  FML(1.48412),   FML(2.01026),   FML(2.17093),   FML(1.66997),   FML(2.02070),   FML(1.67087),   FML(1.45474),
   FML(1.91349),   FML(2.58186),   FML(1.86419),   FML(2.15737),   FML(1.85428),   FML(2.64657),   FML(1.53738),
   FML(1.41311),   FML(1.64591),   FML(1.16304),   FML(1.53718),   FML(2.35961),   FML(2.13083),   FML(1.78067),
   FML(1.77501),   FML(1.98829),   FML(2.14760),   FML(1.88912),   FML(2.56861),   FML(2.47204),   FML(1.77877),
   FML(1.19783),   FML(2.33268),   FML(1.49754),   FML(1.98164),   FML(1.82190),   FML(2.08700),   FML(1.44752),
   FML(1.11321),   FML(0.99522),   FML(1.64907),   FML(1.01389),   FML(1.34936),   FML(1.24283),   FML(0.93653),
    };

    fml correctOutput_k55[] =
    {  FML(3.9638),   FML(3.9881),   FML(4.4168),   FML(3.8941),   FML(4.3311),   FML(3.4467),   FML(2.7786),
   FML(3.4685),   FML(3.7226),   FML(4.1438),   FML(4.6399),   FML(4.5060),   FML(3.8099),   FML(3.0672),
   FML(4.7013),   FML(5.6308),   FML(7.1653),   FML(6.9197),   FML(6.8006),   FML(5.2015),   FML(4.3991),
   FML(4.1075),   FML(5.2066),   FML(6.4800),   FML(7.4366),   FML(6.3012),   FML(5.6713),   FML(4.1263),
   FML(3.4746),   FML(4.3160),   FML(5.0550),   FML(5.1275),   FML(4.7414),   FML(4.1308),   FML(3.1383),
   FML(2.5874),   FML(2.3830),   FML(3.3857),   FML(3.3700),   FML(3.5760),   FML(2.7380),   FML(2.2652),
    };

    fml correctOutput_k57[] =
    {  FML(3.6308),   FML(4.4854),   FML(5.3271),   FML(6.3938),   FML(4.7292),   FML(4.3406),   FML(3.3801),
   FML(3.3275),   FML(5.1102),   FML(5.8391),   FML(7.0431),   FML(4.9495),   FML(5.2916),   FML(4.2738),
   FML(5.7815),   FML(7.3105),   FML(8.2431),   FML(8.3820),   FML(7.3485),   FML(5.8643),   FML(6.1235),
   FML(4.2394),   FML(6.3145),   FML(6.9136),   FML(8.3868),   FML(6.3437),   FML(6.1338),   FML(4.9182),
   FML(3.7491),   FML(4.4554),   FML(4.8768),   FML(5.9100),   FML(5.2122),   FML(5.1151),   FML(4.2441),
   FML(3.0322),   FML(3.3527),   FML(4.3748),   FML(4.3403),   FML(4.5503),   FML(4.0213),   FML(3.5570),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test48(const tTest& t)
{
    fml input[] =
     {  FML(0.962578),   FML(0.694177),   FML(0.841356),   FML(0.759386),   FML(0.613431),   FML(0.693011),   FML(0.084610),  FML(0.378836),
    FML(0.832925),   FML(0.438822),   FML(0.564736),   FML(0.312374),   FML(0.352855),   FML(0.517919),   FML(0.796813),  FML(0.666314),
    FML(0.958399),   FML(0.041478),   FML(0.743234),   FML(0.437948),   FML(0.443018),   FML(0.625765),   FML(0.455626),  FML(0.025708),
    FML(0.791766),   FML(0.663886),   FML(0.223559),   FML(0.880083),   FML(0.168686),   FML(0.825200),   FML(0.293664),  FML(0.553983),
    FML(0.476072),   FML(0.414586),   FML(0.769966),   FML(0.429031),   FML(0.961796),   FML(0.183854),   FML(0.791109),  FML(0.250838),
    FML(0.190547),   FML(0.233330),   FML(0.506003),   FML(0.616473),   FML(0.372900),   FML(0.505964),   FML(0.151670),  FML(0.625091),
     };
    u32 inputRows = 6;
    u32 inputCols = 8;

    fml correctOutput_k33[] =
    {  FML(1.95404),   FML(2.42625),   FML(2.06339),   FML(1.95059),   FML(1.78119),   FML(2.06684),   FML(1.93572),   FML(1.65817),
   FML(2.18479),   FML(3.17826),   FML(2.56272),   FML(2.76683),   FML(2.61101),   FML(2.49363),   FML(2.41741),   FML(1.37472),
   FML(2.38582),   FML(2.63191),   FML(2.70257),   FML(2.21426),   FML(2.63479),   FML(2.43545),   FML(2.72366),   FML(1.57551),
   FML(1.83208),   FML(2.94271),   FML(2.25449),   FML(3.26536),   FML(2.42616),   FML(2.99253),   FML(2.03844),   FML(1.76257),
   FML(1.43357),   FML(2.11625),   FML(2.70021),   FML(2.38377),   FML(3.02377),   FML(1.86432),   FML(2.59637),   FML(1.42017),
   FML(0.69921),   FML(1.19297),   FML(1.34187),   FML(1.74046),   FML(1.24585),   FML(1.56727),   FML(0.93477),   FML(1.19390),
    };

    fml correctOutput_k55[] =
    {  FML(4.7643),   FML(4.6370),   FML(5.3529),   FML(4.9727),   FML(4.7277),   FML(4.7477),   FML(3.9041),   FML(2.8632),
   FML(4.5426),   FML(5.3819),   FML(5.9669),   FML(5.8089),   FML(5.9242),   FML(5.4993),   FML(4.2599),   FML(3.1862),
   FML(6.3175),   FML(7.8111),   FML(9.3352),   FML(8.8263),   FML(8.5634),   FML(7.5957),   FML(5.7340),   FML(4.3318),
   FML(4.9281),   FML(6.7964),   FML(7.7431),   FML(7.8589),   FML(8.0619),   FML(7.7553),   FML(5.8024),   FML(4.9859),
   FML(4.3666),   FML(5.3674),   FML(7.0985),   FML(6.1404),   FML(6.9542),   FML(5.6346),   FML(4.6450),   FML(3.2343),
   FML(2.9120),   FML(4.5078),   FML(4.6789),   FML(5.1512),   FML(4.3236),   FML(5.1050),   FML(3.3968),   FML(2.8416),
    };

    fml correctOutput_k57[] =
     {  FML(3.9648),    FML(5.0325),    FML(6.0515),    FML(7.7366),    FML(6.5672),    FML(6.0464),    FML(4.4633),    FML(4.1181),
    FML(5.7011),    FML(6.4848),    FML(7.9513),    FML(8.9523),    FML(7.9859),    FML(6.2811),    FML(6.4558),    FML(4.6657),
    FML(6.7065),    FML(8.9700),    FML(9.1817),   FML(11.6402),    FML(9.2772),    FML(9.3304),    FML(6.8765),    FML(6.0032),
    FML(5.9849),    FML(7.9506),    FML(7.7694),    FML(9.7589),    FML(8.9863),    FML(8.2309),    FML(7.9083),    FML(5.9298),
    FML(4.9329),    FML(6.2765),    FML(6.3487),    FML(7.3338),    FML(7.8695),    FML(6.3574),    FML(5.8537),    FML(4.0759),
    FML(3.7237),    FML(4.8888),    FML(4.9497),    FML(5.6748),    FML(5.7146),    FML(5.6475),    FML(4.4653),    FML(4.1674),
     };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test49(const tTest& t)
{
    fml input[] =
    {  FML(0.67928),
   FML(0.29224),
   FML(0.92737),
   FML(0.39887),
   FML(0.62352),
   FML(0.35220),
   FML(0.74668),
    };
    u32 inputRows = 7;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(0.92490),
   FML(1.29973),
   FML(1.26550),
   FML(1.22412),
   FML(1.04025),
   FML(1.19521),
   FML(0.85850),
    };

    fml correctOutput_k55[] =
    {  FML(1.9367),
   FML(1.8369),
   FML(2.4785),
   FML(1.8176),
   FML(2.5785),
   FML(1.6716),
   FML(1.3819),
    };

    fml correctOutput_k57[] =
    {  FML(1.7499),
   FML(1.6564),
   FML(2.4988),
   FML(1.9391),
   FML(2.5829),
   FML(1.8052),
   FML(1.9554),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test50(const tTest& t)
{
    fml input[] =
    {  FML(0.402497),   FML(0.309333),
   FML(0.925250),   FML(0.189865),
   FML(0.342442),   FML(0.692782),
   FML(0.667311),   FML(0.301023),
   FML(0.063205),   FML(0.729344),
   FML(0.212050),   FML(0.244546),
   FML(0.404365),   FML(0.792805),
    };
    u32 inputRows = 7;
    u32 inputCols = 2;

    fml correctOutput_k33[] =
    {  FML(1.37385),   FML(1.41430),
   FML(1.90329),   FML(1.59552),
   FML(1.62204),   FML(2.10526),
   FML(1.69705),   FML(1.50569),
   FML(1.02491),   FML(1.55920),
   FML(1.58829),   FML(1.58452),
   FML(0.80404),   FML(1.03897),
    };

    fml correctOutput_k55[] =
    {  FML(2.1828),   FML(2.2326),
   FML(2.7406),   FML(2.5124),
   FML(2.6357),   FML(3.2345),
   FML(3.1265),   FML(3.0507),
   FML(2.6210),   FML(3.3998),
   FML(2.8871),   FML(2.4966),
   FML(1.8265),   FML(1.7292),
    };

    fml correctOutput_k57[] =
    {  FML(2.0716),   FML(2.2246),
   FML(2.3647),   FML(2.8354),
   FML(2.9106),   FML(3.6744),
   FML(3.0454),   FML(3.4064),
   FML(3.0322),   FML(3.7288),
   FML(2.3860),   FML(2.6581),
   FML(2.0966),   FML(2.4954),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test51(const tTest& t)
{
    fml input[] =
    {  FML(0.866449),   FML(0.096947),   FML(0.194797),
   FML(0.779006),   FML(0.503099),   FML(0.432537),
   FML(0.644764),   FML(0.958162),   FML(0.820436),
   FML(0.541274),   FML(0.422137),   FML(0.525190),
   FML(0.099560),   FML(0.505937),   FML(0.460724),
   FML(0.258924),   FML(0.758621),   FML(0.527186),
   FML(0.139247),   FML(0.663341),   FML(0.197476),
    };
    u32 inputRows = 7;
    u32 inputCols = 3;

    fml correctOutput_k33[] =
    {  FML(1.79007),   FML(1.77752),   FML(1.10991),
   FML(2.30127),   FML(3.17286),   FML(2.11634),
   FML(1.96798),   FML(3.02470),   FML(2.09184),
   FML(1.68875),   FML(2.43220),   FML(2.17296),
   FML(1.42669),   FML(2.45109),   FML(2.02492),
   FML(1.34040),   FML(2.02491),   FML(1.81825),
   FML(0.81511),   FML(1.32463),   FML(1.02154),
    };

    fml correctOutput_k55[] =
    {  FML(3.4511),   FML(3.3921),   FML(3.7228),
   FML(4.2998),   FML(4.2757),   FML(4.1656),
   FML(4.9205),   FML(4.8317),   FML(4.7553),
   FML(4.8708),   FML(5.0883),   FML(5.1132),
   FML(5.0546),   FML(5.5201),   FML(5.1670),
   FML(3.9635),   FML(3.6367),   FML(3.1365),
   FML(2.5070),   FML(2.2035),   FML(2.1167),
    };

    fml correctOutput_k57[] =
    {  FML(3.2563),   FML(3.3341),   FML(3.7121),
   FML(3.5426),   FML(4.0579),   FML(3.8494),
   FML(4.3652),   FML(5.0344),   FML(4.4195),
   FML(4.9144),   FML(5.3618),   FML(5.0408),
   FML(4.5363),   FML(5.1820),   FML(4.9450),
   FML(3.0241),   FML(3.6178),   FML(3.5852),
   FML(2.4614),   FML(2.9889),   FML(3.0613),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test52(const tTest& t)
{
    fml input[] =
    {  FML(0.984978),   FML(0.959540),   FML(0.239710),   FML(0.716977),
   FML(0.179323),   FML(0.425984),   FML(0.875313),   FML(0.615380),
   FML(0.144223),   FML(0.339841),   FML(0.445151),   FML(0.651229),
   FML(0.379262),   FML(0.658345),   FML(0.810511),   FML(0.593092),
   FML(0.378128),   FML(0.556256),   FML(0.045362),   FML(0.759422),
   FML(0.447567),   FML(0.283170),   FML(0.841027),   FML(0.297954),
   FML(0.648949),   FML(0.406948),   FML(0.940844),   FML(0.806895),
    };
    u32 inputRows = 7;
    u32 inputCols = 4;

    fml correctOutput_k33[] =
    {  FML(1.52650),   FML(2.18718),   FML(2.09095),   FML(1.95603),
   FML(1.43746),   FML(2.28737),   FML(2.97603),   FML(1.97860),
   FML(1.34864),   FML(2.51134),   FML(2.99542),   FML(2.51672),
   FML(1.45439),   FML(2.00218),   FML(2.61930),   FML(1.79129),
   FML(1.49171),   FML(2.58807),   FML(2.35536),   FML(2.30162),
   FML(1.68920),   FML(2.51733),   FML(3.14983),   FML(2.17654),
   FML(0.99933),   FML(1.43954),   FML(1.61120),   FML(1.48772),
    };

    fml correctOutput_k55[] =
    {  FML(3.1736),   FML(4.2538),   FML(3.9768),   FML(3.6146),
   FML(3.4628),   FML(4.7471),   FML(5.2574),   FML(4.1330),
   FML(5.1860),   FML(7.7543),   FML(6.6389),   FML(5.4346),
   FML(4.8484),   FML(6.1496),   FML(6.1351),   FML(5.3397),
   FML(4.4978),   FML(6.1850),   FML(6.0581),   FML(5.4511),
   FML(5.1732),   FML(5.9725),   FML(6.2945),   FML(4.4640),
   FML(3.0099),   FML(4.3772),   FML(3.7299),   FML(2.8633),
    };

    fml correctOutput_k57[] =
    {  FML(4.1299),   FML(3.6827),   FML(3.7407),   FML(4.6196),
   FML(4.7961),   FML(5.0998),   FML(4.9888),   FML(5.4201),
   FML(6.5184),   FML(6.2696),   FML(6.1774),   FML(5.8876),
   FML(4.8969),   FML(6.1355),   FML(6.0030),   FML(6.4760),
   FML(5.6345),   FML(5.8975),   FML(6.1555),   FML(6.8321),
   FML(5.3306),   FML(5.3262),   FML(5.4576),   FML(4.9778),
   FML(3.5752),   FML(3.9703),   FML(4.0732),   FML(4.5164),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test53(const tTest& t)
{
    fml input[] =
    {  FML(0.022207),   FML(0.932061),   FML(0.924695),   FML(0.419248),   FML(0.103611),
   FML(0.929916),   FML(0.611439),   FML(0.053869),   FML(0.109546),   FML(0.281852),
   FML(0.118004),   FML(0.859019),   FML(0.138629),   FML(0.851143),   FML(0.080080),
   FML(0.822655),   FML(0.202020),   FML(0.541777),   FML(0.435231),   FML(0.974103),
   FML(0.349690),   FML(0.524704),   FML(0.626433),   FML(0.472836),   FML(0.539827),
   FML(0.658706),   FML(0.483833),   FML(0.912476),   FML(0.511470),   FML(0.402096),
   FML(0.842602),   FML(0.229995),   FML(0.606694),   FML(0.273909),   FML(0.292927),
    };
    u32 inputRows = 7;
    u32 inputCols = 5;

    fml correctOutput_k33[] =
    {  FML(1.50329),   FML(2.31460),   FML(1.72950),   FML(1.00572),   FML(0.65447),
   FML(2.08090),   FML(2.29522),   FML(2.72650),   FML(1.80995),   FML(1.46339),
   FML(1.70599),   FML(2.79825),   FML(1.85760),   FML(2.50685),   FML(1.62137),
   FML(1.86437),   FML(2.06264),   FML(2.80230),   FML(2.37771),   FML(2.26601),
   FML(1.69843),   FML(3.03313),   FML(2.70780),   FML(2.98600),   FML(1.91262),
   FML(1.87210),   FML(2.71880),   FML(2.52630),   FML(2.40562),   FML(1.43877),
   FML(1.25268),   FML(1.48499),   FML(1.46882),   FML(1.36260),   FML(0.86197),
    };

    fml correctOutput_k55[] =
    {  FML(3.3646),   FML(4.1358),   FML(3.5305),   FML(3.2107),   FML(2.1078),
   FML(3.8954),   FML(4.3103),   FML(4.5223),   FML(4.0643),   FML(3.2194),
   FML(4.9198),   FML(6.3682),   FML(7.5888),   FML(6.6098),   FML(4.5391),
   FML(5.3695),   FML(5.9642),   FML(8.2621),   FML(5.6168),   FML(4.2592),
   FML(5.2382),   FML(6.9357),   FML(7.7923),   FML(6.1349),   FML(4.3640),
   FML(5.1511),   FML(5.5574),   FML(6.9322),   FML(4.8423),   FML(4.0571),
   FML(3.3816),   FML(3.9165),   FML(4.6904),   FML(3.6050),   FML(2.9606),
    };

    fml correctOutput_k57[] =
    {  FML(2.9448),   FML(3.4991),   FML(4.4765),   FML(3.9639),   FML(4.3938),
   FML(4.4933),   FML(5.6581),   FML(5.0449),   FML(6.1907),   FML(4.6578),
   FML(5.3556),   FML(7.6746),   FML(7.4027),   FML(6.8244),   FML(5.6266),
   FML(6.2032),   FML(7.2482),   FML(6.8785),   FML(6.3518),   FML(6.3194),
   FML(5.6046),   FML(7.4488),   FML(7.0286),   FML(7.5967),   FML(5.8769),
   FML(4.8492),   FML(5.8719),   FML(5.8708),   FML(6.2876),   FML(5.1785),
   FML(3.8103),   FML(4.8650),   FML(4.7598),   FML(5.3463),   FML(3.6204),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test54(const tTest& t)
{
    fml input[] =
    {  FML(0.026854),   FML(0.241593),   FML(0.767320),   FML(0.240538),   FML(0.876281),   FML(0.015978),
   FML(0.806546),   FML(0.554222),   FML(0.154094),   FML(0.311421),   FML(0.956494),   FML(0.933020),
   FML(0.427030),   FML(0.705312),   FML(0.636588),   FML(0.345568),   FML(0.418966),   FML(0.012072),
   FML(0.084334),   FML(0.583817),   FML(0.324520),   FML(0.258036),   FML(0.919513),   FML(0.481835),
   FML(0.635508),   FML(0.661260),   FML(0.563308),   FML(0.225605),   FML(0.585624),   FML(0.685832),
   FML(0.272652),   FML(0.704277),   FML(0.640202),   FML(0.761466),   FML(0.787940),   FML(0.286266),
   FML(0.146807),   FML(0.754597),   FML(0.713298),   FML(0.287031),   FML(0.407646),   FML(0.879975),
    };
    u32 inputRows = 7;
    u32 inputCols = 6;

    fml correctOutput_k33[] =
    {  FML(1.26011),   FML(1.71295),   FML(1.64759),   FML(1.68987),   FML(2.51466),   FML(1.84471),
   FML(1.81546),   FML(2.49250),   FML(2.21120),   FML(2.55325),   FML(2.10318),   FML(1.88006),
   FML(1.57555),   FML(2.28631),   FML(2.22567),   FML(2.31251),   FML(2.62949),   FML(2.19041),
   FML(1.72080),   FML(2.81435),   FML(2.41354),   FML(2.27294),   FML(2.41837),   FML(1.93017),
   FML(1.75261),   FML(2.42989),   FML(2.87543),   FML(2.84332),   FML(2.80433),   FML(2.30620),
   FML(1.62597),   FML(2.82117),   FML(2.88317),   FML(2.70517),   FML(2.68553),   FML(2.04109),
   FML(0.81729),   FML(1.51741),   FML(1.77558),   FML(1.48968),   FML(1.45688),   FML(1.42683),
    };

    fml correctOutput_k55[] =
    {  FML(3.1632),   FML(3.5779),   FML(5.0737),   FML(4.5554),   FML(4.1061),   FML(2.7178),
   FML(3.2859),   FML(4.1127),   FML(5.2796),   FML(5.3455),   FML(4.1838),   FML(3.4051),
   FML(4.4877),   FML(5.3171),   FML(7.6007),   FML(7.0927),   FML(5.8219),   FML(4.4602),
   FML(4.8644),   FML(6.1054),   FML(8.2874),   FML(8.5134),   FML(6.8514),   FML(5.5112),
   FML(5.2271),   FML(6.3548),   FML(8.4054),   FML(8.3240),   FML(5.7391),   FML(4.5069),
   FML(4.1639),   FML(5.1922),   FML(6.7050),   FML(7.2045),   FML(5.4835),   FML(4.1249),
   FML(3.6914),   FML(4.0069),   FML(4.9811),   FML(5.5615),   FML(3.9485),   FML(3.1339),
    };

    fml correctOutput_k57[] =
    {  FML(2.8335),   FML(4.5416),   FML(5.5453),   FML(5.0551),   FML(4.3931),   FML(4.1215),
   FML(3.8948),   FML(5.2135),   FML(5.6372),   FML(6.1657),   FML(5.5630),   FML(5.4627),
   FML(4.3570),   FML(7.3163),   FML(8.4390),   FML(8.7653),   FML(7.3563),   FML(6.0854),
   FML(5.4563),   FML(7.3938),   FML(8.3028),   FML(8.3242),   FML(8.2406),   FML(7.1359),
   FML(5.7295),   FML(7.6469),   FML(8.7481),   FML(7.9117),   FML(7.5799),   FML(6.4009),
   FML(4.1424),   FML(5.4857),   FML(6.9393),   FML(7.3412),   FML(6.3940),   FML(5.6618),
   FML(3.7749),   FML(5.0706),   FML(5.6079),   FML(5.2805),   FML(5.3878),   FML(4.8224),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test55(const tTest& t)
{
    fml input[] =
    {  FML(0.698709),   FML(0.356723),   FML(0.261162),   FML(0.463785),   FML(0.430770),   FML(0.337598),   FML(0.822334),
   FML(0.049362),   FML(0.160906),   FML(0.293867),   FML(0.942320),   FML(0.358625),   FML(0.469130),   FML(0.753740),
   FML(0.776435),   FML(0.196891),   FML(0.135596),   FML(0.719468),   FML(0.206201),   FML(0.404885),   FML(0.860322),
   FML(0.561615),   FML(0.927682),   FML(0.663794),   FML(0.031703),   FML(0.940493),   FML(0.192364),   FML(0.551834),
   FML(0.048447),   FML(0.920314),   FML(0.576221),   FML(0.031167),   FML(0.211606),   FML(0.850587),   FML(0.539751),
   FML(0.316711),   FML(0.063165),   FML(0.285942),   FML(0.707591),   FML(0.711499),   FML(0.171784),   FML(0.881320),
   FML(0.864113),   FML(0.935253),   FML(0.597006),   FML(0.831571),   FML(0.579231),   FML(0.888816),   FML(0.849834),
    };
    u32 inputRows = 7;
    u32 inputCols = 7;

    fml correctOutput_k33[] =
    {  FML(0.93455),   FML(0.99429),   FML(1.55546),   FML(1.85589),   FML(2.03845),   FML(1.84242),   FML(1.81040),
   FML(1.33153),   FML(1.86513),   FML(1.84389),   FML(2.24334),   FML(2.25037),   FML(2.48474),   FML(2.21913),
   FML(1.96271),   FML(2.31339),   FML(2.30778),   FML(2.60944),   FML(2.20250),   FML(2.64017),   FML(1.93543),
   FML(1.76635),   FML(2.74387),   FML(2.50135),   FML(1.51300),   FML(2.33588),   FML(2.35639),   FML(2.21357),
   FML(1.22379),   FML(2.34171),   FML(2.24775),   FML(2.38137),   FML(2.11166),   FML(3.09291),   FML(1.78851),
   FML(2.14663),   FML(2.71049),   FML(3.03241),   FML(2.82030),   FML(3.04633),   FML(2.91096),   FML(2.86092),
   FML(1.11665),   FML(1.43719),   FML(1.34068),   FML(1.65774),   FML(1.56888),   FML(1.89248),   FML(1.31179),
    };

    fml correctOutput_k55[] =
    {  FML(2.4066),   FML(3.4461),   FML(3.6476),   FML(4.1575),   FML(4.7635),   FML(3.9126),   FML(3.4712),
   FML(2.7720),   FML(4.5858),   FML(5.4686),   FML(4.5845),   FML(5.4666),   FML(4.9271),   FML(3.5727),
   FML(4.3746),   FML(6.0972),   FML(7.4878),   FML(6.3233),   FML(7.1947),   FML(6.2583),   FML(5.1621),
   FML(4.0556),   FML(5.2473),   FML(6.3508),   FML(6.5526),   FML(8.3360),   FML(6.3658),   FML(5.0705),
   FML(4.6146),   FML(6.3790),   FML(7.0576),   FML(7.9014),   FML(8.0797),   FML(6.8937),   FML(5.7860),
   FML(5.3805),   FML(5.9505),   FML(7.6860),   FML(7.4735),   FML(7.5749),   FML(5.6502),   FML(4.6716),
   FML(3.3850),   FML(3.9672),   FML(4.7255),   FML(5.4763),   FML(5.0217),   FML(4.3296),   FML(3.2302),
    };

    fml correctOutput_k57[] =
    {  FML(3.5699),   FML(4.0248),   FML(4.2411),   FML(6.1154),   FML(4.8638),   FML(4.3327),   FML(4.6100),
   FML(4.0869),   FML(5.5626),   FML(5.3484),   FML(8.1176),   FML(6.6587),   FML(5.8840),   FML(5.1416),
   FML(5.1103),   FML(6.2538),   FML(7.5225),   FML(8.6047),   FML(8.4349),   FML(6.5971),   FML(6.5300),
   FML(3.9257),   FML(5.8339),   FML(7.3049),   FML(9.8268),   FML(8.7588),   FML(6.9264),   FML(6.3604),
   FML(5.7190),   FML(7.9395),   FML(8.4503),   FML(9.9256),   FML(9.7057),   FML(8.3704),   FML(6.4831),
   FML(5.2399),   FML(6.2425),   FML(7.3984),   FML(8.2111),   FML(6.8054),   FML(6.0025),   FML(5.3932),
   FML(3.6469),   FML(4.8643),   FML(5.3675),   FML(6.3500),   FML(5.7144),   FML(5.6323),   FML(5.0986),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test56(const tTest& t)
{
    fml input[] =
     {  FML(0.012317),   FML(0.445535),   FML(0.997242),   FML(0.063823),   FML(0.968081),   FML(0.999008),   FML(0.457461),  FML(0.207471),
    FML(0.153663),   FML(0.513323),   FML(0.233636),   FML(0.918719),   FML(0.538864),   FML(0.713989),   FML(0.355394),  FML(0.111659),
    FML(0.878237),   FML(0.584607),   FML(0.863915),   FML(0.209955),   FML(0.995664),   FML(0.882498),   FML(0.991883),  FML(0.567924),
    FML(0.447471),   FML(0.763751),   FML(0.107846),   FML(0.486811),   FML(0.249729),   FML(0.865939),   FML(0.491726),  FML(0.347781),
    FML(0.442630),   FML(0.619844),   FML(0.668215),   FML(0.133036),   FML(0.831677),   FML(0.440548),   FML(0.938186),  FML(0.943191),
    FML(0.071281),   FML(0.152114),   FML(0.823139),   FML(0.167356),   FML(0.638708),   FML(0.214527),   FML(0.033100),  FML(0.185556),
    FML(0.137355),   FML(0.337875),   FML(0.399728),   FML(0.941704),   FML(0.214014),   FML(0.544939),   FML(0.710718),  FML(0.550910),
     };
    u32 inputRows = 7;
    u32 inputCols = 8;

    fml correctOutput_k33[] =
    {  FML(0.77610),   FML(1.37935),   FML(2.27044),   FML(1.82817),   FML(2.76415),   FML(2.39287),   FML(1.67264),   FML(0.81612),
   FML(1.65942),   FML(2.96868),   FML(2.43660),   FML(3.54835),   FML(3.11781),   FML(4.05162),   FML(3.22432),   FML(1.90232),
   FML(2.07686),   FML(2.25479),   FML(2.75382),   FML(1.89957),   FML(3.26686),   FML(3.00800),   FML(3.03045),   FML(1.67123),
   FML(1.92104),   FML(3.15244),   FML(2.24115),   FML(2.81101),   FML(2.40838),   FML(3.84478),   FML(3.47967),   FML(2.71626),
   FML(1.24336),   FML(2.10225),   FML(2.31441),   FML(2.11474),   FML(2.36710),   FML(2.11692),   FML(2.21025),   FML(1.57465),
   FML(1.01405),   FML(1.81120),   FML(2.68636),   FML(2.49031),   FML(2.55813),   FML(2.51945),   FML(2.56193),   FML(2.12669),
   FML(0.45599),   FML(0.97185),   FML(1.09962),   FML(1.75593),   FML(0.96695),   FML(1.18507),   FML(1.11181),   FML(0.79658),
    };

    fml correctOutput_k55[] =
    {  FML(3.3537),   FML(4.0235),   FML(5.5502),   FML(5.8714),   FML(6.0046),   FML(5.3184),   FML(4.4753),   FML(3.3392),
   FML(3.8955),   FML(5.3082),   FML(5.6054),   FML(6.9744),   FML(6.6761),   FML(6.2970),   FML(5.6155),   FML(4.0252),
   FML(5.4241),   FML(5.5903),   FML(8.4130),   FML(9.0060),   FML(9.5889),   FML(8.7816),   FML(7.6567),   FML(6.0328),
   FML(4.1923),   FML(5.6953),   FML(7.3539),   FML(8.5670),   FML(8.5673),   FML(8.3548),   FML(6.5787),   FML(4.7861),
   FML(5.1454),   FML(5.5703),   FML(8.1958),   FML(8.4050),   FML(9.1077),   FML(8.0981),   FML(7.2935),   FML(5.6779),
   FML(3.5442),   FML(4.7596),   FML(5.7076),   FML(6.4478),   FML(5.9410),   FML(6.2482),   FML(5.1700),   FML(3.8327),
   FML(2.8245),   FML(3.8137),   FML(4.2980),   FML(4.9154),   FML(5.0830),   FML(5.3890),   FML(4.4287),   FML(3.3156),
    };

    fml correctOutput_k57[] =
     {  FML(3.3550),    FML(5.3111),    FML(5.9659),    FML(7.2563),    FML(6.9045),    FML(7.0973),    FML(5.5396),    FML(5.8092),
    FML(4.3707),    FML(5.4481),    FML(7.8963),    FML(8.9503),    FML(8.6606),    FML(7.5048),    FML(6.7750),    FML(5.7563),
    FML(4.9653),    FML(7.3656),    FML(9.3944),   FML(11.4306),   FML(10.9448),    FML(9.9788),    FML(8.6254),    FML(7.6044),
    FML(4.6416),    FML(7.2574),    FML(8.3444),   FML(10.5573),   FML(10.4191),    FML(9.2809),    FML(6.7496),    FML(6.3057),
    FML(5.6386),    FML(6.9715),    FML(8.4040),    FML(8.8087),   FML(10.4214),    FML(9.1733),    FML(8.7852),    FML(6.8684),
    FML(4.3577),    FML(5.3717),    FML(5.7077),    FML(6.7627),    FML(7.6157),    FML(7.3617),    FML(5.2325),    FML(4.9821),
    FML(3.1481),    FML(4.1016),    FML(4.3350),    FML(5.4928),    FML(4.9621),    FML(6.0993),    FML(5.1571),    FML(4.3705),
     };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test57(const tTest& t)
{
    fml input[] =
    {  FML(0.64921),
   FML(0.16511),
   FML(0.45503),
   FML(0.61139),
   FML(0.47601),
   FML(0.72801),
   FML(0.52943),
   FML(0.49641),
    };
    u32 inputRows = 8;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(0.80983),
   FML(0.85255),
   FML(1.04215),
   FML(1.13796),
   FML(1.26815),
   FML(1.26782),
   FML(1.16993),
   FML(0.72625),
    };

    fml correctOutput_k55[] =
    {  FML(1.3686),
   FML(1.5098),
   FML(2.2423),
   FML(2.0327),
   FML(2.2557),
   FML(2.3148),
   FML(1.6046),
   FML(1.3254),
    };

    fml correctOutput_k57[] =
    {  FML(1.4887),
   FML(1.5060),
   FML(2.1262),
   FML(1.9502),
   FML(2.1535),
   FML(2.3446),
   FML(1.9419),
   FML(1.9306),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test58(const tTest& t)
{
    fml input[] =
    {  FML(0.230140),   FML(0.096101),
   FML(0.034614),   FML(0.294515),
   FML(0.537896),   FML(0.024658),
   FML(0.816617),   FML(0.533590),
   FML(0.410145),   FML(0.301114),
   FML(0.014311),   FML(0.950135),
   FML(0.461045),   FML(0.553143),
   FML(0.871558),   FML(0.807565),
    };
    u32 inputRows = 8;
    u32 inputCols = 2;

    fml correctOutput_k33[] =
    {  FML(0.63389),   FML(0.56673),
   FML(0.80727),   FML(1.05457),
   FML(1.71959),   FML(1.49093),
   FML(1.57476),   FML(1.59161),
   FML(1.69773),   FML(1.79975),
   FML(1.35509),   FML(2.00672),
   FML(2.25149),   FML(2.28309),
   FML(1.34362),   FML(1.34314),
    };

    fml correctOutput_k55[] =
    {  FML(1.3700),   FML(1.2620),
   FML(1.9426),   FML(2.0984),
   FML(2.7583),   FML(2.3838),
   FML(2.1277),   FML(2.7663),
   FML(2.8153),   FML(3.3098),
   FML(3.9047),   FML(4.5418),
   FML(3.5617),   FML(2.7166),
   FML(2.4012),   FML(2.0786),
    };

    fml correctOutput_k57[] =
    {  FML(1.2540),   FML(1.5203),
   FML(1.8980),   FML(2.3312),
   FML(2.2790),   FML(2.4671),
   FML(2.7198),   FML(3.0197),
   FML(2.7637),   FML(3.4902),
   FML(3.5563),   FML(4.5082),
   FML(2.6313),   FML(2.9651),
   FML(2.7776),   FML(3.1888),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test59(const tTest& t)
{
    fml input[] =
    {  FML(0.617363),   FML(0.331286),   FML(0.407785),
   FML(0.561177),   FML(0.952322),   FML(0.674724),
   FML(0.368773),   FML(0.062437),   FML(0.415479),
   FML(0.239609),   FML(0.982642),   FML(0.991037),
   FML(0.476606),   FML(0.744662),   FML(0.911004),
   FML(0.464413),   FML(0.321557),   FML(0.619859),
   FML(0.337717),   FML(0.096690),   FML(0.921597),
   FML(0.429076),   FML(0.846203),   FML(0.248087),
    };
    u32 inputRows = 8;
    u32 inputCols = 3;

    fml correctOutput_k33[] =
    {  FML(1.80632),   FML(2.26435),   FML(1.85299),
   FML(1.40890),   FML(2.35192),   FML(1.50448),
   FML(1.91716),   FML(2.86008),   FML(2.78016),
   FML(1.56106),   FML(3.11114),   FML(2.52777),
   FML(1.72918),   FML(2.86075),   FML(2.52716),
   FML(1.36566),   FML(2.47407),   FML(2.13260),
   FML(1.65493),   FML(2.24513),   FML(2.13759),
   FML(0.80513),   FML(1.53054),   FML(0.83824),
    };

    fml correctOutput_k55[] =
    {  FML(3.5900),   FML(3.0637),   FML(2.9713),
   FML(3.3852),   FML(3.6217),   FML(4.0601),
   FML(5.3614),   FML(5.4647),   FML(5.7615),
   FML(6.3187),   FML(6.1620),   FML(5.9404),
   FML(5.1971),   FML(4.2426),   FML(4.5539),
   FML(5.5364),   FML(5.6541),   FML(5.6823),
   FML(4.9671),   FML(4.5623),   FML(4.2323),
   FML(2.8872),   FML(2.9595),   FML(2.3573),
    };

    fml correctOutput_k57[] =
    {  FML(2.5062),   FML(2.5905),   FML(2.9615),
   FML(3.5729),   FML(4.1763),   FML(4.1652),
   FML(5.0357),   FML(5.2013),   FML(5.1532),
   FML(4.8465),   FML(5.6703),   FML(5.4696),
   FML(4.1707),   FML(4.7317),   FML(4.8924),
   FML(5.4352),   FML(5.4619),   FML(5.9013),
   FML(3.6181),   FML(4.3656),   FML(4.3616),
   FML(2.7984),   FML(3.6662),   FML(2.8991),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test60(const tTest& t)
{
    fml input[] =
    {  FML(0.7557131),   FML(0.1079628),   FML(0.6200200),   FML(0.3970555),
   FML(0.0089226),   FML(0.8506779),   FML(0.9574002),   FML(0.2402831),
   FML(0.5228424),   FML(0.4292025),   FML(0.4904183),   FML(0.1774224),
   FML(0.4758912),   FML(0.4522684),   FML(0.8959212),   FML(0.6389640),
   FML(0.1062923),   FML(0.3466603),   FML(0.7919037),   FML(0.9794351),
   FML(0.1070061),   FML(0.5923312),   FML(0.4748874),   FML(0.5180600),
   FML(0.2227547),   FML(0.9544729),   FML(0.7887086),   FML(0.8663598),
   FML(0.0542460),   FML(0.9216077),   FML(0.8429705),   FML(0.8078373),
    };
    u32 inputRows = 8;
    u32 inputCols = 4;

    fml correctOutput_k33[] =
    {  FML(1.39273),   FML(1.82370),   FML(2.33197),   FML(1.57531),
   FML(1.30931),   FML(2.79087),   FML(2.32349),   FML(1.52526),
   FML(1.69161),   FML(2.69275),   FML(2.98430),   FML(2.22324),
   FML(1.28574),   FML(2.27840),   FML(3.05102),   FML(2.51323),
   FML(1.16998),   FML(2.23232),   FML(2.97917),   FML(2.49159),
   FML(1.40537),   FML(2.66905),   FML(3.54187),   FML(2.66356),
   FML(1.50509),   FML(2.81487),   FML(3.70866),   FML(2.66606),
   FML(0.86731),   FML(1.75086),   FML(2.20344),   FML(1.60451),
    };

    fml correctOutput_k55[] =
    {  FML(3.5134),   FML(3.9792),   FML(3.8580),   FML(2.6902),
   FML(3.8706),   FML(4.5561),   FML(4.3785),   FML(3.5118),
   FML(5.2326),   FML(6.0456),   FML(6.0197),   FML(5.5272),
   FML(4.9156),   FML(6.5777),   FML(6.6507),   FML(5.9596),
   FML(4.4728),   FML(6.8816),   FML(6.4690),   FML(5.6318),
   FML(5.1171),   FML(7.7942),   FML(7.5068),   FML(6.7674),
   FML(4.7946),   FML(6.8206),   FML(5.9967),   FML(5.5265),
   FML(3.4365),   FML(4.3673),   FML(3.9827),   FML(3.6127),
    };

    fml correctOutput_k57[] =
    {  FML(3.4673),   FML(2.9773),   FML(3.4130),   FML(4.2605),
   FML(3.8591),   FML(4.6643),   FML(4.5060),   FML(5.4573),
   FML(5.9815),   FML(5.8093),   FML(5.9626),   FML(6.0994),
   FML(5.9354),   FML(6.0895),   FML(6.0900),   FML(6.2769),
   FML(5.9380),   FML(6.1524),   FML(6.0556),   FML(6.5691),
   FML(6.4705),   FML(6.7788),   FML(6.8710),   FML(7.1038),
   FML(4.8486),   FML(5.1635),   FML(5.6076),   FML(5.9815),
   FML(3.6923),   FML(4.2581),   FML(4.9407),   FML(4.4296),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test61(const tTest& t)
{
    fml input[] =
    {  FML(0.502919),   FML(0.875277),   FML(0.219675),   FML(0.083234),   FML(0.273300),
   FML(0.627441),   FML(0.642280),   FML(0.741832),   FML(0.115944),   FML(0.885631),
   FML(0.605854),   FML(0.890927),   FML(0.447958),   FML(0.276242),   FML(0.900586),
   FML(0.565256),   FML(0.594083),   FML(0.477097),   FML(0.822302),   FML(0.630880),
   FML(0.480339),   FML(0.953519),   FML(0.904744),   FML(0.552411),   FML(0.418375),
   FML(0.558017),   FML(0.965713),   FML(0.054988),   FML(0.062948),   FML(0.032917),
   FML(0.089627),   FML(0.817221),   FML(0.714297),   FML(0.324187),   FML(0.478492),
   FML(0.107824),   FML(0.129820),   FML(0.959269),   FML(0.571092),   FML(0.832722),
    };
    u32 inputRows = 8;
    u32 inputCols = 5;

    fml correctOutput_k33[] =
    {  FML(1.64341),   FML(2.48628),   FML(1.68645),   FML(1.68701),   FML(1.17421),
   FML(2.35780),   FML(2.98888),   FML(2.71176),   FML(2.03202),   FML(1.89098),
   FML(2.08263),   FML(3.10414),   FML(2.75701),   FML(2.88500),   FML(2.37373),
   FML(2.29213),   FML(3.31886),   FML(3.36490),   FML(3.14164),   FML(1.97179),
   FML(2.22211),   FML(3.01801),   FML(2.79963),   FML(1.69294),   FML(1.30538),
   FML(1.95685),   FML(3.09504),   FML(2.84938),   FML(2.26677),   FML(1.28839),
   FML(1.14519),   FML(2.40446),   FML(2.69670),   FML(2.51129),   FML(1.74460),
   FML(0.67961),   FML(1.06135),   FML(1.80735),   FML(1.59486),   FML(1.22008),
    };

    fml correctOutput_k55[] =
    {  FML(3.7607),   FML(4.2201),   FML(4.9294),   FML(3.5296),   FML(3.1729),
   FML(4.6351),   FML(4.7190),   FML(6.1835),   FML(5.1053),   FML(4.0299),
   FML(5.5731),   FML(6.9766),   FML(8.7568),   FML(7.2257),   FML(4.4858),
   FML(6.3017),   FML(7.8868),   FML(9.1741),   FML(6.9588),   FML(4.2100),
   FML(5.7495),   FML(6.9088),   FML(8.3409),   FML(6.2362),   FML(4.1417),
   FML(5.1051),   FML(6.2494),   FML(7.6479),   FML(6.8437),   FML(5.3636),
   FML(5.0489),   FML(5.7443),   FML(7.3568),   FML(6.0809),   FML(4.3321),
   FML(3.2760),   FML(3.5344),   FML(4.2891),   FML(3.3186),   FML(2.0478),
    };

    fml correctOutput_k57[] =
    {  FML(3.5313),   FML(4.8827),   FML(5.0486),   FML(5.0927),   FML(4.5803),
   FML(4.3404),   FML(6.1584),   FML(6.5784),   FML(6.1385),   FML(5.8756),
   FML(6.5055),   FML(8.7733),   FML(7.8303),   FML(7.8271),   FML(6.8110),
   FML(6.5270),   FML(7.8644),   FML(7.7392),   FML(7.9937),   FML(6.7549),
   FML(5.5680),   FML(7.5810),   FML(7.7035),   FML(7.9141),   FML(7.1783),
   FML(5.6934),   FML(7.6665),   FML(7.4393),   FML(7.8760),   FML(6.8167),
   FML(5.1053),   FML(6.3567),   FML(6.3954),   FML(5.0740),   FML(5.2427),
   FML(3.5706),   FML(4.0679),   FML(4.2215),   FML(3.2933),   FML(3.4731),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test62(const tTest& t)
{
    fml input[] =
    {  FML(0.707724),   FML(0.174751),   FML(0.195435),   FML(0.288876),   FML(0.338207),   FML(0.568246),
   FML(0.661539),   FML(0.193433),   FML(0.966537),   FML(0.769928),   FML(0.116613),   FML(0.661813),
   FML(0.248232),   FML(0.349191),   FML(0.683282),   FML(0.997213),   FML(0.543044),   FML(0.690634),
   FML(0.299183),   FML(0.316728),   FML(0.377368),   FML(0.846860),   FML(0.134448),   FML(0.654555),
   FML(0.991576),   FML(0.722491),   FML(0.263258),   FML(0.527016),   FML(0.644091),   FML(0.052483),
   FML(0.485371),   FML(0.676203),   FML(0.408190),   FML(0.248191),   FML(0.694966),   FML(0.911218),
   FML(0.455118),   FML(0.678191),   FML(0.097091),   FML(0.678836),   FML(0.653468),   FML(0.522440),
   FML(0.069031),   FML(0.374899),   FML(0.959708),   FML(0.598976),   FML(0.064385),   FML(0.610033),
    };
    u32 inputRows = 8;
    u32 inputCols = 6;

    fml correctOutput_k33[] =
    {  FML(1.38246),   FML(1.86264),   FML(1.84871),   FML(1.96644),   FML(1.79801),   FML(1.26435),
   FML(1.43001),   FML(2.08088),   FML(2.84486),   FML(2.95658),   FML(2.73537),   FML(2.01792),
   FML(1.16468),   FML(2.18420),   FML(2.76843),   FML(2.95684),   FML(2.87206),   FML(1.64335),
   FML(1.93937),   FML(2.61394),   FML(2.64309),   FML(2.84937),   FML(2.48153),   FML(1.77721),
   FML(2.10783),   FML(2.53494),   FML(2.27373),   FML(2.26015),   FML(2.90340),   FML(1.85807),
   FML(2.07316),   FML(2.64600),   FML(2.45944),   FML(2.17404),   FML(2.81583),   FML(2.27808),
   FML(1.39265),   FML(2.42911),   FML(2.52900),   FML(2.69780),   FML(2.56812),   FML(1.83053),
   FML(0.73758),   FML(1.13784),   FML(1.74793),   FML(1.32370),   FML(1.22504),   FML(1.17113),
    };

    fml correctOutput_k55[] =
    {  FML(3.0273),   FML(3.7214),   FML(4.5675),   FML(5.0726),   FML(4.1560),   FML(3.4485),
   FML(3.3446),   FML(4.8690),   FML(5.4097),   FML(6.0971),   FML(4.8287),   FML(3.8889),
   FML(4.4511),   FML(7.1505),   FML(7.1487),   FML(7.0340),   FML(6.2682),   FML(4.6985),
   FML(5.4140),   FML(7.5005),   FML(7.8898),   FML(7.7980),   FML(6.4627),   FML(5.2793),
   FML(4.8209),   FML(6.7995),   FML(7.8603),   FML(8.0444),   FML(7.3084),   FML(5.7619),
   FML(4.1567),   FML(5.7355),   FML(7.5322),   FML(8.6057),   FML(5.9664),   FML(5.0709),
   FML(4.3404),   FML(5.9642),   FML(7.0082),   FML(6.6071),   FML(4.7462),   FML(3.5763),
   FML(3.2440),   FML(3.7118),   FML(4.7497),   FML(5.1532),   FML(3.7720),   FML(3.2610),
    };

    fml correctOutput_k57[] =
    {  FML(4.2624),   FML(4.0297),   FML(5.1455),   FML(5.5083),   FML(4.7501),   FML(4.6190),
   FML(4.7589),   FML(5.2516),   FML(6.3353),   FML(6.7172),   FML(5.6885),   FML(5.6271),
   FML(6.0057),   FML(7.2176),   FML(7.1783),   FML(8.8911),   FML(7.0945),   FML(6.4748),
   FML(5.7179),   FML(7.8314),   FML(8.6712),   FML(8.8613),   FML(7.8646),   FML(6.2294),
   FML(5.4569),   FML(7.2696),   FML(9.3614),   FML(9.5540),   FML(8.3865),   FML(5.5754),
   FML(4.9962),   FML(7.5279),   FML(8.2402),   FML(8.0158),   FML(7.6778),   FML(6.2458),
   FML(5.4244),   FML(6.0634),   FML(6.3613),   FML(6.6424),   FML(5.8188),   FML(4.6892),
   FML(3.3950),   FML(4.3140),   FML(5.2413),   FML(4.9210),   FML(5.3401),   FML(4.7349),
    };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test63(const tTest& t)
{
    fml input[] =
    {  FML(0.822175),   FML(0.756177),   FML(0.295717),   FML(0.755425),   FML(0.692304),   FML(0.807515),   FML(0.267850),
   FML(0.901176),   FML(0.853355),   FML(0.908667),   FML(0.828102),   FML(0.746620),   FML(0.045184),   FML(0.623891),
   FML(0.557507),   FML(0.896021),   FML(0.499019),   FML(0.618004),   FML(0.662782),   FML(0.402813),   FML(0.058187),
   FML(0.323507),   FML(0.463512),   FML(0.999924),   FML(0.431537),   FML(0.756370),   FML(0.808295),   FML(0.377656),
   FML(0.931947),   FML(0.838593),   FML(0.781481),   FML(0.026071),   FML(0.095640),   FML(0.600026),   FML(0.336261),
   FML(0.990978),   FML(0.831174),   FML(0.303157),   FML(0.412490),   FML(0.297923),   FML(0.820357),   FML(0.309572),
   FML(0.451462),   FML(0.882456),   FML(0.560360),   FML(0.062414),   FML(0.230777),   FML(0.779855),   FML(0.579998),
   FML(0.136466),   FML(0.273098),   FML(0.378519),   FML(0.466807),   FML(0.518483),   FML(0.016704),   FML(0.167288),
    };
    u32 inputRows = 8;
    u32 inputCols = 7;

    fml correctOutput_k33[] =
    {  FML(2.20566),   FML(2.96255),   FML(2.63821),   FML(2.82056),   FML(2.24141),   FML(2.05236),   FML(1.03068),
   FML(2.60075),   FML(3.38303),   FML(3.50275),   FML(3.08013),   FML(3.11829),   FML(2.07066),   FML(1.56891),
   FML(1.94273),   FML(3.45814),   FML(3.30559),   FML(3.57275),   FML(3.02462),   FML(2.78707),   FML(1.47145),
   FML(2.33278),   FML(3.47575),   FML(3.25926),   FML(2.26719),   FML(2.17231),   FML(2.23802),   FML(1.59722),
   FML(2.63171),   FML(3.46951),   FML(2.83573),   FML(2.11358),   FML(2.31376),   FML(2.58593),   FML(2.00879),
   FML(2.63985),   FML(3.50766),   FML(2.50967),   FML(1.76004),   FML(1.69601),   FML(2.44462),   FML(2.06320),
   FML(1.60753),   FML(2.49930),   FML(2.31199),   FML(1.81482),   FML(1.95211),   FML(1.99904),   FML(1.42660),
   FML(0.85293),   FML(1.22359),   FML(1.27362),   FML(1.11785),   FML(1.08908),   FML(0.92016),   FML(0.92652),
    };

    fml correctOutput_k55[] =
     {  FML(4.4743),    FML(5.9365),    FML(6.6990),    FML(6.0913),    FML(5.2046),    FML(3.9904),    FML(2.6668),
    FML(4.9324),    FML(5.8410),    FML(7.7321),    FML(6.6968),    FML(6.0296),    FML(4.6633),    FML(3.5949),
    FML(6.4398),    FML(8.5979),   FML(10.3660),    FML(9.4967),    FML(7.6707),    FML(6.5924),    FML(4.7456),
    FML(7.8232),    FML(8.9967),   FML(10.0539),    FML(8.7100),    FML(7.7660),    FML(5.9691),    FML(4.4801),
    FML(6.6806),    FML(7.6551),    FML(8.3885),    FML(8.2500),    FML(6.7257),    FML(5.5356),    FML(4.5098),
    FML(5.7933),    FML(6.3852),    FML(7.3224),    FML(8.3623),    FML(7.8067),    FML(5.5447),    FML(4.3174),
    FML(5.0300),    FML(5.2515),    FML(6.1380),    FML(6.0046),    FML(5.1692),    FML(3.2859),    FML(2.8687),
    FML(3.3614),    FML(4.0358),    FML(4.5941),    FML(4.4884),    FML(3.8194),    FML(2.8245),    FML(2.4516),
     };

    fml correctOutput_k57[] =
     {  FML(5.0345),    FML(5.9518),    FML(6.3721),    FML(7.3412),    FML(6.8339),    FML(5.1065),    FML(4.6550),
    FML(5.3250),    FML(7.5141),    FML(8.2317),    FML(9.3554),    FML(8.4393),    FML(7.1139),    FML(6.0819),
    FML(7.4012),    FML(9.5383),   FML(10.1732),   FML(11.4453),   FML(10.0652),    FML(8.0012),    FML(5.5930),
    FML(6.9829),    FML(8.6412),   FML(10.6771),   FML(11.0098),    FML(9.3297),    FML(7.2184),    FML(5.5329),
    FML(6.2845),    FML(8.1781),   FML(10.0404),   FML(10.8648),    FML(8.4201),    FML(7.1116),    FML(4.8335),
    FML(5.3649),    FML(7.0937),    FML(8.5609),   FML(10.5206),    FML(7.8268),    FML(6.5895),    FML(5.2521),
    FML(5.1089),    FML(6.7533),    FML(6.3555),    FML(6.1714),    FML(5.5953),    FML(4.5179),    FML(4.2440),
    FML(3.8301),    FML(4.6287),    FML(4.6816),    FML(4.7448),    FML(4.8220),    FML(4.1229),    FML(3.6144),
     };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


void test64(const tTest& t)
{
    fml input[] =
     {  FML(0.456856),   FML(0.261959),   FML(0.338672),   FML(0.849813),   FML(0.990856),   FML(0.062682),   FML(0.648889),  FML(0.268108),
    FML(0.865384),   FML(0.333262),   FML(0.726868),   FML(0.519095),   FML(0.629879),   FML(0.062169),   FML(0.518624),  FML(0.457721),
    FML(0.362565),   FML(0.141017),   FML(0.420340),   FML(0.266387),   FML(0.643787),   FML(0.521007),   FML(0.038506),  FML(0.741142),
    FML(0.158871),   FML(0.956585),   FML(0.335191),   FML(0.678630),   FML(0.916138),   FML(0.302149),   FML(0.504510),  FML(0.039489),
    FML(0.978881),   FML(0.546221),   FML(0.812476),   FML(0.463414),   FML(0.833529),   FML(0.882513),   FML(0.277061),  FML(0.588530),
    FML(0.086290),   FML(0.148524),   FML(0.514036),   FML(0.712651),   FML(0.408509),   FML(0.305170),   FML(0.449842),  FML(0.897725),
    FML(0.575527),   FML(0.960105),   FML(0.342854),   FML(0.588042),   FML(0.445364),   FML(0.128668),   FML(0.763461),  FML(0.728943),
    FML(0.145075),   FML(0.657346),   FML(0.944012),   FML(0.227580),   FML(0.896239),   FML(0.730448),   FML(0.833815),  FML(0.473033),
     };
    u32 inputRows = 8;
    u32 inputCols = 8;

    fml correctOutput_k33[] =
    {  FML(1.46302),   FML(2.01441),   FML(1.82364),   FML(2.48504),   FML(2.01703),   FML(1.46105),   FML(1.49901),   FML(1.27948),
   FML(1.50381),   FML(1.87808),   FML(2.08584),   FML(2.64421),   FML(2.59245),   FML(2.21460),   FML(1.99399),   FML(1.63181),
   FML(1.67779),   FML(2.37937),   FML(2.71141),   FML(2.80811),   FML(2.74833),   FML(2.61669),   FML(1.46704),   FML(1.62089),
   FML(1.75654),   FML(3.19431),   FML(2.40467),   FML(3.09082),   FML(3.20014),   FML(2.71107),   FML(2.59161),   FML(1.21436),
   FML(1.62812),   FML(1.93806),   FML(2.86714),   FML(2.80903),   FML(2.90146),   FML(2.70562),   FML(2.19357),   FML(2.00518),
   FML(1.90594),   FML(2.87019),   FML(2.91638),   FML(2.82945),   FML(2.43214),   FML(2.39673),   FML(2.72324),   FML(2.42514),
   FML(1.45301),   FML(2.63242),   FML(2.63259),   FML(3.00729),   FML(2.64330),   FML(2.87244),   FML(3.13145),   FML(2.40213),
   FML(0.99043),   FML(1.59823),   FML(1.90207),   FML(1.21231),   FML(1.52271),   FML(1.61827),   FML(1.59971),   FML(1.30738),
    };

    fml correctOutput_k55[] =
    {  FML(3.2150),   FML(3.8505),   FML(5.2437),   FML(4.0774),   FML(4.6007),   FML(4.3336),   FML(3.3028),   FML(2.3679),
   FML(3.0220),   FML(4.2871),   FML(5.8901),   FML(5.0256),   FML(5.4398),   FML(5.5054),   FML(4.1699),   FML(2.6348),
   FML(4.6692),   FML(6.8275),   FML(8.9450),   FML(8.1893),   FML(8.1980),   FML(8.3224),   FML(5.8231),   FML(3.7388),
   FML(5.4200),   FML(6.8391),   FML(8.5767),   FML(8.3992),   FML(8.2656),   FML(7.1736),   FML(5.8847),   FML(4.3305),
   FML(4.6804),   FML(5.9164),   FML(7.9841),   FML(8.0269),   FML(7.1681),   FML(7.7927),   FML(6.1492),   FML(5.1573),
   FML(4.6765),   FML(6.9683),   FML(9.0235),   FML(8.4818),   FML(8.2972),   FML(8.4085),   FML(7.2322),   FML(4.9505),
   FML(5.2167),   FML(6.4250),   FML(7.7370),   FML(7.6168),   FML(7.9240),   FML(7.8966),   FML(6.5085),   FML(4.1249),
   FML(3.0245),   FML(3.4190),   FML(4.7578),   FML(4.8935),   FML(5.3285),   FML(5.5303),   FML(4.0592),   FML(2.9945),
    };

    fml correctOutput_k57[] =
     {  FML(3.4526),    FML(4.3841),    FML(4.4380),    FML(5.5009),    FML(5.9789),    FML(5.0599),    FML(4.9156),    FML(4.0667),
    FML(4.5100),    FML(5.8645),    FML(6.7331),    FML(7.6563),    FML(7.8371),    FML(7.1557),    FML(5.4371),    FML(5.4775),
    FML(5.4035),    FML(8.5613),    FML(8.5444),   FML(10.6384),    FML(9.8063),    FML(8.8672),    FML(6.7809),    FML(6.5425),
    FML(5.4511),    FML(7.7702),    FML(8.4330),    FML(9.3413),    FML(9.4127),    FML(8.5302),    FML(7.5155),    FML(6.0657),
    FML(5.9945),    FML(7.1770),    FML(8.4320),    FML(9.6019),   FML(10.7525),    FML(9.2378),    FML(7.1150),    FML(6.5929),
    FML(5.5420),    FML(8.1125),    FML(8.7028),   FML(10.8885),   FML(11.1360),   FML(10.0496),    FML(7.3723),    FML(6.6406),
    FML(5.2037),    FML(6.7435),    FML(7.3999),    FML(8.7985),    FML(9.0864),    FML(7.1300),    FML(6.4807),    FML(5.1688),
    FML(2.9351),    FML(4.3019),    FML(5.2800),    FML(5.8172),    FML(6.8794),    FML(5.8254),    FML(5.1769),    FML(5.1773),
     };

    TEST_ALL_1_COMPONENT_1_COUNT_KERNELS
}


fml kKernel33_2[] = {
    FML(0.28086),   FML(0.39329),   FML(0.14117),
    FML(0.31479),   FML(0.49841),   FML(0.31661),
    FML(0.97568),   FML(0.62691),   FML(0.12740),
    FML(0.98908),   FML(0.40378),   FML(0.45498),
    FML(0.14673),   FML(0.91406),   FML(0.60231),
    FML(0.56700),   FML(0.66996),   FML(0.55526),
};
fml kBias33_2[]   = {
    FML(0.869575),   FML(0.086500),
};

fml kKernel55_2[] = {
    FML(0.907000),   FML(0.054459),   FML(0.235051),   FML(0.768103),   FML(0.551615),
    FML(0.275941),   FML(0.271272),   FML(0.933579),   FML(0.377911),   FML(0.200617),
    FML(0.734078),   FML(0.720822),   FML(0.732311),   FML(0.524343),   FML(0.496579),
    FML(0.325613),   FML(0.072304),   FML(0.667797),   FML(0.746921),   FML(0.886914),
    FML(0.411621),   FML(0.046760),   FML(0.563113),   FML(0.202633),   FML(0.070462),
    FML(0.102013),   FML(0.608712),   FML(0.474003),   FML(0.407083),   FML(0.158062),
    FML(0.481358),   FML(0.150423),   FML(0.318247),   FML(0.998995),   FML(0.416624),
    FML(0.117101),   FML(0.081782),   FML(0.316664),   FML(0.849075),   FML(0.738875),
    FML(0.302141),   FML(0.659962),   FML(0.713374),   FML(0.233061),   FML(0.370858),
    FML(0.517848),   FML(0.550278),   FML(0.841520),   FML(0.406841),   FML(0.301979),
};
fml kBias55_2[]   = {
    FML(0.74618),   FML(0.50787),
};

fml kKernel57_2[] = {
    FML(0.5421388),   FML(0.7651711),   FML(0.4560648),   FML(0.6310110),   FML(0.5158787),   FML(0.7113897),  FML(0.9948258),
    FML(0.4561297),   FML(0.4293956),   FML(0.4767157),   FML(0.0808466),   FML(0.1151705),   FML(0.1505206),  FML(0.2449001),
    FML(0.8509617),   FML(0.1828592),   FML(0.6155789),   FML(0.4816474),   FML(0.2063322),   FML(0.8090261),  FML(0.6107243),
    FML(0.1328188),   FML(0.8402662),   FML(0.7689384),   FML(0.5975446),   FML(0.4680271),   FML(0.4832406),  FML(0.9964094),
    FML(0.8551356),   FML(0.1171444),   FML(0.9135925),   FML(0.0010581),   FML(0.9552209),   FML(0.9227060),  FML(0.1588100),
    FML(0.4305374),   FML(0.8160474),   FML(0.5984576),   FML(0.2051604),   FML(0.7323206),   FML(0.5141637),  FML(0.5491517),
    FML(0.8655712),   FML(0.9229134),   FML(0.9137695),   FML(0.8404263),   FML(0.5182482),   FML(0.5821799),  FML(0.1535598),
    FML(0.6630843),   FML(0.3027921),   FML(0.7070472),   FML(0.0565810),   FML(0.5603621),   FML(0.9729775),  FML(0.5201322),
    FML(0.4284560),   FML(0.7536597),   FML(0.2380418),   FML(0.8368284),   FML(0.3380159),   FML(0.1105703),  FML(0.3089580),
    FML(0.2454619),   FML(0.2258277),   FML(0.9099615),   FML(0.0160570),   FML(0.6176303),   FML(0.7451312),  FML(0.5223446),
};
fml kBias57_2[]   = {
    FML(0.77918),   FML(0.62658),
};


#define TEST_ALL_1_COMPONENT_2_COUNT_KERNELS \
    fml* output = new fml[2*inputRows*inputCols]; \
    for (u32 i = 0; i < 2*inputRows*inputCols; i++) \
        output[i] = FML(0.0); \
 \
    { \
        s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel33_2, 3, 3, \
                              1, 1, \
                              2, \
                 kBias33_2, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k33, 2*inputRows*inputCols); \
    } \
 \
    { \
        s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel55_2, 5, 5, \
                              1, 1, \
                              2, \
                 kBias55_2, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k55, 2*inputRows*inputCols); \
    } \
 \
    { \
        s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel57_2, 5, 7, \
                              1, 1, \
                              2, \
                 kBias57_2, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k57, 2*inputRows*inputCols); \
    } \
 \
    delete [] output; \


void test65(const tTest& t)
{
    fml input[] = {  FML(0.81477),  };
    u32 inputRows = 1;
    u32 inputCols = 1;

    fml correctOutput_k33[] = {  FML(1.27566),   FML(0.83125),  };

    fml correctOutput_k55[] = {  FML(1.34285),   FML(0.76588),  };

    fml correctOutput_k57[] = {  FML(1.17161),   FML(0.67268),  };

    TEST_ALL_1_COMPONENT_2_COUNT_KERNELS
}


int main()
{
    tCrashReporter::init();

    srand(time(0));

    tTest("convolve 2d test 0", test0);

    tTest("convolve 2d test 1", test1);
    tTest("convolve 2d test 2", test2);
    tTest("convolve 2d test 3", test3);
    tTest("convolve 2d test 4", test4);
    tTest("convolve 2d test 5", test5);
    tTest("convolve 2d test 6", test6);
    tTest("convolve 2d test 7", test7);
    tTest("convolve 2d test 8", test8);
    tTest("convolve 2d test 9", test9);
    tTest("convolve 2d test 10", test10);
    tTest("convolve 2d test 11", test11);
    tTest("convolve 2d test 12", test12);
    tTest("convolve 2d test 13", test13);
    tTest("convolve 2d test 14", test14);
    tTest("convolve 2d test 15", test15);
    tTest("convolve 2d test 16", test16);
    tTest("convolve 2d test 17", test17);
    tTest("convolve 2d test 18", test18);
    tTest("convolve 2d test 19", test19);
    tTest("convolve 2d test 20", test20);
    tTest("convolve 2d test 21", test21);
    tTest("convolve 2d test 22", test22);
    tTest("convolve 2d test 23", test23);
    tTest("convolve 2d test 24", test24);
    tTest("convolve 2d test 25", test25);
    tTest("convolve 2d test 26", test26);
    tTest("convolve 2d test 27", test27);
    tTest("convolve 2d test 28", test28);
    tTest("convolve 2d test 29", test29);
    tTest("convolve 2d test 30", test30);
    tTest("convolve 2d test 31", test31);
    tTest("convolve 2d test 32", test32);
    tTest("convolve 2d test 33", test33);
    tTest("convolve 2d test 34", test34);
    tTest("convolve 2d test 35", test35);
    tTest("convolve 2d test 36", test36);
    tTest("convolve 2d test 37", test37);
    tTest("convolve 2d test 38", test38);
    tTest("convolve 2d test 39", test39);
    tTest("convolve 2d test 40", test40);
    tTest("convolve 2d test 41", test41);
    tTest("convolve 2d test 42", test42);
    tTest("convolve 2d test 43", test43);
    tTest("convolve 2d test 44", test44);
    tTest("convolve 2d test 45", test45);
    tTest("convolve 2d test 46", test46);
    tTest("convolve 2d test 47", test47);
    tTest("convolve 2d test 48", test48);
    tTest("convolve 2d test 49", test49);
    tTest("convolve 2d test 50", test50);
    tTest("convolve 2d test 51", test51);
    tTest("convolve 2d test 52", test52);
    tTest("convolve 2d test 53", test53);
    tTest("convolve 2d test 54", test54);
    tTest("convolve 2d test 55", test55);
    tTest("convolve 2d test 56", test56);
    tTest("convolve 2d test 57", test57);
    tTest("convolve 2d test 58", test58);
    tTest("convolve 2d test 59", test59);
    tTest("convolve 2d test 60", test60);
    tTest("convolve 2d test 61", test61);
    tTest("convolve 2d test 62", test62);
    tTest("convolve 2d test 63", test63);
    tTest("convolve 2d test 64", test64);

    tTest("convolve 2d test 65", test65);

    return 0;
}
