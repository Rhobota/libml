#include <ml/common.h>
#include <ml/conv2d/cpu_golden.h>

#include <rho/tTest.h>
#include <rho/tCrashReporter.h>

using namespace rho;
using ml::fml;


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
void s_checkOutput(const tTest& t, fml* output, fml* correctOutput, u32 outputRows, u32 outputCols, u32 stepY=1, u32 stepX=1, u32 numOutputComponents=1)
{
    for (u32 r = 0; r < outputRows; r += stepY)
    {
        for (u32 c = 0; c < outputCols; c += stepX)
        {
            for (u32 k = 0; k < numOutputComponents; k++)
            {
                fml out = *output++;
                fml correct = correctOutput[(r*outputCols + c) * numOutputComponents + k];
                if (fabs(out - correct) >= 0.0001)
                {
                    std::cerr << "Fail at element " << (r*outputCols + c) << ": output = " << out << "  correctOutput = " << correct << std::endl;
                    t.fail();
                }
            }
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

    fml correctOutput[] = {  FML(38.0),  FML(33.0),
                             FML(23.0),  FML(18.0),  };
    fml output[] = {  FML(0.0),  FML(0.0),
                      FML(0.0),  FML(0.0),  };

    ml::s_conv2d(input, inputRows, inputCols, inputComponents,
             kernel, kernelRows, kernelCols,
                     1, 1,
                     1,
             &bias, FML(0.5),
             output);

    s_checkOutput(t, output, correctOutput, inputRows, inputCols);
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
    /* Step (1, 1) */ \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel33_1, 3, 3, \
                              1, 1, \
                              1, \
                 &kBias33_1, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k33, inputRows, inputCols); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel55_1, 5, 5, \
                              1, 1, \
                              1, \
                 &kBias55_1, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k55, inputRows, inputCols); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel57_1, 5, 7, \
                              1, 1, \
                              1, \
                 &kBias57_1, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k57, inputRows, inputCols); \
    } \
 \
    /* Step (2, 1) */ \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel33_1, 3, 3, \
                              2, 1, \
                              1, \
                 &kBias33_1, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k33, inputRows, inputCols, 2, 1); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel55_1, 5, 5, \
                              2, 1, \
                              1, \
                 &kBias55_1, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k55, inputRows, inputCols, 2, 1); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel57_1, 5, 7, \
                              2, 1, \
                              1, \
                 &kBias57_1, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k57, inputRows, inputCols, 2, 1); \
    } \
 \
    /* Step (1, 2) */ \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel33_1, 3, 3, \
                              1, 2, \
                              1, \
                 &kBias33_1, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k33, inputRows, inputCols, 1, 2); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel55_1, 5, 5, \
                              1, 2, \
                              1, \
                 &kBias55_1, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k55, inputRows, inputCols, 1, 2); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel57_1, 5, 7, \
                              1, 2, \
                              1, \
                 &kBias57_1, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k57, inputRows, inputCols, 1, 2); \
    } \
 \
    /* Step (2, 2) */ \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel33_1, 3, 3, \
                              2, 2, \
                              1, \
                 &kBias33_1, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k33, inputRows, inputCols, 2, 2); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel55_1, 5, 5, \
                              2, 2, \
                              1, \
                 &kBias55_1, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k55, inputRows, inputCols, 2, 2); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel57_1, 5, 7, \
                              2, 2, \
                              1, \
                 &kBias57_1, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k57, inputRows, inputCols, 2, 2); \
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
    /* Step (1, 1) */ \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel33_2, 3, 3, \
                              1, 1, \
                              2, \
                 kBias33_2, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k33, inputRows, inputCols, 1, 1, 2); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel55_2, 5, 5, \
                              1, 1, \
                              2, \
                 kBias55_2, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k55, inputRows, inputCols, 1, 1, 2); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel57_2, 5, 7, \
                              1, 1, \
                              2, \
                 kBias57_2, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k57, inputRows, inputCols, 1, 1, 2); \
    } \
 \
    /* Step (1, 2) */ \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel33_2, 3, 3, \
                              1, 2, \
                              2, \
                 kBias33_2, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k33, inputRows, inputCols, 1, 2, 2); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel55_2, 5, 5, \
                              1, 2, \
                              2, \
                 kBias55_2, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k55, inputRows, inputCols, 1, 2, 2); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel57_2, 5, 7, \
                              1, 2, \
                              2, \
                 kBias57_2, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k57, inputRows, inputCols, 1, 2, 2); \
    } \
 \
    /* Step (2, 1) */ \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel33_2, 3, 3, \
                              2, 1, \
                              2, \
                 kBias33_2, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k33, inputRows, inputCols, 2, 1, 2); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel55_2, 5, 5, \
                              2, 1, \
                              2, \
                 kBias55_2, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k55, inputRows, inputCols, 2, 1, 2); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel57_2, 5, 7, \
                              2, 1, \
                              2, \
                 kBias57_2, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k57, inputRows, inputCols, 2, 1, 2); \
    } \
 \
    /* Step (2, 2) */ \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel33_2, 3, 3, \
                              2, 2, \
                              2, \
                 kBias33_2, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k33, inputRows, inputCols, 2, 2, 2); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel55_2, 5, 5, \
                              2, 2, \
                              2, \
                 kBias55_2, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k55, inputRows, inputCols, 2, 2, 2); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel57_2, 5, 7, \
                              2, 2, \
                              2, \
                 kBias57_2, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k57, inputRows, inputCols, 2, 2, 2); \
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


void test66(const tTest& t)
{
    fml input[] =
    {  FML(0.013133),
   FML(0.955461),
    };
    u32 inputRows = 2;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(1.47511),   FML(0.73862),
   FML(1.35095),   FML(0.96515),
    };

    fml correctOutput_k55[] =
    {  FML(1.39385),   FML(1.19363),
   FML(1.45814),   FML(0.81461),
    };

    fml correctOutput_k57[] =
    {  FML(1.35643),   FML(1.42688),
   FML(1.24043),   FML(0.69168),
    };

    TEST_ALL_1_COMPONENT_2_COUNT_KERNELS
}


void test67(const tTest& t)
{
    fml input[] =
    {  FML(0.82757),
   FML(0.81545),
   FML(0.19833),
    };
    u32 inputRows = 3;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(1.79326),   FML(1.38926),
   FML(1.72581),   FML(1.29889),
   FML(1.28914),   FML(0.59705),
    };

    fml correctOutput_k55[] =
    {  FML(2.0085),   FML(1.5185),
   FML(2.2484),   FML(1.1709),
   FML(1.8472),   FML(1.2225),
    };

    fml correctOutput_k57[] =
    {  FML(1.6652),   FML(1.3590),
   FML(1.3574),   FML(1.5342),
   FML(1.4628),   FML(1.4929),
    };

    TEST_ALL_1_COMPONENT_2_COUNT_KERNELS
}


void test68(const tTest& t)
{
    fml input[] =
    {  FML(0.614772),
   FML(0.070874),
   FML(0.695933),
   FML(0.773934),
    };
    u32 inputRows = 4;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(1.22042),   FML(0.69592),
   FML(1.58297),   FML(0.86576),
   FML(1.72950),   FML(1.26974),
   FML(1.52902),   FML(1.07492),
    };

    fml correctOutput_k55[] =
    {  FML(1.6356),   FML(1.3387),
   FML(2.2726),   FML(1.8737),
   FML(1.9833),   FML(1.5943),
   FML(1.9793),   FML(1.0080),
    };

    fml correctOutput_k57[] =
    {  FML(1.11837),   FML(0.73185),
   FML(1.27969),   FML(1.74206),
   FML(1.97049),   FML(1.49930),
   FML(1.25293),   FML(1.26979),
    };

    TEST_ALL_1_COMPONENT_2_COUNT_KERNELS
}


void test69(const tTest& t)
{
    fml input[] =
    {  FML(0.32358),
   FML(0.48935),
   FML(0.49278),
   FML(0.41679),
   FML(0.33730),
    };
    u32 inputRows = 5;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(1.33763),   FML(0.71012),
   FML(1.54966),   FML(0.99459),
   FML(1.56893),   FML(1.01374),
   FML(1.48257),   FML(0.89242),
   FML(1.20161),   FML(0.56310),
    };

    fml correctOutput_k55[] =
    {  FML(1.58742),   FML(1.37411),
   FML(1.97040),   FML(1.46808),
   FML(2.10822),   FML(1.55420),
   FML(1.85172),   FML(1.26925),
   FML(1.49812),   FML(0.98090),
    };

    fml correctOutput_k57[] =
    {  FML(1.2280),   FML(1.0623),
   FML(1.3359),   FML(1.3453),
   FML(1.5097),   FML(1.4863),
   FML(1.5301),   FML(1.4470),
   FML(1.2863),   FML(1.0970),
    };

    TEST_ALL_1_COMPONENT_2_COUNT_KERNELS
}


void test70(const tTest& t)
{
    fml input[] =
    {  FML(0.802264),
   FML(0.493702),
   FML(0.305327),
   FML(0.285125),
   FML(0.223882),
   FML(0.097451),
    };
    u32 inputRows = 6;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(1.57894),   FML(1.15057),
   FML(1.62258),   FML(1.06626),
   FML(1.39467),   FML(0.75595),
   FML(1.27212),   FML(0.62040),
   FML(1.15439),   FML(0.47156),
   FML(1.00620),   FML(0.26597),
    };

    fml correctOutput_k55[] =
    {  FML(1.83532),   FML(1.37105),
   FML(2.22116),   FML(1.37728),
   FML(1.93574),   FML(1.53375),
   FML(1.56046),   FML(1.17106),
   FML(1.31317),   FML(0.88375),
   FML(1.09358),   FML(0.74513),
    };

    fml correctOutput_k57[] =
    {  FML(1.46092),   FML(1.09002),
   FML(1.26458),   FML(1.58884),
   FML(1.64300),   FML(1.46556),
   FML(1.38661),   FML(1.18952),
   FML(1.16096),   FML(1.02306),
   FML(1.02413),   FML(0.87874),
    };

    TEST_ALL_1_COMPONENT_2_COUNT_KERNELS
}


void test71(const tTest& t)
{
    fml input[] =
    {  FML(0.954710),
   FML(0.705332),
   FML(0.107897),
   FML(0.393215),
   FML(0.523242),
   FML(0.057792),
   FML(0.328289),
    };
    u32 inputRows = 7;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(1.78759),   FML(1.43170),
   FML(1.66424),   FML(1.18899),
   FML(1.44727),   FML(0.73336),
   FML(1.43602),   FML(0.84003),
   FML(1.32124),   FML(0.76226),
   FML(1.30998),   FML(0.57054),
   FML(1.05593),   FML(0.40991),
    };

    fml correctOutput_k55[] =
    {  FML(1.97710),   FML(1.40415),
   FML(2.44748),   FML(1.44292),
   FML(2.26532),   FML(1.93987),
   FML(1.68262),   FML(1.42295),
   FML(1.74527),   FML(1.16733),
   FML(1.58865),   FML(1.11327),
   FML(1.16353),   FML(0.87824),
    };

    fml correctOutput_k57[] =
    {  FML(1.66059),   FML(1.27257),
   FML(1.26097),   FML(1.56545),
   FML(1.72612),   FML(1.75879),
   FML(1.73509),   FML(1.32300),
   FML(1.16595),   FML(1.06242),
   FML(1.29361),   FML(1.42499),
   FML(1.27214),   FML(0.80107),
    };

    TEST_ALL_1_COMPONENT_2_COUNT_KERNELS
}


void test72(const tTest& t)
{
    fml input[] =
    {  FML(0.763171),
   FML(0.188399),
   FML(0.107985),
   FML(0.774685),
   FML(0.803080),
   FML(0.633849),
   FML(0.834126),
   FML(0.087026),
    };
    u32 inputRows = 8;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(1.36806),   FML(0.91030),
   FML(1.33132),   FML(0.63920),
   FML(1.48315),   FML(0.78028),
   FML(1.80162),   FML(1.37624),
   FML(1.97189),   FML(1.55801),
   FML(2.02426),   FML(1.54897),
   FML(1.58916),   FML(1.16318),
   FML(1.24101),   FML(0.50285),
    };

    fml correctOutput_k55[] =
    {  FML(1.49168),   FML(0.97481),
   FML(2.10498),   FML(1.53935),
   FML(2.15009),   FML(2.19222),
   FML(2.35181),   FML(1.98314),
   FML(2.97589),   FML(2.21401),
   FML(2.74822),   FML(1.99964),
   FML(2.19565),   FML(1.41647),
   FML(1.73762),   FML(1.10133),
    };

    fml correctOutput_k57[] =
    {  FML(1.25945),   FML(0.82915),
   FML(0.99697),   FML(1.38143),
   FML(1.79175),   FML(1.60877),
   FML(1.76046),   FML(1.48203),
   FML(1.67639),   FML(1.88905),
   FML(2.13675),   FML(2.19572),
   FML(1.79093),   FML(1.44406),
   FML(1.28850),   FML(1.46256),
    };

    TEST_ALL_1_COMPONENT_2_COUNT_KERNELS
}


void test73(const tTest& t)
{
    fml input[] =
    {  FML(0.87718),   FML(0.12306),
    };
    u32 inputRows = 1;
    u32 inputCols = 2;

    fml correctOutput_k33[] =
    {  FML(1.34573),   FML(0.96241),   FML(1.20704),   FML(0.32769),
    };

    fml correctOutput_k55[] =
    {  FML(1.45308),   FML(0.89013),   FML(1.46859),   FML(0.61858),
    };

    fml correctOutput_k57[] =
    {  FML(1.22706),   FML(0.74517),   FML(1.37842),   FML(1.25375),
    };

    TEST_ALL_1_COMPONENT_2_COUNT_KERNELS
}


void test74(const tTest& t)
{
    fml input[] =
    {  FML(0.79357),   FML(0.19883),   FML(0.55991),
    };
    u32 inputRows = 1;
    u32 inputCols = 3;

    fml correctOutput_k33[] =
    {  FML(1.32805),   FML(0.93163),   FML(1.39576),   FML(0.72193),   FML(1.21123),   FML(0.62746),
    };

    fml correctOutput_k55[] =
    {  FML(1.70962),   FML(1.34170),   FML(1.75740),   FML(1.11114),   FML(1.88208),   FML(0.79436),
    };

    fml correctOutput_k57[] =
    {  FML(1.6554),   FML(1.3277),   FML(1.4790),   FML(1.5127),   FML(1.3164),   FML(1.0391),
    };

    TEST_ALL_1_COMPONENT_2_COUNT_KERNELS
}


void test75(const tTest& t)
{
    fml input[] =
    {  FML(0.44639),   FML(0.60285),   FML(0.81655),   FML(0.99021),
    };
    u32 inputRows = 1;
    u32 inputCols = 4;

    fml correctOutput_k33[] =
    {  FML(1.28293),   FML(0.85763),   FML(1.56909),   FML(1.19485),   FML(1.77983),   FML(1.51775),   FML(1.62015),   FML(1.11142),
    };

    fml correctOutput_k55[] =
    {  FML(1.79466),   FML(1.76442),   FML(2.42930),   FML(2.16023),   FML(2.62559),   FML(1.70878),   FML(2.50245),   FML(0.95881),
    };

    fml correctOutput_k57[] =
    {  FML(2.3839),   FML(2.2992),   FML(2.3139),   FML(2.3973),   FML(1.8295),   FML(1.7891),   FML(2.2489),   FML(1.7385),
    };

    TEST_ALL_1_COMPONENT_2_COUNT_KERNELS
}


void test76(const tTest& t)
{
    fml input[] =
    {  FML(0.42982),   FML(0.55418),   FML(0.82427),   FML(0.87749),   FML(0.19492),
    };
    u32 inputRows = 1;
    u32 inputCols = 5;

    fml correctOutput_k33[] =
     {  FML(1.25926),   FML(0.81317),   FML(1.54206),   FML(1.15258),   FML(1.73267),   FML(1.44976),   FML(1.62811),   FML(1.12692),  FML(1.24295),   FML(0.39342),
    };

    fml correctOutput_k55[] =
     {  FML(1.76084),   FML(1.72354),   FML(2.32978),   FML(2.06673),   FML(2.62169),   FML(1.75362),   FML(2.49194),   FML(1.08354),  FML(2.12652),   FML(0.73788),
    };

    fml correctOutput_k57[] =
     {  FML(2.3033),   FML(2.2198),   FML(2.3097),   FML(2.3789),   FML(1.9347),   FML(1.8766),   FML(2.2165),   FML(1.8211),  FML(2.0355),   FML(1.8751),
     };

    TEST_ALL_1_COMPONENT_2_COUNT_KERNELS
}


void test77(const tTest& t)
{
    fml input[] =
    {  FML(0.759691),   FML(0.401859),   FML(0.272445),   FML(0.807054),   FML(0.612789),   FML(0.031398),
    };
    u32 inputRows = 1;
    u32 inputCols = 6;

    fml correctOutput_k33[] =
     {  FML(1.37545),   FML(1.02294),   FML(1.39527),   FML(0.72938),   FML(1.38739),   FML(0.88059),   FML(1.55160),   FML(1.23326),  FML(1.43899),   FML(0.78395),   FML(1.07812),   FML(0.20511),
    };

    fml correctOutput_k55[] =
     {  FML(1.64852),   FML(1.29095),   FML(2.13169),   FML(1.52489),   FML(2.52051),   FML(1.85399),   FML(2.16548),   FML(1.37628),  FML(1.99314),   FML(0.82648),   FML(1.80333),   FML(0.66243),
    };

    fml correctOutput_k57[] =
     {  FML(1.9413),   FML(1.5796),   FML(2.5238),   FML(2.4431),   FML(1.9782),   FML(2.2210),   FML(2.2074),   FML(1.8642),  FML(1.9694),   FML(1.5984),   FML(1.5509),   FML(1.4866),
     };

    TEST_ALL_1_COMPONENT_2_COUNT_KERNELS
}


void test78(const tTest& t)
{
    fml input[] =
    {  FML(0.99473),   FML(0.50862),   FML(0.27996),   FML(0.73488),   FML(0.90786),   FML(0.29745),   FML(0.16796),
    };
    u32 inputRows = 1;
    u32 inputCols = 7;

    fml correctOutput_k33[] =
     {  FML(1.52639),   FML(1.30208),   FML(1.52484),   FML(0.86598),   FML(1.40189),   FML(0.85966),   FML(1.61142),   FML(1.34612),  FML(1.64757),   FML(1.20332),   FML(1.35679),   FML(0.59275),   FML(1.04692),   FML(0.28366),
    };

    fml correctOutput_k55[] =
     {  FML(1.88035),   FML(1.46158),   FML(2.34740),   FML(1.53098),   FML(2.88419),   FML(2.04937),   FML(2.48325),   FML(1.81365),  FML(2.38562),   FML(1.26490),   FML(2.24594),   FML(0.90497),   FML(1.75003),   FML(0.69169),
    };

    fml correctOutput_k57[] =
     {  FML(2.0385),   FML(1.6225),   FML(2.8432),   FML(2.7028),   FML(2.4768),   FML(2.7531),   FML(2.7755),   FML(2.5652),  FML(2.3501),   FML(1.9497),   FML(1.8886),   FML(1.7876),   FML(1.8345),   FML(1.6086),
     };

    TEST_ALL_1_COMPONENT_2_COUNT_KERNELS
}


void test79(const tTest& t)
{
    fml input[] =
     {  FML(0.055739),   FML(0.626009),   FML(0.869583),   FML(0.697015),   FML(0.920122),   FML(0.583032),   FML(0.078973),  FML(0.896550),
    };
    u32 inputRows = 1;
    u32 inputCols = 8;

    fml correctOutput_k33[] =
     {  FML(1.09556),   FML(0.51450),   FML(1.47445),   FML(1.19065),   FML(1.72073),   FML(1.39302),   FML(1.78203),   FML(1.40540),  FML(1.73218),   FML(1.38098),   FML(1.47481),   FML(0.80200),   FML(1.37632),   FML(0.78424),   FML(1.34128),   FML(0.91758),
    };

    fml correctOutput_k55[] =
     {  FML(1.54706),   FML(1.69956),   FML(2.04688),   FML(1.96401),   FML(2.69754),   FML(2.11263),   FML(3.11495),   FML(2.08505),  FML(2.90569),   FML(1.51146),   FML(2.83467),   FML(1.57886),   FML(2.36982),   FML(1.44954),   FML(1.88765),   FML(0.86651),
    };

    fml correctOutput_k57[] =
     {  FML(2.0644),   FML(2.1891),   FML(2.4203),   FML(2.3455),   FML(2.8379),   FML(2.7244),   FML(2.5219),   FML(2.6313),  FML(3.0749),   FML(2.7197),   FML(3.2355),   FML(3.0144),   FML(2.1225),   FML(2.2865),   FML(2.1492),   FML(1.5198),
     };

    TEST_ALL_1_COMPONENT_2_COUNT_KERNELS
}


void test80(const tTest& t)
{
    fml input[] =
     {  FML(0.235445),   FML(0.588533),   FML(0.422798),   FML(0.596196),   FML(0.673341),   FML(0.117915),   FML(0.639336),  FML(0.283129),
    FML(0.519378),   FML(0.273980),   FML(0.748642),   FML(0.654390),   FML(0.236341),   FML(0.503418),   FML(0.557124),  FML(0.623233),
    FML(0.576177),   FML(0.977037),   FML(0.598682),   FML(0.386034),   FML(0.463120),   FML(0.281903),   FML(0.397778),  FML(0.346687),
    FML(0.402726),   FML(0.746217),   FML(0.783786),   FML(0.752348),   FML(0.982114),   FML(0.204472),   FML(0.181106),  FML(0.538679),
    FML(0.150195),   FML(0.892030),   FML(0.798167),   FML(0.845185),   FML(0.915378),   FML(0.428578),   FML(0.272652),  FML(0.760332),
    FML(0.372624),   FML(0.198441),   FML(0.486771),   FML(0.053301),   FML(0.125731),   FML(0.208560),   FML(0.443974),  FML(0.468624),
    FML(0.893004),   FML(0.459111),   FML(0.013819),   FML(0.898380),   FML(0.693117),   FML(0.129526),   FML(0.069245),  FML(0.899693),
    FML(0.014718),   FML(0.216360),   FML(0.684710),   FML(0.659945),   FML(0.866187),   FML(0.993860),   FML(0.333889),  FML(0.647957),
    };
    u32 inputRows = 8;
    u32 inputCols = 8;

    fml correctOutput_k33[] =
     {  FML(1.53377),   FML(1.15628),   FML(2.14477),   FML(1.80739),   FML(2.27435),   FML(1.93867),   FML(2.68379),   FML(2.09318),  FML(2.28096),   FML(1.66937),   FML(1.95990),   FML(1.45878),   FML(2.23483),   FML(1.86346),   FML(2.14623),   FML(1.17253),
    FML(1.87655),   FML(2.01762),   FML(3.01488),   FML(2.84062),   FML(3.32966),   FML(3.39865),   FML(2.83964),   FML(2.75734),  FML(2.50456),   FML(2.30266),   FML(2.37628),   FML(2.59370),   FML(2.39616),   FML(2.16742),   FML(2.25194),   FML(1.94240),
    FML(2.05658),   FML(2.22016),   FML(3.04737),   FML(3.55313),   FML(3.37680),   FML(3.24656),   FML(3.25956),   FML(3.41214),  FML(3.03478),   FML(2.90606),   FML(2.73431),   FML(2.13663),   FML(2.09649),   FML(2.24308),   FML(2.08358),   FML(1.72802),
    FML(1.87890),   FML(2.17718),   FML(3.05448),   FML(3.66251),   FML(3.77608),   FML(4.25912),   FML(3.61281),   FML(3.96652),  FML(3.44409),   FML(3.24528),   FML(2.83161),   FML(2.23699),   FML(2.16527),   FML(2.05152),   FML(2.18582),   FML(1.80286),
    FML(1.74948),   FML(1.62302),   FML(2.68138),   FML(3.07540),   FML(2.94540),   FML(3.32109),   FML(3.01093),   FML(3.43487),  FML(2.51138),   FML(2.76933),   FML(2.14946),   FML(2.37066),   FML(2.12733),   FML(2.05292),   FML(2.32405),   FML(1.78382),
    FML(1.92145),   FML(1.86632),   FML(2.90645),   FML(2.30922),   FML(2.44635),   FML(2.95018),   FML(2.43997),   FML(2.82414),  FML(3.00050),   FML(2.78108),   FML(2.38409),   FML(2.28370),   FML(1.92423),   FML(2.30450),   FML(2.25010),   FML(1.79869),
    FML(1.67137),   FML(1.55002),   FML(1.87253),   FML(1.84914),   FML(2.28455),   FML(2.07252),   FML(2.90870),   FML(3.19868),  FML(2.94624),   FML(2.63460),   FML(2.86498),   FML(2.10123),   FML(2.79061),   FML(2.45645),   FML(2.38077),   FML(2.17079),
    FML(1.36144),   FML(0.79973),   FML(1.63216),   FML(1.77376),   FML(1.74910),   FML(2.01002),   FML(2.14334),   FML(2.00367),  FML(2.36690),   FML(2.80106),   FML(1.99869),   FML(2.09250),   FML(1.74462),   FML(1.49320),   FML(1.67092),   FML(1.15952),
    };

    fml correctOutput_k55[] =
     {  FML(3.2172),   FML(3.1697),   FML(3.9504),   FML(3.9678),   FML(4.4179),   FML(4.4699),   FML(4.2958),   FML(4.1783),  FML(4.2882),   FML(3.6895),   FML(4.2109),   FML(3.4544),   FML(3.3911),   FML(2.8563),   FML(2.4512),   FML(2.2846),
    FML(4.2477),   FML(4.0371),   FML(5.0659),   FML(5.4053),   FML(5.5342),   FML(6.1601),   FML(5.6479),   FML(5.7298),  FML(5.6215),   FML(5.2181),   FML(4.6203),   FML(5.0623),   FML(4.2597),   FML(3.7937),   FML(3.1921),   FML(2.3066),
    FML(5.3000),   FML(4.6134),   FML(6.5090),   FML(6.2055),   FML(8.0601),   FML(7.3017),   FML(7.7327),   FML(6.8847),  FML(6.3410),   FML(6.8870),   FML(6.0596),   FML(6.3386),   FML(5.2670),   FML(4.4887),   FML(3.6306),   FML(3.3205),
    FML(5.3531),   FML(4.8854),   FML(6.9522),   FML(5.8303),   FML(8.0986),   FML(6.8119),   FML(7.1037),   FML(6.4166),  FML(6.9745),   FML(5.3594),   FML(6.5041),   FML(5.0596),   FML(5.0732),   FML(4.4162),   FML(3.6016),   FML(3.3075),
    FML(5.1773),   FML(5.3033),   FML(5.4695),   FML(6.2933),   FML(7.0166),   FML(7.0804),   FML(7.6013),   FML(6.5505),  FML(6.7965),   FML(4.9452),   FML(6.1062),   FML(5.0433),   FML(5.0349),   FML(4.4859),   FML(3.7576),   FML(3.1061),
    FML(4.1544),   FML(4.1004),   FML(5.4869),   FML(5.4680),   FML(7.0665),   FML(6.1566),   FML(7.0196),   FML(6.7970),  FML(6.0401),   FML(6.6731),   FML(6.1155),   FML(6.6714),   FML(5.5043),   FML(5.0534),   FML(4.2241),   FML(3.5630),
    FML(4.1077),   FML(2.5856),   FML(5.1999),   FML(3.5706),   FML(6.4799),   FML(4.9352),   FML(6.5846),   FML(4.9060),  FML(5.8001),   FML(4.5024),   FML(6.2998),   FML(5.0599),   FML(5.1162),   FML(4.1866),   FML(3.5283),   FML(2.6829),
    FML(2.7289),   FML(2.2851),   FML(2.9285),   FML(2.8426),   FML(3.6264),   FML(4.0651),   FML(4.5173),   FML(3.9493),  FML(4.8823),   FML(2.8243),   FML(4.5469),   FML(3.1905),   FML(3.9034),   FML(3.1614),   FML(3.4089),   FML(1.7295),
    };

    fml correctOutput_k57[] =
      {  FML(4.7202),    FML(3.7638),    FML(5.1826),    FML(4.1842),    FML(5.9460),    FML(5.4340),    FML(6.3544),    FML(5.1976),   FML(7.1490),    FML(5.2847),    FML(5.4544),    FML(4.9167),    FML(4.1193),    FML(3.4356),    FML(3.9517),    FML(3.2565),
     FML(5.4191),    FML(5.1611),    FML(6.4449),    FML(6.8469),    FML(7.3174),    FML(7.2304),    FML(8.3474),    FML(8.2880),   FML(7.3112),    FML(8.3452),    FML(7.0467),    FML(7.4009),    FML(5.3048),    FML(5.7578),    FML(4.2421),    FML(4.1351),
     FML(7.1122),    FML(6.6268),    FML(8.9167),    FML(7.9226),    FML(9.6328),    FML(9.5804),   FML(10.5351),   FML(10.7566),  FML(11.0353),   FML(10.7647),    FML(9.2136),    FML(9.6862),    FML(7.2183),    FML(7.4129),    FML(5.0484),    FML(5.4993),
     FML(6.5998),    FML(5.9670),    FML(8.0648),    FML(8.1475),    FML(8.6900),    FML(8.8841),   FML(10.0190),   FML(10.1843),  FML(10.5926),   FML(10.5291),    FML(9.2684),    FML(9.0821),    FML(6.3716),    FML(6.9262),    FML(5.4176),    FML(5.6938),
     FML(5.4945),    FML(6.2283),    FML(7.8811),    FML(8.2796),    FML(8.9592),   FML(10.0842),    FML(9.0572),    FML(9.4903),   FML(9.8226),   FML(10.4030),    FML(8.1940),    FML(9.4099),    FML(6.8639),    FML(7.1817),    FML(5.1886),    FML(4.9805),
     FML(6.4067),    FML(6.0588),    FML(7.9891),    FML(7.4176),    FML(8.5553),    FML(8.9310),    FML(9.4840),   FML(11.1633),  FML(10.4568),   FML(10.7959),    FML(9.4560),    FML(9.9027),    FML(7.4806),    FML(8.3734),    FML(5.2823),    FML(5.9134),
     FML(5.0653),    FML(4.0463),    FML(7.1896),    FML(5.8874),    FML(7.2716),    FML(6.9413),    FML(8.0331),    FML(7.0934),   FML(9.3623),    FML(8.0957),    FML(7.4535),    FML(7.8558),    FML(6.2376),    FML(6.6512),    FML(5.2875),    FML(5.1310),
     FML(2.8719),    FML(3.3939),    FML(3.5753),    FML(4.6807),    FML(4.5389),    FML(5.4375),    FML(4.8217),    FML(6.3267),   FML(5.2015),    FML(5.9497),    FML(5.0199),    FML(5.8468),    FML(4.0260),    FML(5.1559),    FML(3.4191),    FML(3.8997),
      };

    TEST_ALL_1_COMPONENT_2_COUNT_KERNELS
}


fml kKernel33_3[] = {
    FML(0.949953),   FML(0.429395),   FML(0.155611),   FML(0.942828),   FML(0.153559),   FML(0.411244),
    FML(0.620832),   FML(0.718138),   FML(0.164167),   FML(0.294378),   FML(0.326532),   FML(0.047242),
    FML(0.858914),   FML(0.426591),   FML(0.679221),   FML(0.839225),   FML(0.849002),   FML(0.837592),
};
fml kBias33_3     = FML(0.32018);

fml kKernel55_3[] = {
    FML(0.190823),   FML(0.090197),   FML(0.145543),   FML(0.835100),   FML(0.697089),   FML(0.518416),   FML(0.254674),  FML(0.069730),   FML(0.056365),   FML(0.078404),
    FML(0.942368),   FML(0.504369),   FML(0.995490),   FML(0.857426),   FML(0.963833),   FML(0.420007),   FML(0.245717),  FML(0.169354),   FML(0.950606),   FML(0.332520),
    FML(0.605449),   FML(0.760562),   FML(0.105182),   FML(0.970178),   FML(0.158836),   FML(0.612095),   FML(0.288663),  FML(0.428532),   FML(0.180745),   FML(0.421864),
    FML(0.089072),   FML(0.591821),   FML(0.510324),   FML(0.370252),   FML(0.116792),   FML(0.466311),   FML(0.087237),  FML(0.174164),   FML(0.461513),   FML(0.449285),
    FML(0.020371),   FML(0.159996),   FML(0.271759),   FML(0.261803),   FML(0.173699),   FML(0.611178),   FML(0.347460),  FML(0.769762),   FML(0.357367),   FML(0.044347),
};
fml kBias55_3     = FML(0.84436);

fml kKernel57_3[] = {
    FML(0.0213532),   FML(0.1999500),   FML(0.7609853),   FML(0.8745935),   FML(0.0815307),   FML(0.0016322),  FML(0.7562107),   FML(0.3719662),   FML(0.3359384),   FML(0.6992579),   FML(0.3825840),   FML(0.9852955),  FML(0.9738489),   FML(0.4974391),
    FML(0.2762497),   FML(0.0456990),   FML(0.5038456),   FML(0.4030547),   FML(0.4023364),   FML(0.9566434),  FML(0.0971793),   FML(0.6499797),   FML(0.7745309),   FML(0.2905131),   FML(0.4940301),   FML(0.0828166),  FML(0.8771658),   FML(0.9233643),
    FML(0.6586268),   FML(0.2488343),   FML(0.4879419),   FML(0.1402583),   FML(0.7912674),   FML(0.7572233),  FML(0.7936432),   FML(0.3977686),   FML(0.8955799),   FML(0.1854243),   FML(0.0412163),   FML(0.9565226),  FML(0.2703150),   FML(0.9532071),
    FML(0.2467477),   FML(0.9606165),   FML(0.5741950),   FML(0.5516112),   FML(0.0191520),   FML(0.6962618),  FML(0.1872503),   FML(0.5046333),   FML(0.6357956),   FML(0.1834882),   FML(0.0618589),   FML(0.5941736),  FML(0.9719768),   FML(0.5786454),
    FML(0.8293359),   FML(0.2747554),   FML(0.0195070),   FML(0.3395821),   FML(0.8516997),   FML(0.3573254),  FML(0.6979173),   FML(0.6837583),   FML(0.5113474),   FML(0.1848593),   FML(0.6939300),   FML(0.0303456),  FML(0.6107402),   FML(0.1582097),
};
fml kBias57_3     = FML(0.88555);


#define TEST_ALL_2_COMPONENT_1_COUNT_KERNELS \
    fml* output = new fml[inputRows*inputCols]; \
    for (u32 i = 0; i < inputRows*inputCols; i++) \
        output[i] = FML(0.0); \
 \
    /* Step (1, 1) */ \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 2, \
                 kKernel33_3, 3, 3, \
                              1, 1, \
                              1, \
                 &kBias33_3, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k33, inputRows, inputCols); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 2, \
                 kKernel55_3, 5, 5, \
                              1, 1, \
                              1, \
                 &kBias55_3, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k55, inputRows, inputCols); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 2, \
                 kKernel57_3, 5, 7, \
                              1, 1, \
                              1, \
                 &kBias57_3, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k57, inputRows, inputCols); \
    } \
 \
    /* Step (1, 2) */ \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 2, \
                 kKernel33_3, 3, 3, \
                              1, 2, \
                              1, \
                 &kBias33_3, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k33, inputRows, inputCols, 1, 2); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 2, \
                 kKernel55_3, 5, 5, \
                              1, 2, \
                              1, \
                 &kBias55_3, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k55, inputRows, inputCols, 1, 2); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 2, \
                 kKernel57_3, 5, 7, \
                              1, 2, \
                              1, \
                 &kBias57_3, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k57, inputRows, inputCols, 1, 2); \
    } \
 \
    /* Step (2, 1) */ \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 2, \
                 kKernel33_3, 3, 3, \
                              2, 1, \
                              1, \
                 &kBias33_3, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k33, inputRows, inputCols, 2, 1); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 2, \
                 kKernel55_3, 5, 5, \
                              2, 1, \
                              1, \
                 &kBias55_3, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k55, inputRows, inputCols, 2, 1); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 2, \
                 kKernel57_3, 5, 7, \
                              2, 1, \
                              1, \
                 &kBias57_3, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k57, inputRows, inputCols, 2, 1); \
    } \
 \
    /* Step (2, 2) */ \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 2, \
                 kKernel33_3, 3, 3, \
                              2, 2, \
                              1, \
                 &kBias33_3, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k33, inputRows, inputCols, 2, 2); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 2, \
                 kKernel55_3, 5, 5, \
                              2, 2, \
                              1, \
                 &kBias55_3, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k55, inputRows, inputCols, 2, 2); \
    } \
 \
    { \
        ml::s_conv2d(input, inputRows, inputCols, 2, \
                 kKernel57_3, 5, 7, \
                              2, 2, \
                              1, \
                 &kBias57_3, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k57, inputRows, inputCols, 2, 2); \
    } \
 \
    delete [] output; \


void test81(const tTest& t)
{
    fml input[] = {  FML(0.72890),   FML(0.41273),  };
    u32 inputRows = 1;
    u32 inputCols = 1;

    fml correctOutput_k33[] = {  FML(0.56134),  };

    fml correctOutput_k55[] = {  FML(1.2128),  };

    fml correctOutput_k57[] = {  FML(1.6282),  };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test82(const tTest& t)
{
    fml input[] =
    {  FML(0.616663),   FML(0.084653),   FML(0.510743),   FML(0.927563),
    };
    u32 inputRows = 1;
    u32 inputCols = 2;

    fml correctOutput_k33[] =
    {  FML(0.65693),   FML(1.12072),
    };

    fml correctOutput_k55[] =
    {  FML(1.5390),   FML(1.6402),
    };

    fml correctOutput_k57[] =
    {  FML(2.0380),   FML(2.2119),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test83(const tTest& t)
{
    fml input[] =
    {  FML(0.66304),   FML(0.17153),   FML(0.28597),   FML(0.66619),   FML(0.84100),   FML(0.35952),
    };
    u32 inputRows = 1;
    u32 inputCols = 3;

    fml correctOutput_k33[] =
    {  FML(0.60438),   FML(1.38966),   FML(1.22003),
    };

    fml correctOutput_k55[] =
    {  FML(1.7264),   FML(1.9305),   FML(2.4063),
    };

    fml correctOutput_k57[] =
    {  FML(2.2382),   FML(2.8519),   FML(2.7743),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test84(const tTest& t)
{
    fml input[] =
    {  FML(0.67717),   FML(0.52460),   FML(0.37139),   FML(0.10462),   FML(0.95209),   FML(0.77208),   FML(0.67718),   FML(0.93950),
    };
    u32 inputRows = 1;
    u32 inputCols = 4;

    fml correctOutput_k33[] =
    {  FML(0.71199),   FML(1.55646),   FML(1.27497),   FML(1.85347),
    };

    fml correctOutput_k55[] =
    {  FML(1.9229),   FML(2.6720),   FML(3.0158),   FML(2.6806),
    };

    fml correctOutput_k57[] =
    {  FML(3.8400),   FML(4.0774),   FML(3.5060),   FML(3.9071),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test85(const tTest& t)
{
    fml input[] =
     {  FML(0.136919),   FML(0.176414),   FML(0.834235),   FML(0.684148),   FML(0.887645),   FML(0.062487),   FML(0.052352),  FML(0.191065),   FML(0.800266),   FML(0.937010),
    };
    u32 inputRows = 1;
    u32 inputCols = 5;

    fml correctOutput_k33[] =
    {  FML(0.69932),   FML(1.16302),   FML(1.51965),   FML(1.28655),   FML(0.89711),
    };

    fml correctOutput_k55[] =
    {  FML(1.6949),   FML(1.9543),   FML(2.6291),   FML(2.7816),   FML(2.3208),
    };

    fml correctOutput_k57[] =
    {  FML(2.2310),   FML(4.1626),   FML(3.8962),   FML(3.2803),   FML(3.2411),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test86(const tTest& t)
{
    fml input[] =
     {  FML(0.4335509),   FML(0.4475875),   FML(0.9553051),   FML(0.4285347),   FML(0.8512858),   FML(0.8698196),  FML(0.9550344),   FML(0.1262155),   FML(0.0785071),   FML(0.2315102),   FML(0.6246067),   FML(0.0066509),
    };
    u32 inputRows = 1;
    u32 inputCols = 6;

    fml correctOutput_k33[] =
    {  FML(0.85530),   FML(1.51282),   FML(1.93463),   FML(1.70385),   FML(1.28904),   FML(0.63967),
    };

    fml correctOutput_k55[] =
    {  FML(2.1674),   FML(2.5826),   FML(3.0728),   FML(3.1486),   FML(2.5815),   FML(1.8547),
    };

    fml correctOutput_k57[] =
    {  FML(3.5882),   FML(3.8218),   FML(4.5404),   FML(4.0945),   FML(3.7250),   FML(2.8822),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test87(const tTest& t)
{
    fml input[] =
     {  FML(0.408465),   FML(0.216876),   FML(0.237262),   FML(0.058396),   FML(0.738701),   FML(0.296075),   FML(0.401029),  FML(0.998387),   FML(0.806837),   FML(0.297910),   FML(0.999812),   FML(0.736789),   FML(0.782612),   FML(0.792392),
    };
    u32 inputRows = 1;
    u32 inputCols = 7;

    fml correctOutput_k33[] =
    {  FML(0.53131),   FML(1.04085),   FML(0.89596),   FML(1.62868),   FML(1.86756),   FML(1.70904),   FML(1.83175),
    };

    fml correctOutput_k55[] =
    {  FML(1.3939),   FML(2.0049),   FML(2.4519),   FML(2.9243),   FML(3.9182),   FML(3.3957),   FML(2.9887),
    };

    fml correctOutput_k57[] =
    {  FML(2.8930),   FML(3.7745),   FML(3.8863),   FML(5.3473),   FML(5.1127),   FML(4.5799),   FML(4.1189),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test88(const tTest& t)
{
    fml input[] =
      {  FML(0.130457),   FML(0.961893),   FML(0.741034),   FML(0.147846),   FML(0.304180),   FML(0.098890),   FML(0.668044),  FML(0.545613),   FML(0.064649),   FML(0.879900),   FML(0.246002),   FML(0.850785),   FML(0.079074),   FML(0.652209),  FML(0.103233),   FML(0.744241),
    };
    u32 inputRows = 1;
    u32 inputCols = 8;

    fml correctOutput_k33[] =
    {  FML(0.87371),   FML(1.36112),   FML(1.20937),   FML(0.91301),   FML(1.51691),   FML(1.33968),   FML(1.35773),   FML(1.07368),
    };

    fml correctOutput_k55[] =
    {  FML(1.8278),   FML(2.4806),   FML(2.7947),   FML(2.7726),   FML(2.9772),   FML(3.7190),   FML(3.1645),   FML(2.7534),
    };

    fml correctOutput_k57[] =
    {  FML(2.8706),   FML(4.0604),   FML(4.4844),   FML(4.3440),   FML(4.6583),   FML(3.6720),   FML(3.0078),   FML(2.3208),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test89(const tTest& t)
{
    fml input[] =
    {  FML(0.51521),   FML(0.75933),
   FML(0.51037),   FML(0.35091),
    };
    u32 inputRows = 2;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(1.2694),
   FML(1.3034),
    };

    fml correctOutput_k55[] =
    {  FML(1.6142),
   FML(1.9557),
    };

    fml correctOutput_k57[] =
    {  FML(1.8691),
   FML(1.9738),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test90(const tTest& t)
{
    fml input[] =
    {  FML(0.98106),   FML(0.17031),   FML(0.64179),   FML(0.17990),
   FML(0.94254),   FML(0.90724),   FML(0.33854),   FML(0.20517),
    };
    u32 inputRows = 2;
    u32 inputCols = 2;

    fml correctOutput_k33[] =
    {  FML(2.6103),   FML(2.8086),
   FML(1.3480),   FML(2.9474),
    };

    fml correctOutput_k55[] =
    {  FML(1.9652),   FML(2.2770),
   FML(2.9403),   FML(3.8198),
    };

    fml correctOutput_k57[] =
    {  FML(3.2272),   FML(3.1884),
   FML(3.0911),   FML(3.4056),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test91(const tTest& t)
{
    fml input[] =
    {  FML(0.61307),   FML(0.74391),   FML(0.78476),   FML(0.55504),   FML(0.43909),   FML(0.32646),
   FML(0.43487),   FML(0.92640),   FML(0.52259),   FML(0.30364),   FML(0.31581),   FML(0.30547),
    };
    u32 inputRows = 2;
    u32 inputCols = 3;

    fml correctOutput_k33[] =
    {  FML(2.6931),   FML(3.5885),   FML(2.4234),
   FML(1.9948),   FML(3.2971),   FML(2.3644),
    };

    fml correctOutput_k55[] =
    {  FML(2.9428),   FML(3.2099),   FML(3.8174),
   FML(3.6635),   FML(4.6806),   FML(5.1693),
    };

    fml correctOutput_k57[] =
    {  FML(3.9420),   FML(4.3926),   FML(4.0040),
   FML(3.9848),   FML(4.6364),   FML(3.9536),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test92(const tTest& t)
{
    fml input[] =
     {  FML(0.438525),   FML(0.886649),   FML(0.666517),   FML(0.074898),   FML(0.114775),   FML(0.999479),   FML(0.099609),  FML(0.828174),
    FML(0.079335),   FML(0.623688),   FML(0.753824),   FML(0.178228),   FML(0.023571),   FML(0.454736),   FML(0.584967),  FML(0.211405),
    };
    u32 inputRows = 2;
    u32 inputCols = 4;

    fml correctOutput_k33[] =
    {  FML(2.2409),   FML(2.8420),   FML(2.9673),   FML(2.1583),
   FML(1.8087),   FML(2.4230),   FML(3.2363),   FML(2.1542),
    };

    fml correctOutput_k55[] =
    {  FML(2.7357),   FML(3.6200),   FML(4.0702),   FML(3.3293),
   FML(3.1418),   FML(4.5214),   FML(4.4438),   FML(4.1832),
    };

    fml correctOutput_k57[] =
    {  FML(5.7783),   FML(4.4742),   FML(3.7121),   FML(4.6700),
   FML(4.8488),   FML(4.0567),   FML(4.4025),   FML(4.4751),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test93(const tTest& t)
{
    fml input[] =
     {  FML(0.75529),   FML(0.85786),   FML(0.56790),   FML(0.77456),   FML(0.33291),   FML(0.58240),   FML(0.76291),   FML(0.47753),  FML(0.71546),   FML(0.87634),
    FML(0.89354),   FML(0.25331),   FML(0.60075),   FML(0.64517),   FML(0.63071),   FML(0.86058),   FML(0.18400),   FML(0.77690),  FML(0.48138),   FML(0.82068),
    };
    u32 inputRows = 2;
    u32 inputCols = 5;

    fml correctOutput_k33[] =
    {  FML(2.7887),   FML(4.9439),   FML(4.4755),   FML(4.2678),   FML(3.0173),
   FML(2.1002),   FML(3.7871),   FML(3.3966),   FML(3.3902),   FML(3.1803),
    };

    fml correctOutput_k55[] =
    {  FML(3.3561),   FML(4.5651),   FML(6.0907),   FML(5.0174),   FML(4.0684),
   FML(3.9374),   FML(6.0108),   FML(7.8926),   FML(6.6121),   FML(6.0667),
    };

    fml correctOutput_k57[] =
    {  FML(5.6849),   FML(7.4209),   FML(6.9885),   FML(6.9444),   FML(6.2317),
   FML(6.6113),   FML(8.8688),   FML(7.5399),   FML(6.8403),   FML(5.3102),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test94(const tTest& t)
{
    fml input[] =
     {  FML(0.6281141),   FML(0.9715067),   FML(0.4635012),   FML(0.0992576),   FML(0.6555763),   FML(0.4276631),  FML(0.2878540),   FML(0.1851725),   FML(0.8863978),   FML(0.4119649),   FML(0.5978300),   FML(0.9212854),
    FML(0.6455811),   FML(0.9978208),   FML(0.0022565),   FML(0.3808842),   FML(0.0935457),   FML(0.7520841),  FML(0.6946868),   FML(0.9044056),   FML(0.5273966),   FML(0.7368179),   FML(0.5871047),   FML(0.5969142),
    };
    u32 inputRows = 2;
    u32 inputCols = 6;

    fml correctOutput_k33[] =
    {  FML(2.4622),   FML(3.7581),   FML(3.2219),   FML(4.1419),   FML(4.0949),   FML(3.2027),
   FML(1.8643),   FML(3.0722),   FML(2.2099),   FML(2.8370),   FML(3.2743),   FML(3.4294),
    };

    fml correctOutput_k55[] =
    {  FML(3.0023),   FML(4.2328),   FML(4.9153),   FML(4.5407),   FML(4.4400),   FML(3.7972),
   FML(3.9653),   FML(5.4856),   FML(7.3550),   FML(6.3883),   FML(5.9927),   FML(5.7452),
    };

    fml correctOutput_k57[] =
    {  FML(5.2413),   FML(6.6298),   FML(7.8006),   FML(8.0045),   FML(5.9684),   FML(6.0220),
   FML(5.5037),   FML(7.5062),   FML(7.7032),   FML(6.8628),   FML(5.7284),   FML(5.1067),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test95(const tTest& t)
{
    fml input[] =
      {  FML(7.5371e-01),   FML(6.7931e-01),   FML(8.1969e-01),   FML(9.7713e-01),   FML(8.1963e-02),   FML(6.0360e-01),  FML(9.9345e-01),   FML(7.4763e-02),   FML(4.2322e-01),   FML(9.6443e-01),   FML(1.5075e-01),   FML(3.2370e-01),  FML(3.8067e-01),   FML(7.2881e-01),
     FML(6.0505e-01),   FML(4.3344e-01),   FML(4.8418e-01),   FML(6.7734e-02),   FML(6.5336e-01),   FML(2.6482e-04),  FML(2.1044e-01),   FML(5.1282e-01),   FML(5.0984e-01),   FML(1.3106e-01),   FML(8.9374e-01),   FML(6.6442e-01),  FML(3.9282e-01),   FML(1.0148e-01),
    };
    u32 inputRows = 2;
    u32 inputCols = 7;

    fml correctOutput_k33[] =
    {  FML(2.2002),   FML(3.3987),   FML(3.5468),   FML(2.8506),   FML(3.5796),   FML(3.6313),   FML(2.3263),
   FML(1.9939),   FML(3.6372),   FML(2.8331),   FML(2.1079),   FML(3.3719),   FML(2.7092),   FML(2.4751),
    };

    fml correctOutput_k55[] =
    {  FML(2.9331),   FML(3.7464),   FML(5.0971),   FML(5.0396),   FML(3.6430),   FML(4.2243),   FML(3.5888),
   FML(3.1500),   FML(5.5148),   FML(6.2079),   FML(5.6221),   FML(5.5213),   FML(4.9685),   FML(4.1017),
    };

    fml correctOutput_k57[] =
    {  FML(4.7839),   FML(6.1098),   FML(7.5909),   FML(7.5301),   FML(6.1205),   FML(4.8118),   FML(4.3707),
   FML(5.0217),   FML(7.1163),   FML(7.3291),   FML(7.3998),   FML(5.7427),   FML(5.7428),   FML(4.7422),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test96(const tTest& t)
{
    fml input[] =
      {  FML(0.632531),   FML(0.578811),   FML(0.793521),   FML(0.508687),   FML(0.229095),   FML(0.108050),   FML(0.391269),  FML(0.557728),   FML(0.931734),   FML(0.238129),   FML(0.140206),   FML(0.413744),   FML(0.861415),   FML(0.095246),  FML(0.266905),   FML(0.257546),
     FML(0.302880),   FML(0.082823),   FML(0.800166),   FML(0.151535),   FML(0.182306),   FML(0.934315),   FML(0.575160),  FML(0.815383),   FML(0.548455),   FML(0.888308),   FML(0.566752),   FML(0.876257),   FML(0.380451),   FML(0.951972),  FML(0.679041),   FML(0.621793),
    };
    u32 inputRows = 2;
    u32 inputCols = 8;

    fml correctOutput_k33[] =
    {  FML(1.9590),   FML(3.3920),   FML(4.2328),   FML(3.9237),   FML(4.4270),   FML(4.5909),   FML(3.9884),   FML(2.7589),
   FML(1.6379),   FML(2.3794),   FML(2.8563),   FML(2.7516),   FML(3.0132),   FML(3.3895),   FML(2.5767),   FML(2.6781),
    };

    fml correctOutput_k55[] =
    {  FML(2.5073),   FML(3.4880),   FML(4.8097),   FML(4.7201),   FML(4.8031),   FML(4.9655),   FML(4.2324),   FML(3.1119),
   FML(3.0538),   FML(4.8138),   FML(6.4479),   FML(6.4478),   FML(7.2484),   FML(6.9192),   FML(6.3393),   FML(4.9514),
    };

    fml correctOutput_k57[] =
    {  FML(5.4053),   FML(6.0544),   FML(6.6465),   FML(8.1625),   FML(8.0972),   FML(7.3002),   FML(6.6512),   FML(5.6614),
   FML(5.9179),   FML(6.7198),   FML(7.7367),   FML(8.8176),   FML(8.3759),   FML(6.9015),   FML(5.9539),   FML(4.8121),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test97(const tTest& t)
{
    fml input[] =
    {  FML(0.514722),   FML(0.594536),
   FML(0.726410),   FML(0.712631),
   FML(0.934200),   FML(0.070251),
    };
    u32 inputRows = 3;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(1.6711),
   FML(1.9833),
   FML(1.2792),
    };

    fml correctOutput_k55[] =
    {  FML(1.9124),
   FML(2.2836),
   FML(2.7022),
    };

    fml correctOutput_k57[] =
    {  FML(2.7262),
   FML(2.3924),
   FML(2.7991),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test98(const tTest& t)
{
    fml input[] =
    {  FML(0.184457),   FML(0.154941),   FML(0.183571),   FML(0.716152),
   FML(0.075461),   FML(0.704401),   FML(0.511775),   FML(0.983699),
   FML(0.312546),   FML(0.265790),   FML(0.768607),   FML(0.735128),
    };
    u32 inputRows = 3;
    u32 inputCols = 2;

    fml correctOutput_k33[] =
    {  FML(2.3907),   FML(2.3254),
   FML(2.9546),   FML(3.7128),
   FML(1.8944),   FML(2.4289),
    };

    fml correctOutput_k55[] =
    {  FML(2.9313),   FML(3.0367),
   FML(2.6216),   FML(3.7038),
   FML(2.5601),   FML(3.9481),
    };

    fml correctOutput_k57[] =
    {  FML(3.1950),   FML(4.0638),
   FML(3.1514),   FML(3.6878),
   FML(3.9709),   FML(4.0504),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test99(const tTest& t)
{
    fml input[] =
    {  FML(0.244078),   FML(0.847440),   FML(0.876934),   FML(0.129023),   FML(0.852256),   FML(0.883934),
   FML(0.557901),   FML(0.595810),   FML(0.388106),   FML(0.708172),   FML(0.374165),   FML(0.058585),
   FML(0.085595),   FML(0.452064),   FML(0.936831),   FML(0.625364),   FML(0.775679),   FML(0.888187),
    };
    u32 inputRows = 3;
    u32 inputCols = 3;

    fml correctOutput_k33[] =
    {  FML(2.7038),   FML(3.5403),   FML(2.2961),
   FML(3.5287),   FML(5.6698),   FML(5.3465),
   FML(1.8021),   FML(2.9265),   FML(2.5259),
    };

    fml correctOutput_k55[] =
    {  FML(4.3510),   FML(5.0987),   FML(4.6340),
   FML(4.9066),   FML(5.1719),   FML(6.7817),
   FML(4.5255),   FML(5.9678),   FML(6.2279),
    };

    fml correctOutput_k57[] =
    {  FML(5.4748),   FML(6.3757),   FML(6.5174),
   FML(5.4314),   FML(6.1722),   FML(5.3484),
   FML(6.1889),   FML(6.3833),   FML(6.5756),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test100(const tTest& t)
{
    fml input[] =
     {  FML(0.596922),   FML(0.806232),   FML(0.288842),   FML(0.535833),   FML(0.411525),   FML(0.100634),   FML(0.277817),  FML(0.744358),
    FML(0.537706),   FML(0.631130),   FML(0.087143),   FML(0.761519),   FML(0.724620),   FML(0.873743),   FML(0.966070),  FML(0.921897),
    FML(0.333603),   FML(0.998961),   FML(0.945220),   FML(0.893771),   FML(0.327651),   FML(0.957333),   FML(0.120677),  FML(0.478490),
    };
    u32 inputRows = 3;
    u32 inputCols = 4;

    fml correctOutput_k33[] =
    {  FML(2.3819),   FML(4.3904),   FML(4.3249),   FML(3.3376),
   FML(4.3925),   FML(6.3764),   FML(5.3908),   FML(4.1800),
   FML(2.0252),   FML(3.7991),   FML(3.8203),   FML(3.4548),
    };

    fml correctOutput_k55[] =
    {  FML(4.9313),   FML(6.5159),   FML(6.0898),   FML(4.5772),
   FML(4.9917),   FML(7.0923),   FML(8.0839),   FML(6.1088),
   FML(5.5169),   FML(7.7306),   FML(7.8120),   FML(7.4537),
    };

    fml correctOutput_k57[] =
    {  FML(7.5569),   FML(7.9550),   FML(7.7300),   FML(6.9968),
   FML(8.2070),   FML(7.8339),   FML(7.7241),   FML(8.2650),
   FML(8.9785),   FML(8.1535),   FML(8.5403),   FML(6.9425),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test101(const tTest& t)
{
    fml input[] =
     {  FML(0.833063),   FML(0.306343),   FML(0.683186),   FML(0.330075),   FML(0.481556),   FML(0.339392),   FML(0.870621),  FML(0.963626),   FML(0.090858),   FML(0.721020),
    FML(0.985941),   FML(0.966802),   FML(0.906676),   FML(0.520511),   FML(0.483582),   FML(0.398897),   FML(0.343013),  FML(0.831386),   FML(0.264292),   FML(0.065082),
    FML(0.972789),   FML(0.741876),   FML(0.243115),   FML(0.364019),   FML(0.635323),   FML(0.085948),   FML(0.996895),  FML(0.440040),   FML(0.530581),   FML(0.348226),
    };
    u32 inputRows = 3;
    u32 inputCols = 5;

    fml correctOutput_k33[] =
    {  FML(3.4726),   FML(4.4966),   FML(4.1417),   FML(3.1483),   FML(2.6633),
   FML(3.5410),   FML(5.8934),   FML(5.4033),   FML(5.6259),   FML(4.8241),
   FML(2.2131),   FML(4.0374),   FML(3.1392),   FML(2.7945),   FML(2.2300),
    };

    fml correctOutput_k55[] =
    {  FML(4.0891),   FML(5.6273),   FML(6.4797),   FML(5.3506),   FML(4.4173),
   FML(4.9352),   FML(7.9823),   FML(8.3048),   FML(6.8853),   FML(6.0768),
   FML(5.0801),   FML(7.5011),   FML(8.5701),   FML(6.5384),   FML(5.6079),
    };

    fml correctOutput_k57[] =
    {  FML(8.7656),   FML(9.8659),   FML(8.5713),   FML(9.7910),   FML(7.3248),
   FML(9.1642),   FML(9.6349),   FML(8.4934),   FML(8.2329),   FML(6.7960),
   FML(9.0035),   FML(9.6823),   FML(9.6586),   FML(8.7899),   FML(5.9733),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test102(const tTest& t)
{
    fml input[] =
     {  FML(0.021651),   FML(0.432130),   FML(0.872652),   FML(0.592465),   FML(0.699033),   FML(0.886539),   FML(0.696973),  FML(0.960095),   FML(0.498586),   FML(0.747683),   FML(0.844824),   FML(0.759833),
    FML(0.389873),   FML(0.335640),   FML(0.948393),   FML(0.563729),   FML(0.035069),   FML(0.705815),   FML(0.295636),  FML(0.900053),   FML(0.125046),   FML(0.240215),   FML(0.828965),   FML(0.216026),
    FML(0.983519),   FML(0.639943),   FML(0.409201),   FML(0.094877),   FML(0.884789),   FML(0.107154),   FML(0.348580),  FML(0.536285),   FML(0.427257),   FML(0.096935),   FML(0.976610),   FML(0.109048),
    };
    u32 inputRows = 3;
    u32 inputCols = 6;

    fml correctOutput_k33[] =
    {  FML(2.5877),   FML(3.4480),   FML(4.6122),   FML(3.5807),   FML(3.8652),   FML(2.4833),
   FML(3.2397),   FML(4.8584),   FML(6.0242),   FML(5.5934),   FML(5.6415),   FML(3.5757),
   FML(1.5627),   FML(3.2686),   FML(3.1878),   FML(2.6546),   FML(2.4738),   FML(1.4021),
    };

    fml correctOutput_k55[] =
    {  FML(3.9328),   FML(5.3457),   FML(6.3006),   FML(7.3341),   FML(6.0215),   FML(4.7536),
   FML(4.3296),   FML(6.5667),   FML(7.9607),   FML(9.8992),   FML(7.2772),   FML(6.3971),
   FML(3.4810),   FML(6.4514),   FML(7.3775),   FML(7.8766),   FML(6.5110),   FML(5.6762),
    };

    fml correctOutput_k57[] =
     {  FML(8.3906),    FML(8.8352),   FML(10.6108),   FML(10.9524),    FML(9.2625),    FML(7.8683),
    FML(8.4966),    FML(8.9485),   FML(10.6499),    FML(8.9157),    FML(9.1545),    FML(5.9738),
    FML(8.5515),    FML(9.5868),   FML(11.0576),   FML(10.4211),    FML(8.9959),    FML(6.5506),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test103(const tTest& t)
{
    fml input[] =
      {  FML(5.3742e-01),   FML(2.5216e-01),   FML(1.0298e-02),   FML(6.4674e-01),   FML(5.7720e-01),   FML(4.6947e-01),  FML(5.0059e-01),   FML(5.8141e-01),   FML(2.5832e-01),   FML(1.9246e-01),   FML(9.6363e-01),   FML(4.7877e-01),  FML(5.2672e-02),   FML(3.1172e-02),
     FML(9.4257e-01),   FML(3.3760e-01),   FML(3.9178e-02),   FML(6.5225e-01),   FML(5.8501e-01),   FML(9.8483e-01),  FML(7.4261e-02),   FML(7.0470e-01),   FML(2.7373e-01),   FML(7.8586e-05),   FML(3.7969e-02),   FML(1.9264e-01),  FML(9.7663e-01),   FML(7.5669e-01),
     FML(6.5894e-01),   FML(1.6392e-01),   FML(8.1401e-01),   FML(5.5930e-01),   FML(5.1895e-01),   FML(9.9771e-01),  FML(3.0976e-02),   FML(6.9590e-01),   FML(4.2133e-01),   FML(9.2631e-01),   FML(8.5903e-01),   FML(1.6938e-01),  FML(5.2130e-01),   FML(8.7420e-01),
    };
    u32 inputRows = 3;
    u32 inputCols = 7;

    fml correctOutput_k33[] =
    {  FML(2.0197),   FML(4.0868),   FML(3.4039),   FML(3.1593),   FML(2.2288),   FML(2.8221),   FML(2.6933),
   FML(2.9515),   FML(6.0301),   FML(5.1290),   FML(5.8044),   FML(4.4887),   FML(4.5155),   FML(3.9216),
   FML(1.5081),   FML(3.5181),   FML(3.2871),   FML(3.4468),   FML(1.9704),   FML(2.5581),   FML(2.3023),
    };

    fml correctOutput_k55[] =
    {  FML(3.9212),   FML(5.4302),   FML(6.0643),   FML(6.4581),   FML(6.2798),   FML(4.7286),   FML(3.3819),
   FML(4.4630),   FML(5.8002),   FML(7.8989),   FML(8.2390),   FML(7.8866),   FML(5.8692),   FML(5.1489),
   FML(4.7151),   FML(5.4606),   FML(8.0058),   FML(7.6467),   FML(8.5002),   FML(5.5660),   FML(5.0562),
    };

    fml correctOutput_k57[] =
     {  FML(5.9076),    FML(7.4994),    FML(8.9467),   FML(10.8689),    FML(9.3697),    FML(8.2326),    FML(6.5377),
    FML(7.1778),    FML(8.2326),    FML(9.6514),   FML(10.8073),    FML(9.2218),    FML(7.2789),    FML(5.8412),
    FML(7.9871),    FML(8.8439),   FML(10.3989),   FML(11.4341),    FML(8.9577),    FML(7.7080),    FML(4.5837),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test104(const tTest& t)
{
    fml input[] =
      {  FML(0.0209467),   FML(0.8007943),   FML(0.1063376),   FML(0.6484605),   FML(0.4175776),   FML(0.1462544),  FML(0.1884219),   FML(0.5971049),   FML(0.3504967),   FML(0.3380008),   FML(0.0055922),   FML(0.9624966),  FML(0.4610141),   FML(0.4516228),   FML(0.4427689),   FML(0.7188037),
     FML(0.6152650),   FML(0.2558498),   FML(0.6809506),   FML(0.8936265),   FML(0.8588630),   FML(0.9345067),  FML(0.7906945),   FML(0.6119106),   FML(0.6652394),   FML(0.0220821),   FML(0.9204425),   FML(0.5132386),  FML(0.6193225),   FML(0.0448396),   FML(0.1976331),   FML(0.8244733),
     FML(0.3707193),   FML(0.4992769),   FML(0.8704576),   FML(0.9748285),   FML(0.4090757),   FML(0.4517590),  FML(0.8643504),   FML(0.2591758),   FML(0.6965646),   FML(0.7119745),   FML(0.4537312),   FML(0.0724149),  FML(0.3952974),   FML(0.0036666),   FML(0.7211463),   FML(0.6116959),
    };
    u32 inputRows = 3;
    u32 inputCols = 8;

    fml correctOutput_k33[] =
    {  FML(2.5840),   FML(4.6219),   FML(4.5708),   FML(3.7918),   FML(3.6922),   FML(3.4367),   FML(3.7282),   FML(2.5922),
   FML(4.0287),   FML(5.3677),   FML(5.7161),   FML(5.8337),   FML(5.2658),   FML(4.5082),   FML(4.5751),   FML(3.7348),
   FML(1.6673),   FML(3.6527),   FML(4.4737),   FML(3.4062),   FML(3.0091),   FML(2.8711),   FML(2.5870),   FML(2.2824),
    };

    fml correctOutput_k55[] =
    {  FML(4.5946),   FML(6.0776),   FML(6.3983),   FML(6.7721),   FML(5.8756),   FML(5.7107),   FML(5.0824),   FML(4.3310),
   FML(4.0793),   FML(6.0092),   FML(7.9749),   FML(7.5663),   FML(7.3775),   FML(6.8775),   FML(5.5501),   FML(5.1580),
   FML(4.8257),   FML(7.1239),   FML(9.0611),   FML(9.3806),   FML(7.8875),   FML(7.5649),   FML(6.8796),   FML(5.2397),
    };

    fml correctOutput_k57[] =
     {  FML(6.8775),    FML(9.0122),   FML(10.2763),   FML(10.8743),   FML(11.8298),    FML(9.2643),    FML(7.3688),    FML(6.2417),
    FML(8.0433),    FML(9.2067),   FML(10.6793),   FML(11.0089),   FML(12.0952),    FML(9.7724),    FML(8.0084),    FML(6.1000),
    FML(7.6290),    FML(9.0944),   FML(11.9545),   FML(11.6127),   FML(12.0951),    FML(8.9131),    FML(7.2368),    FML(6.1485),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test105(const tTest& t)
{
    fml input[] =
    {  FML(0.706489),   FML(0.537708),
   FML(0.384041),   FML(0.991230),
   FML(0.568498),   FML(0.497086),
   FML(0.072526),   FML(0.593238),
    };
    u32 inputRows = 4;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(1.6872),
   FML(2.0952),
   FML(2.1013),
   FML(1.0639),
    };

    fml correctOutput_k55[] =
    {  FML(2.1953),
   FML(3.0922),
   FML(3.0817),
   FML(2.7573),
    };

    fml correctOutput_k57[] =
    {  FML(2.9689),
   FML(2.8163),
   FML(3.2633),
   FML(2.2165),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test106(const tTest& t)
{
    fml input[] =
    {  FML(0.1865802),   FML(0.6599003),   FML(0.1358030),   FML(0.6452412),
   FML(0.7743710),   FML(0.9292930),   FML(0.9673460),   FML(0.1839769),
   FML(0.2969881),   FML(0.6744408),   FML(0.5189804),   FML(0.2023802),
   FML(0.1265907),   FML(0.9423454),   FML(0.2062258),   FML(0.0055982),
    };
    u32 inputRows = 4;
    u32 inputCols = 2;

    fml correctOutput_k33[] =
    {  FML(2.9011),   FML(2.9951),
   FML(3.3607),   FML(3.8365),
   FML(3.0240),   FML(3.2478),
   FML(1.5310),   FML(1.9543),
    };

    fml correctOutput_k55[] =
    {  FML(3.0337),   FML(3.3299),
   FML(3.5975),   FML(4.1237),
   FML(3.9713),   FML(5.7612),
   FML(3.5153),   FML(4.9257),
    };

    fml correctOutput_k57[] =
    {  FML(3.7713),   FML(3.8277),
   FML(5.1121),   FML(5.4487),
   FML(4.8885),   FML(4.5968),
   FML(3.8594),   FML(3.6764),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test107(const tTest& t)
{
    fml input[] =
    {  FML(0.771371),   FML(0.735055),   FML(0.674328),   FML(0.457090),   FML(0.674893),   FML(0.269633),
   FML(0.084458),   FML(0.770260),   FML(0.894941),   FML(0.156496),   FML(0.559609),   FML(0.960067),
   FML(0.296853),   FML(0.707428),   FML(0.182838),   FML(0.037996),   FML(0.625259),   FML(0.201136),
   FML(0.137421),   FML(0.132873),   FML(0.124527),   FML(0.102769),   FML(0.012645),   FML(0.193921),
    };
    u32 inputRows = 4;
    u32 inputCols = 3;

    fml correctOutput_k33[] =
    {  FML(2.4997),   FML(4.2249),   FML(3.2785),
   FML(2.9474),   FML(4.5578),   FML(3.3255),
   FML(1.9765),   FML(2.9646),   FML(2.8546),
   FML(1.1843),   FML(1.3937),   FML(1.0074),
    };

    fml correctOutput_k55[] =
    {  FML(4.0161),   FML(3.7506),   FML(4.5585),
   FML(5.1940),   FML(5.7153),   FML(6.2532),
   FML(4.4753),   FML(5.6943),   FML(5.8644),
   FML(3.2113),   FML(4.0091),   FML(4.0347),
    };

    fml correctOutput_k57[] =
    {  FML(5.6032),   FML(5.7564),   FML(5.0656),
   FML(5.5725),   FML(6.3147),   FML(5.4630),
   FML(5.6285),   FML(5.4587),   FML(5.6431),
   FML(4.2514),   FML(4.2657),   FML(3.5623),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test108(const tTest& t)
{
    fml input[] =
     {  FML(0.956654),   FML(0.897534),   FML(0.165897),   FML(0.314759),   FML(0.737485),   FML(0.807657),   FML(0.082322),  FML(0.769937),
    FML(0.425549),   FML(0.968845),   FML(0.241633),   FML(0.617565),   FML(0.207162),   FML(0.844235),   FML(0.922484),  FML(0.362524),
    FML(0.518131),   FML(0.692292),   FML(0.494692),   FML(0.795207),   FML(0.329510),   FML(0.943718),   FML(0.984547),  FML(0.879694),
    FML(0.321762),   FML(0.515789),   FML(0.204467),   FML(0.178767),   FML(0.872613),   FML(0.548181),   FML(0.288836),  FML(0.384121),
    };
    u32 inputRows = 4;
    u32 inputCols = 4;

    fml correctOutput_k33[] =
    {  FML(2.6350),   FML(4.3017),   FML(3.4783),   FML(3.0671),
   FML(3.9523),   FML(6.4852),   FML(6.3655),   FML(5.1919),
   FML(3.0536),   FML(5.4109),   FML(5.3786),   FML(4.1696),
   FML(1.7364),   FML(3.3423),   FML(3.2580),   FML(3.1169),
    };

    fml correctOutput_k55[] =
    {  FML(4.7642),   FML(6.7050),   FML(7.0362),   FML(4.7944),
   FML(6.6527),   FML(8.8577),   FML(9.2191),   FML(7.0699),
   FML(6.0277),   FML(8.5390),   FML(8.8134),   FML(8.1879),
   FML(4.3906),   FML(7.1626),   FML(7.0269),   FML(6.9996),
    };

    fml correctOutput_k57[] =
     {  FML(8.4569),    FML(8.3583),    FML(7.7941),    FML(8.6994),
    FML(9.9893),    FML(8.9297),    FML(9.5670),   FML(10.0683),
   FML(10.7776),    FML(9.6151),   FML(10.8354),    FML(8.0561),
    FML(8.7832),    FML(7.3696),    FML(8.0419),    FML(7.0631),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test109(const tTest& t)
{
    fml input[] =
     {  FML(9.1930e-01),   FML(2.0594e-01),   FML(3.1550e-01),   FML(8.9321e-04),   FML(8.4468e-01),   FML(9.1158e-01),  FML(4.6704e-01),   FML(7.6685e-01),   FML(4.6590e-01),   FML(4.9785e-01),
    FML(6.1510e-01),   FML(3.3561e-01),   FML(2.5159e-01),   FML(1.7229e-01),   FML(1.5491e-01),   FML(8.5495e-01),  FML(8.8427e-01),   FML(9.9535e-01),   FML(6.7350e-01),   FML(1.8392e-01),
    FML(2.4307e-01),   FML(7.2177e-01),   FML(6.3916e-01),   FML(6.6984e-01),   FML(1.1944e-01),   FML(5.5212e-01),  FML(2.2099e-01),   FML(6.6756e-01),   FML(2.4941e-01),   FML(3.5271e-01),
    FML(2.2667e-01),   FML(4.1609e-01),   FML(2.0345e-01),   FML(2.4335e-02),   FML(8.4808e-01),   FML(5.4365e-01),  FML(1.6433e-01),   FML(9.6599e-01),   FML(4.9200e-01),   FML(3.7661e-01),
    };
    u32 inputRows = 4;
    u32 inputCols = 5;

    fml correctOutput_k33[] =
    {  FML(1.6921),   FML(3.2443),   FML(3.8092),   FML(4.6368),   FML(3.1798),
   FML(2.8708),   FML(4.7191),   FML(5.0170),   FML(5.5185),   FML(4.0035),
   FML(2.0308),   FML(4.3682),   FML(5.3382),   FML(5.5069),   FML(3.8306),
   FML(1.6394),   FML(2.6201),   FML(2.6049),   FML(2.9246),   FML(2.1755),
    };

    fml correctOutput_k55[] =
    {  FML(3.7692),   FML(5.1500),   FML(6.0708),   FML(5.1806),   FML(5.3094),
   FML(5.3897),   FML(7.1502),   FML(9.7389),   FML(8.4873),   FML(7.5114),
   FML(5.1324),   FML(7.1183),   FML(8.7514),   FML(8.6998),   FML(7.9068),
   FML(3.4465),   FML(5.4086),   FML(6.4185),   FML(6.7872),   FML(6.2378),
    };

    fml correctOutput_k57[] =
     {  FML(7.6269),    FML(8.2543),    FML(7.6361),    FML(8.4242),    FML(6.6145),
    FML(9.1398),    FML(9.7544),    FML(9.1220),   FML(11.5155),    FML(8.8519),
   FML(10.0726),   FML(11.1466),   FML(10.3008),    FML(8.8388),    FML(8.4626),
    FML(8.0881),    FML(9.1031),    FML(7.6789),    FML(7.1985),    FML(5.9452),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test110(const tTest& t)
{
    fml input[] =
     {  FML(0.3270242),   FML(0.5715860),   FML(0.2376162),   FML(0.1303724),   FML(0.0605080),   FML(0.4183847),  FML(0.8439878),   FML(0.3839273),   FML(0.3822773),   FML(0.0643853),   FML(0.7854767),   FML(0.5055855),
    FML(0.9374918),   FML(0.2860440),   FML(0.1571638),   FML(0.5372054),   FML(0.3511241),   FML(0.8168550),  FML(0.1718557),   FML(0.8984396),   FML(0.5597990),   FML(0.0478648),   FML(0.7729720),   FML(0.0901948),
    FML(0.6350966),   FML(0.6644444),   FML(0.6054246),   FML(0.9008932),   FML(0.1359590),   FML(0.3200278),  FML(0.8893486),   FML(0.4930846),   FML(0.3602062),   FML(0.1120300),   FML(0.6843359),   FML(0.5725919),
    FML(0.1927189),   FML(0.4376462),   FML(0.9423379),   FML(0.1135282),   FML(0.4690608),   FML(0.9935981),  FML(0.6636832),   FML(0.0082173),   FML(0.5385436),   FML(0.1054736),   FML(0.7246833),   FML(0.1875337),
    };
    u32 inputRows = 4;
    u32 inputCols = 6;

    fml correctOutput_k33[] =
    {  FML(2.0861),   FML(3.5177),   FML(3.1748),   FML(3.0738),   FML(3.1650),   FML(1.9835),
   FML(3.5725),   FML(4.7218),   FML(4.6066),   FML(4.0890),   FML(5.2236),   FML(3.1478),
   FML(2.9142),   FML(5.8680),   FML(5.9545),   FML(4.5422),   FML(4.2063),   FML(2.8200),
   FML(1.9825),   FML(3.1270),   FML(3.1975),   FML(2.5884),   FML(2.6619),   FML(1.9411),
    };

    fml correctOutput_k55[] =
    {  FML(3.9203),   FML(5.2068),   FML(5.3649),   FML(5.2833),   FML(4.5442),   FML(3.8601),
   FML(4.4121),   FML(7.8767),   FML(8.2351),   FML(8.6313),   FML(7.0563),   FML(5.6818),
   FML(5.4526),   FML(6.8186),   FML(8.9958),   FML(8.8158),   FML(6.9673),   FML(6.0246),
   FML(4.3588),   FML(6.4437),   FML(7.2648),   FML(8.0769),   FML(6.1927),   FML(4.9037),
    };

    fml correctOutput_k57[] =
     {  FML(6.4175),    FML(7.3385),    FML(8.6140),    FML(8.7669),    FML(8.2987),    FML(5.9722),
    FML(9.5186),    FML(9.7465),   FML(12.7013),   FML(10.8106),   FML(10.0836),    FML(7.3284),
    FML(9.3868),    FML(9.1700),   FML(12.7102),   FML(11.5838),    FML(9.2290),    FML(7.3326),
    FML(8.5851),    FML(8.2862),   FML(11.0880),    FML(8.0674),    FML(7.4986),    FML(6.0634),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test111(const tTest& t)
{
    fml input[] =
     {  FML(0.925956),   FML(0.273450),   FML(0.381328),   FML(0.989999),   FML(0.972582),   FML(0.730275),   FML(0.716721),  FML(0.412263),   FML(0.211265),   FML(0.470360),   FML(0.750153),   FML(0.879279),   FML(0.766945),   FML(0.566498),
    FML(0.495115),   FML(0.629103),   FML(0.607816),   FML(0.042378),   FML(0.319137),   FML(0.245828),   FML(0.768096),  FML(0.370093),   FML(0.847323),   FML(0.822045),   FML(0.226162),   FML(0.379119),   FML(0.776226),   FML(0.491566),
    FML(0.869348),   FML(0.075230),   FML(0.242311),   FML(0.196559),   FML(0.026273),   FML(0.340395),   FML(0.340810),  FML(0.952268),   FML(0.503253),   FML(0.121459),   FML(0.158459),   FML(0.322027),   FML(0.760946),   FML(0.549307),
    FML(0.439038),   FML(0.455224),   FML(0.271773),   FML(0.180532),   FML(0.768407),   FML(0.397054),   FML(0.836965),  FML(0.260508),   FML(0.456993),   FML(0.486925),   FML(0.671956),   FML(0.993191),   FML(0.941421),   FML(0.599227),
    };
    u32 inputRows = 4;
    u32 inputCols = 7;

    fml correctOutput_k33[] =
    {  FML(2.13976),   FML(3.41642),   FML(3.82134),   FML(4.39785),   FML(4.11341),   FML(4.06929),   FML(3.00576),
   FML(2.67869),   FML(5.16261),   FML(4.71028),   FML(4.93641),   FML(4.89365),   FML(5.52921),   FML(3.99993),
   FML(2.41658),   FML(3.95368),   FML(4.11410),   FML(4.84496),   FML(6.33850),   FML(6.17829),   FML(4.04149),
   FML(0.94779),   FML(2.51228),   FML(2.23074),   FML(2.71790),   FML(2.59382),   FML(2.89322),   FML(2.70667),
    };

    fml correctOutput_k55[] =
     {  FML(3.3033),    FML(4.7919),    FML(7.0176),    FML(5.9745),    FML(6.7643),    FML(5.8121),    FML(4.4931),
    FML(5.3363),    FML(7.7598),    FML(9.1132),    FML(9.6305),   FML(10.5077),    FML(9.1029),    FML(7.4339),
    FML(4.5313),    FML(7.0551),    FML(9.4089),    FML(8.7743),    FML(9.9498),    FML(8.9145),    FML(7.9290),
    FML(3.6217),    FML(5.3036),    FML(5.8241),    FML(5.9760),    FML(7.9404),    FML(6.8634),    FML(6.3567),
    };

    fml correctOutput_k57[] =
     {  FML(6.7163),    FML(8.5940),    FML(9.5210),   FML(12.2112),    FML(8.6325),    FML(8.4805),    FML(7.6214),
    FML(8.6553),   FML(10.9705),   FML(12.9507),   FML(15.5756),   FML(12.2465),   FML(11.8405),   FML(10.3241),
   FML(10.7015),   FML(10.9342),   FML(12.8307),   FML(15.6503),   FML(11.9208),   FML(11.1075),    FML(8.3312),
    FML(6.3142),    FML(7.6799),   FML(10.2764),   FML(12.2642),    FML(9.0054),    FML(8.3701),    FML(7.5482),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test112(const tTest& t)
{
    fml input[] =
      {  FML(0.0856126),   FML(0.0293341),   FML(0.8068558),   FML(0.3324714),   FML(0.0229737),   FML(0.1743675),  FML(0.5451600),   FML(0.4384099),   FML(0.2714367),   FML(0.2515561),   FML(0.0703564),   FML(0.6125732),  FML(0.6403723),   FML(0.6772501),   FML(0.3467126),   FML(0.5884870),
     FML(0.3513141),   FML(0.9067309),   FML(0.0284723),   FML(0.8322730),   FML(0.3436426),   FML(0.8540936),  FML(0.6711253),   FML(0.7335019),   FML(0.8113735),   FML(0.8265454),   FML(0.9202546),   FML(0.0094973),  FML(0.8124277),   FML(0.6321086),   FML(0.7132039),   FML(0.7133686),
     FML(0.6881798),   FML(0.4865801),   FML(0.6014436),   FML(0.5793208),   FML(0.5032916),   FML(0.7510384),  FML(0.4805133),   FML(0.0870436),   FML(0.6199893),   FML(0.5703164),   FML(0.6626114),   FML(0.2120633),  FML(0.6085161),   FML(0.1625250),   FML(0.5398803),   FML(0.8787472),
     FML(0.3099566),   FML(0.2514015),   FML(0.2592915),   FML(0.2037610),   FML(0.5715975),   FML(0.3442427),  FML(0.1114172),   FML(0.0878262),   FML(0.1198391),   FML(0.5985110),   FML(0.8699432),   FML(0.1817516),  FML(0.6854983),   FML(0.5851034),   FML(0.5202987),   FML(0.7225293),
    };
    u32 inputRows = 4;
    u32 inputCols = 8;

    fml correctOutput_k33[] =
    {  FML(2.3429),   FML(3.0540),   FML(3.8275),   FML(3.8908),   FML(4.0674),   FML(4.0040),   FML(4.3291),   FML(3.4848),
   FML(2.8666),   FML(4.9496),   FML(5.0551),   FML(4.6731),   FML(5.2791),   FML(5.2543),   FML(5.2514),   FML(4.8097),
   FML(2.8687),   FML(4.9764),   FML(4.3771),   FML(4.8498),   FML(4.7880),   FML(5.3285),   FML(6.0812),   FML(4.7869),
   FML(1.4358),   FML(2.8872),   FML(2.5793),   FML(2.3210),   FML(2.2579),   FML(2.5694),   FML(2.8923),   FML(3.0244),
    };

    fml correctOutput_k55[] =
    {  FML(3.7676),   FML(5.0043),   FML(5.6321),   FML(6.0960),   FML(6.0843),   FML(6.2638),   FML(5.6894),   FML(4.5309),
   FML(4.3760),   FML(7.0320),   FML(8.5018),   FML(8.6238),   FML(8.9261),   FML(9.0240),   FML(7.9145),   FML(6.7785),
   FML(4.5078),   FML(6.4769),   FML(8.5658),   FML(9.1797),   FML(9.4135),   FML(9.6935),   FML(9.0312),   FML(7.8853),
   FML(4.1665),   FML(5.8204),   FML(7.6583),   FML(7.6086),   FML(7.8860),   FML(7.9780),   FML(6.7131),   FML(6.7295),
    };

    fml correctOutput_k57[] =
     {  FML(6.6890),    FML(8.5158),   FML(10.0236),   FML(11.5506),   FML(12.2657),    FML(9.5609),    FML(8.3526),    FML(7.5628),
    FML(7.9797),    FML(9.1145),   FML(11.1844),   FML(13.3610),   FML(14.3276),   FML(12.7181),   FML(10.5292),    FML(9.5980),
    FML(7.9121),   FML(11.0259),   FML(11.8569),   FML(14.6298),   FML(15.0015),   FML(12.6269),    FML(9.4126),    FML(8.6320),
    FML(6.9841),    FML(9.2303),   FML(10.9338),   FML(10.8282),   FML(12.2338),   FML(10.2539),    FML(8.4768),    FML(6.6529),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test113(const tTest& t)
{
    fml input[] =
    {  FML(0.11896),   FML(0.62539),
   FML(0.71183),   FML(0.29090),
   FML(0.33289),   FML(0.34253),
   FML(0.27703),   FML(0.15386),
   FML(0.92808),   FML(0.76141),
    };
    u32 inputRows = 5;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(1.25143),
   FML(1.64439),
   FML(1.17799),
   FML(2.05506),
   FML(0.88486),
    };

    fml correctOutput_k55[] =
    {  FML(1.7320),
   FML(1.8536),
   FML(3.0530),
   FML(2.5577),
   FML(2.1991),
    };

    fml correctOutput_k57[] =
    {  FML(1.9753),
   FML(2.5180),
   FML(3.1647),
   FML(2.6261),
   FML(2.4310),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test114(const tTest& t)
{
    fml input[] =
    {  FML(0.648863),   FML(0.397707),   FML(0.222533),   FML(0.436246),
   FML(0.225811),   FML(0.352880),   FML(0.476911),   FML(0.846264),
   FML(0.101727),   FML(0.071190),   FML(0.723540),   FML(0.381456),
   FML(0.266933),   FML(0.117902),   FML(0.283381),   FML(0.147933),
   FML(0.217553),   FML(0.882054),   FML(0.695795),   FML(0.450244),
    };
    u32 inputRows = 5;
    u32 inputCols = 2;

    fml correctOutput_k33[] =
    {  FML(2.2003),   FML(2.5522),
   FML(2.4090),   FML(3.2036),
   FML(2.0460),   FML(2.4999),
   FML(2.7050),   FML(2.6737),
   FML(1.1211),   FML(1.8232),
    };

    fml correctOutput_k55[] =
    {  FML(2.4282),   FML(2.7021),
   FML(3.0229),   FML(3.7726),
   FML(3.9502),   FML(4.3909),
   FML(2.5711),   FML(3.9698),
   FML(2.5328),   FML(3.5876),
    };

    fml correctOutput_k57[] =
    {  FML(3.0779),   FML(3.6945),
   FML(3.4340),   FML(3.9411),
   FML(5.1090),   FML(4.6637),
   FML(4.2177),   FML(3.5673),
   FML(3.0939),   FML(3.4982),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test115(const tTest& t)
{
    fml input[] =
    {  FML(0.263477),   FML(0.310587),   FML(0.887372),   FML(0.153209),   FML(0.850808),   FML(0.707712),
   FML(0.396489),   FML(0.389158),   FML(0.375203),   FML(0.975406),   FML(0.406133),   FML(0.125681),
   FML(0.881633),   FML(0.610522),   FML(0.520562),   FML(0.305148),   FML(0.030096),   FML(0.210489),
   FML(0.460702),   FML(0.866215),   FML(0.760893),   FML(0.216962),   FML(0.016081),   FML(0.208669),
   FML(0.181705),   FML(0.958851),   FML(0.935707),   FML(0.346467),   FML(0.608418),   FML(0.899269),
    };
    u32 inputRows = 5;
    u32 inputCols = 3;

    fml correctOutput_k33[] =
    {  FML(2.4833),   FML(3.2389),   FML(2.4488),
   FML(3.0103),   FML(4.2502),   FML(3.8400),
   FML(3.5840),   FML(4.7894),   FML(2.8185),
   FML(3.8405),   FML(5.7529),   FML(3.9606),
   FML(2.0486),   FML(2.8392),   FML(2.5297),
    };

    fml correctOutput_k55[] =
    {  FML(3.4869),   FML(3.7082),   FML(3.4456),
   FML(5.1213),   FML(5.5698),   FML(6.5066),
   FML(5.7788),   FML(7.1422),   FML(8.1252),
   FML(5.2052),   FML(6.3454),   FML(7.2470),
   FML(4.5874),   FML(5.9621),   FML(5.3669),
    };

    fml correctOutput_k57[] =
    {  FML(4.9239),   FML(5.7052),   FML(5.0425),
   FML(5.7946),   FML(6.1995),   FML(6.2055),
   FML(8.3192),   FML(8.5071),   FML(8.4709),
   FML(6.8045),   FML(6.3982),   FML(5.9136),
   FML(5.9850),   FML(5.3868),   FML(5.5330),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test116(const tTest& t)
{
    fml input[] =
     {  FML(0.5385639),   FML(0.5823827),   FML(0.2626679),   FML(0.8096628),   FML(0.2150414),   FML(0.9544872),  FML(0.1966575),   FML(0.8936348),
    FML(0.9040106),   FML(0.6121465),   FML(0.7771423),   FML(0.0553189),   FML(0.6905963),   FML(0.0029182),  FML(0.6178406),   FML(0.7928248),
    FML(0.0713449),   FML(0.9474446),   FML(0.2721108),   FML(0.9211294),   FML(0.8463957),   FML(0.9632431),  FML(0.5558855),   FML(0.0727049),
    FML(0.4667072),   FML(0.8216621),   FML(0.5496455),   FML(0.0871814),   FML(0.7656391),   FML(0.9381845),  FML(0.7318972),   FML(0.5507533),
    FML(0.1153523),   FML(0.8132295),   FML(0.0605117),   FML(0.1288448),   FML(0.9268616),   FML(0.3669989),  FML(0.4031061),   FML(0.4890101),
    };
    u32 inputRows = 5;
    u32 inputCols = 4;

    fml correctOutput_k33[] =
    {  FML(2.5379),   FML(3.6702),   FML(3.8386),   FML(3.1139),
   FML(3.7575),   FML(6.6307),   FML(5.6669),   FML(4.1496),
   FML(3.1493),   FML(5.7183),   FML(5.9506),   FML(5.1681),
   FML(3.0674),   FML(5.2113),   FML(4.9964),   FML(4.7611),
   FML(1.5719),   FML(2.8109),   FML(2.7673),   FML(3.1326),
    };

    fml correctOutput_k55[] =
     {  FML(4.6794),    FML(6.5087),    FML(6.0715),    FML(4.8195),
    FML(5.8381),    FML(7.8089),    FML(8.5410),    FML(6.6684),
    FML(7.6025),   FML(10.3211),   FML(10.6193),    FML(8.7163),
    FML(6.1676),    FML(7.7391),    FML(8.0649),    FML(7.7069),
    FML(4.5377),    FML(7.1879),    FML(7.3870),    FML(6.3490),
    };

    fml correctOutput_k57[] =
     {  FML(7.8176),    FML(7.2994),    FML(6.9534),    FML(6.9906),
    FML(9.8784),   FML(10.2398),   FML(10.4193),   FML(10.5559),
   FML(12.7371),   FML(10.5933),   FML(11.0744),   FML(10.6345),
    FML(9.8152),   FML(10.0923),    FML(9.9189),    FML(8.8396),
    FML(7.9722),    FML(7.2290),    FML(6.9591),    FML(6.5721),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test117(const tTest& t)
{
    fml input[] =
     {  FML(0.474869),   FML(0.871089),   FML(0.239824),   FML(0.926292),   FML(0.563384),   FML(0.258999),   FML(0.753080),  FML(0.885487),   FML(0.502743),   FML(0.956036),
    FML(0.837615),   FML(0.640469),   FML(0.763886),   FML(0.179292),   FML(0.101918),   FML(0.451610),   FML(0.775936),  FML(0.326155),   FML(0.122623),   FML(0.422295),
    FML(0.011509),   FML(0.705117),   FML(0.278237),   FML(0.863465),   FML(0.655849),   FML(0.875720),   FML(0.172217),  FML(0.620650),   FML(0.455275),   FML(0.023198),
    FML(0.631826),   FML(0.784589),   FML(0.435898),   FML(0.316891),   FML(0.566331),   FML(0.631564),   FML(0.169927),  FML(0.073240),   FML(0.814461),   FML(0.096478),
    FML(0.066790),   FML(0.987638),   FML(0.686732),   FML(0.217331),   FML(0.745173),   FML(0.957861),   FML(0.273404),  FML(0.408269),   FML(0.761272),   FML(0.375613),
    };
    u32 inputRows = 5;
    u32 inputCols = 5;

    fml correctOutput_k33[] =
    {  FML(2.6818),   FML(3.8756),   FML(3.7035),   FML(2.9883),   FML(3.0309),
   FML(3.7761),   FML(5.9764),   FML(5.2326),   FML(5.0407),   FML(3.9972),
   FML(3.3096),   FML(5.3983),   FML(4.6915),   FML(4.3335),   FML(3.0601),
   FML(3.5159),   FML(5.8499),   FML(5.5673),   FML(5.6737),   FML(2.4045),
   FML(1.8917),   FML(3.1866),   FML(2.7054),   FML(2.9715),   FML(1.4292),
    };

    fml correctOutput_k55[] =
     {  FML(4.3420),    FML(6.3122),    FML(7.3257),    FML(5.0780),    FML(4.4461),
    FML(5.9527),    FML(7.9817),    FML(9.5956),    FML(7.6490),    FML(6.3188),
    FML(7.1258),   FML(10.5426),   FML(11.4649),    FML(9.2723),    FML(7.5013),
    FML(5.9964),    FML(6.8891),    FML(9.1077),    FML(7.6172),    FML(6.0947),
    FML(4.7475),    FML(6.7044),    FML(8.3951),    FML(6.7112),    FML(5.7863),
    };

    fml correctOutput_k57[] =
     {  FML(7.0073),    FML(8.3696),    FML(8.8481),    FML(8.1776),    FML(6.2575),
    FML(9.8916),   FML(12.3666),   FML(10.6043),   FML(10.5506),    FML(8.7593),
   FML(11.7126),   FML(13.3088),   FML(13.5778),   FML(12.6677),    FML(9.1876),
    FML(9.6980),   FML(10.8571),    FML(9.6289),   FML(10.9049),    FML(6.8075),
    FML(7.5050),    FML(9.3957),    FML(7.7503),    FML(7.7927),    FML(5.9356),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test118(const tTest& t)
{
    fml input[] =
     {  FML(0.538216),   FML(0.018055),   FML(0.675858),   FML(0.135353),   FML(0.999027),   FML(0.919180),   FML(0.649462),  FML(0.226272),   FML(0.626433),   FML(0.858231),   FML(0.749408),   FML(0.619864),
    FML(0.993662),   FML(0.933822),   FML(0.351001),   FML(0.186230),   FML(0.367204),   FML(0.143920),   FML(0.750107),  FML(0.993490),   FML(0.449775),   FML(0.122833),   FML(0.846137),   FML(0.455540),
    FML(0.289160),   FML(0.383394),   FML(0.751842),   FML(0.456418),   FML(0.452107),   FML(0.247715),   FML(0.232606),  FML(0.748882),   FML(0.353766),   FML(0.654359),   FML(0.384711),   FML(0.629410),
    FML(0.370760),   FML(0.755881),   FML(0.792514),   FML(0.607008),   FML(0.168413),   FML(0.425960),   FML(0.318071),  FML(0.654177),   FML(0.132745),   FML(0.624219),   FML(0.731698),   FML(0.640660),
    FML(0.382564),   FML(0.261174),   FML(0.863512),   FML(0.956692),   FML(0.774778),   FML(0.023150),   FML(0.458988),  FML(0.813843),   FML(0.166734),   FML(0.668686),   FML(0.454625),   FML(0.545343),
    };
    u32 inputRows = 5;
    u32 inputCols = 6;

    fml correctOutput_k33[] =
    {  FML(2.5535),   FML(3.2666),   FML(3.7144),   FML(4.2236),   FML(4.0920),   FML(3.0267),
   FML(2.6806),   FML(5.0270),   FML(5.1616),   FML(5.4559),   FML(6.0604),   FML(3.9978),
   FML(3.9806),   FML(5.0712),   FML(4.7327),   FML(4.4944),   FML(5.3748),   FML(3.7143),
   FML(3.6141),   FML(5.1416),   FML(5.8605),   FML(5.0215),   FML(5.2532),   FML(3.6248),
   FML(1.9287),   FML(2.9962),   FML(3.6248),   FML(2.5050),   FML(3.1561),   FML(2.2511),
    };

    fml correctOutput_k55[] =
     {  FML(3.6807),    FML(4.8867),    FML(5.8015),    FML(6.6956),    FML(6.0621),    FML(4.8912),
    FML(5.8783),    FML(7.5532),   FML(10.0917),   FML(10.2414),    FML(8.8577),    FML(7.7566),
    FML(6.6631),    FML(9.3806),   FML(11.0785),   FML(11.7082),    FML(9.8814),    FML(9.1078),
    FML(5.3664),    FML(7.8122),    FML(8.8966),    FML(9.8566),    FML(7.7995),    FML(7.1971),
    FML(3.8167),    FML(6.5172),    FML(7.3349),    FML(7.8453),    FML(6.3947),    FML(6.3892),
    };

    fml correctOutput_k57[] =
     {  FML(7.1033),    FML(8.4456),   FML(10.7012),   FML(10.1070),    FML(8.6352),    FML(7.7336),
    FML(9.0367),   FML(11.7144),   FML(12.3382),   FML(12.3287),   FML(11.6713),    FML(8.7735),
   FML(12.6530),   FML(14.6056),   FML(17.5508),   FML(14.8516),   FML(14.2372),   FML(10.0018),
   FML(10.3189),   FML(11.2853),   FML(14.2019),   FML(11.0221),   FML(10.3063),    FML(8.6102),
    FML(7.4658),    FML(9.7166),   FML(11.5842),    FML(9.1909),    FML(8.1566),    FML(6.0298),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test119(const tTest& t)
{
    fml input[] =
     {  FML(0.018073),   FML(0.855188),   FML(0.201590),   FML(0.415980),   FML(0.247714),   FML(0.409350),   FML(0.900338),  FML(0.487934),   FML(0.612168),   FML(0.662856),   FML(0.326771),   FML(0.732583),   FML(0.432896),   FML(0.944632),
    FML(0.275198),   FML(0.935939),   FML(0.741324),   FML(0.896705),   FML(0.600796),   FML(0.599831),   FML(0.082547),  FML(0.423461),   FML(0.463754),   FML(0.112159),   FML(0.663823),   FML(0.283270),   FML(0.684932),   FML(0.284654),
    FML(0.230022),   FML(0.421078),   FML(0.831967),   FML(0.998226),   FML(0.507133),   FML(0.919843),   FML(0.174371),  FML(0.045593),   FML(0.101736),   FML(0.914068),   FML(0.151496),   FML(0.556848),   FML(0.905819),   FML(0.497747),
    FML(0.064809),   FML(0.279012),   FML(0.567317),   FML(0.122217),   FML(0.365275),   FML(0.832997),   FML(0.651358),  FML(0.143087),   FML(0.849137),   FML(0.083667),   FML(0.418560),   FML(0.091737),   FML(0.474587),   FML(0.203129),
    FML(0.226585),   FML(0.941389),   FML(0.109199),   FML(0.259257),   FML(0.346985),   FML(0.544940),   FML(0.882275),  FML(0.357315),   FML(0.359397),   FML(0.199704),   FML(0.072612),   FML(0.899553),   FML(0.702922),   FML(0.634861),
    };
    u32 inputRows = 5;
    u32 inputCols = 7;

    fml correctOutput_k33[] =
    {  FML(3.0132),   FML(4.1055),   FML(3.5778),   FML(2.9616),   FML(3.1280),   FML(3.5863),   FML(2.7934),
   FML(3.9885),   FML(5.7687),   FML(5.3231),   FML(4.6334),   FML(4.7517),   FML(5.4142),   FML(4.1156),
   FML(3.0708),   FML(5.0383),   FML(6.0612),   FML(4.7944),   FML(3.1790),   FML(4.3285),   FML(3.1280),
   FML(2.8288),   FML(4.4136),   FML(5.4412),   FML(4.8158),   FML(4.5845),   FML(4.8759),   FML(3.2332),
   FML(1.0929),   FML(2.1537),   FML(2.6890),   FML(2.4095),   FML(2.3017),   FML(2.3736),   FML(2.0159),
    };

    fml correctOutput_k55[] =
     {  FML(4.6376),    FML(5.8310),    FML(6.4460),    FML(6.2339),    FML(6.1471),    FML(5.6956),    FML(4.6721),
    FML(5.0465),    FML(7.7530),    FML(9.4735),    FML(9.0572),    FML(9.1699),    FML(6.7486),    FML(6.0632),
    FML(6.3832),    FML(8.6593),   FML(11.1217),   FML(10.8636),   FML(10.5570),    FML(8.4843),    FML(7.7378),
    FML(5.1533),    FML(7.5265),    FML(8.9470),    FML(8.5797),    FML(8.2279),    FML(5.7900),    FML(5.9774),
    FML(3.6166),    FML(6.0902),    FML(7.4907),    FML(7.0047),    FML(6.8706),    FML(6.5902),    FML(5.9533),
    };

    fml correctOutput_k57[] =
     {  FML(6.0384),    FML(8.3525),    FML(9.5258),   FML(11.5353),   FML(10.2983),    FML(7.7527),    FML(6.5314),
    FML(8.5514),   FML(11.1848),   FML(12.5843),   FML(13.8372),   FML(11.8070),   FML(10.7536),    FML(7.9636),
   FML(11.2726),   FML(13.0182),   FML(15.4379),   FML(17.3875),   FML(12.8721),   FML(11.6723),    FML(8.9354),
    FML(9.0435),    FML(9.8688),   FML(11.8852),   FML(14.8820),   FML(10.2905),    FML(9.7946),    FML(7.2825),
    FML(6.9167),    FML(7.6520),    FML(9.1269),   FML(12.0685),    FML(8.4054),    FML(6.1531),    FML(6.1634),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test120(const tTest& t)
{
    fml input[] =
      {  FML(0.2663015),   FML(0.5298670),   FML(0.1694511),   FML(0.3949479),   FML(0.1471167),   FML(0.4739497),  FML(0.4404414),   FML(0.5924317),   FML(0.9135795),   FML(0.0522817),   FML(0.2670924),   FML(0.9146010),  FML(0.7975569),   FML(0.9093769),   FML(0.3271568),   FML(0.4657511),
     FML(0.2524345),   FML(0.9057547),   FML(0.9800038),   FML(0.4355511),   FML(0.4084311),   FML(0.6231749),  FML(0.0581660),   FML(0.1502395),   FML(0.7306737),   FML(0.5598986),   FML(0.0229796),   FML(0.5716656),  FML(0.0162640),   FML(0.0864149),   FML(0.7056092),   FML(0.4059839),
     FML(0.3886418),   FML(0.3932279),   FML(0.9314111),   FML(0.4395079),   FML(0.9270462),   FML(0.4930364),  FML(0.8705306),   FML(0.6011853),   FML(0.6919503),   FML(0.9595409),   FML(0.7035105),   FML(0.7339199),  FML(0.1183638),   FML(0.0065769),   FML(0.3029382),   FML(0.0663218),
     FML(0.1301186),   FML(0.3031619),   FML(0.2697365),   FML(0.3548264),   FML(0.9163647),   FML(0.4955929),  FML(0.5000574),   FML(0.3098664),   FML(0.6721016),   FML(0.9546612),   FML(0.8817705),   FML(0.2473672),  FML(0.7851411),   FML(0.6428333),   FML(0.6617618),   FML(0.6058511),
     FML(0.4444356),   FML(0.5496572),   FML(0.5558183),   FML(0.0480081),   FML(0.3228006),   FML(0.6061164),  FML(0.5854228),   FML(0.1833574),   FML(0.9351736),   FML(0.7809511),   FML(0.6087617),   FML(0.2029624),  FML(0.7108193),   FML(0.6130200),   FML(0.7567624),   FML(0.0163132),
    };
    u32 inputRows = 5;
    u32 inputCols = 8;

    fml correctOutput_k33[] =
    {  FML(2.7223),   FML(3.5836),   FML(3.0476),   FML(3.1709),   FML(2.8934),   FML(2.9894),   FML(2.9566),   FML(2.5300),
   FML(3.4512),   FML(5.3791),   FML(5.9031),   FML(5.9658),   FML(5.5563),   FML(5.8514),   FML(4.2084),   FML(2.6317),
   FML(2.9165),   FML(4.8782),   FML(5.7693),   FML(5.9581),   FML(5.7280),   FML(6.3205),   FML(5.0390),   FML(2.9193),
   FML(2.5659),   FML(4.3483),   FML(5.1567),   FML(6.5138),   FML(6.6099),   FML(6.6561),   FML(4.9528),   FML(3.1809),
   FML(1.2321),   FML(2.2051),   FML(2.3536),   FML(3.3973),   FML(3.2570),   FML(3.6861),   FML(3.4118),   FML(3.0268),
    };

    fml correctOutput_k55[] =
     {  FML(4.0603),    FML(4.9288),    FML(6.8259),    FML(6.4172),    FML(6.1241),    FML(5.9356),    FML(4.8561),    FML(3.9161),
    FML(5.0904),    FML(7.1785),    FML(9.1171),    FML(9.8647),    FML(9.6122),    FML(9.0084),    FML(8.9616),    FML(6.0874),
    FML(5.7627),    FML(8.0845),   FML(11.0563),   FML(11.4472),   FML(10.8471),   FML(10.8541),    FML(9.6782),    FML(6.4288),
    FML(5.3272),    FML(8.1238),   FML(10.2449),   FML(10.1791),   FML(10.6207),   FML(10.1210),    FML(8.2509),    FML(5.9179),
    FML(3.9950),    FML(5.3928),    FML(7.7681),    FML(8.4975),    FML(9.2285),    FML(9.2741),    FML(7.5773),    FML(5.8165),
    };

    fml correctOutput_k57[] =
     {  FML(6.8049),    FML(8.6142),    FML(9.2042),   FML(12.1226),   FML(11.4240),    FML(9.2263),    FML(8.2516),    FML(6.4280),
    FML(8.8333),   FML(11.4603),   FML(13.5715),   FML(14.7116),   FML(13.6899),   FML(12.2570),   FML(10.8597),    FML(9.3744),
   FML(10.5432),   FML(15.3451),   FML(16.9326),   FML(17.9931),   FML(19.3393),   FML(15.6411),   FML(12.9790),    FML(9.8557),
    FML(9.1604),   FML(12.6378),   FML(13.9171),   FML(14.4208),   FML(15.2605),   FML(12.1655),   FML(10.4939),    FML(8.2806),
    FML(7.6776),   FML(10.5414),   FML(11.3659),   FML(12.9334),   FML(12.1770),    FML(9.9026),    FML(8.4598),    FML(7.4646),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test121(const tTest& t)
{
    fml input[] =
    {  FML(0.882270),   FML(0.057989),
   FML(0.801343),   FML(0.353002),
   FML(0.182156),   FML(0.859805),
   FML(0.013863),   FML(0.028086),
   FML(0.773786),   FML(0.503133),
   FML(0.878506),   FML(0.865465),
    };
    u32 inputRows = 6;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(1.3226),
   FML(1.5929),
   FML(1.0937),
   FML(2.1175),
   FML(1.9470),
   FML(1.3140),
    };

    fml correctOutput_k55[] =
    {  FML(1.8353),
   FML(2.5042),
   FML(3.4219),
   FML(3.1486),
   FML(2.3793),
   FML(2.4950),
    };

    fml correctOutput_k57[] =
    {  FML(2.6520),
   FML(2.2822),
   FML(3.2690),
   FML(3.8252),
   FML(2.7782),
   FML(2.3502),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test122(const tTest& t)
{
    fml input[] =
    {  FML(0.2529933),   FML(0.5223073),   FML(0.0407558),   FML(0.4633277),
   FML(0.1801390),   FML(0.3561978),   FML(0.0015948),   FML(0.4414440),
   FML(0.7807534),   FML(0.2424685),   FML(0.5717727),   FML(0.2727570),
   FML(0.0342339),   FML(0.6719698),   FML(0.4916853),   FML(0.5289262),
   FML(0.3150944),   FML(0.1122526),   FML(0.4535860),   FML(0.2343748),
   FML(0.1989366),   FML(0.3135770),   FML(0.4381261),   FML(0.5983162),
    };
    u32 inputRows = 6;
    u32 inputCols = 2;

    fml correctOutput_k33[] =
    {  FML(1.34305),   FML(1.67365),
   FML(2.65228),   FML(3.11712),
   FML(2.71262),   FML(2.98762),
   FML(2.14885),   FML(3.07566),
   FML(2.76743),   FML(2.74048),
   FML(0.93738),   FML(1.55602),
    };

    fml correctOutput_k55[] =
    {  FML(2.3712),   FML(2.6393),
   FML(3.1286),   FML(3.6890),
   FML(3.2006),   FML(3.7402),
   FML(4.0207),   FML(5.0487),
   FML(2.8666),   FML(3.9121),
   FML(2.4871),   FML(3.7340),
    };

    fml correctOutput_k57[] =
    {  FML(2.7655),   FML(3.5096),
   FML(3.2984),   FML(3.7838),
   FML(4.5932),   FML(4.7870),
   FML(4.1869),   FML(4.2654),
   FML(4.2518),   FML(3.9253),
   FML(3.0053),   FML(2.8691),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test123(const tTest& t)
{
    fml input[] =
    {  FML(0.728014),   FML(0.379671),   FML(0.302480),   FML(0.088828),   FML(0.724682),   FML(0.025404),
   FML(0.679322),   FML(0.606552),   FML(0.428059),   FML(0.397568),   FML(0.783968),   FML(0.567133),
   FML(0.141792),   FML(0.943262),   FML(0.785151),   FML(0.961378),   FML(0.391408),   FML(0.863411),
   FML(0.705159),   FML(0.251472),   FML(0.622439),   FML(0.938085),   FML(0.128542),   FML(0.226518),
   FML(0.292076),   FML(0.350563),   FML(0.421489),   FML(0.567667),   FML(0.219009),   FML(0.235142),
   FML(0.916564),   FML(0.151372),   FML(0.623698),   FML(0.947180),   FML(0.107355),   FML(0.561343),
    };
    u32 inputRows = 6;
    u32 inputCols = 3;

    fml correctOutput_k33[] =
    {  FML(2.3213),   FML(3.9657),   FML(2.2439),
   FML(3.6828),   FML(5.6745),   FML(3.7042),
   FML(3.8339),   FML(5.5888),   FML(4.2627),
   FML(3.5108),   FML(4.8991),   FML(4.4520),
   FML(3.5364),   FML(5.5759),   FML(3.8061),
   FML(1.4378),   FML(2.5998),   FML(2.4704),
    };

    fml correctOutput_k55[] =
    {  FML(4.3370),   FML(4.3956),   FML(4.1782),
   FML(6.1777),   FML(6.1505),   FML(6.9893),
   FML(6.8796),   FML(7.7636),   FML(8.2744),
   FML(6.1726),   FML(7.7699),   FML(8.5876),
   FML(4.5470),   FML(6.9345),   FML(7.3602),
   FML(3.7104),   FML(4.6490),   FML(5.7661),
    };

    fml correctOutput_k57[] =
    {  FML(4.7407),   FML(6.0197),   FML(5.8274),
   FML(6.6028),   FML(8.0722),   FML(6.6073),
   FML(7.4885),   FML(7.8503),   FML(8.4901),
   FML(8.3833),   FML(8.9344),   FML(9.1549),
   FML(6.5954),   FML(6.1307),   FML(7.0723),
   FML(5.5882),   FML(4.9773),   FML(5.0387),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test124(const tTest& t)
{
    fml input[] =
     {  FML(0.924720),   FML(0.945141),   FML(0.550653),   FML(0.680476),   FML(0.756074),   FML(0.086045),   FML(0.106809),  FML(0.923181),
    FML(0.340620),   FML(0.484858),   FML(0.683733),   FML(0.835804),   FML(0.855392),   FML(0.358668),   FML(0.599477),  FML(0.199529),
    FML(0.126502),   FML(0.855239),   FML(0.202738),   FML(0.039565),   FML(0.466463),   FML(0.897922),   FML(0.371731),  FML(0.898931),
    FML(0.172949),   FML(0.166371),   FML(0.532999),   FML(0.825295),   FML(0.158783),   FML(0.386097),   FML(0.218639),  FML(0.937665),
    FML(0.026100),   FML(0.557545),   FML(0.566034),   FML(0.188026),   FML(0.158896),   FML(0.115980),   FML(0.161290),  FML(0.130475),
    FML(0.932902),   FML(0.136821),   FML(0.366523),   FML(0.714141),   FML(0.480489),   FML(0.646207),   FML(0.307490),  FML(0.125860),
    };
    u32 inputRows = 6;
    u32 inputCols = 4;

    fml correctOutput_k33[] =
    {  FML(2.8810),   FML(4.8066),   FML(3.8806),   FML(2.6030),
   FML(3.1899),   FML(5.4899),   FML(5.5361),   FML(4.6988),
   FML(3.0205),   FML(4.6900),   FML(4.8457),   FML(4.0648),
   FML(2.6077),   FML(2.9793),   FML(4.0726),   FML(3.1470),
   FML(2.9451),   FML(4.8795),   FML(4.4489),   FML(2.8046),
   FML(1.3610),   FML(2.0571),   FML(2.2670),   FML(1.5189),
    };

    fml correctOutput_k55[] =
    {  FML(4.0724),   FML(5.6964),   FML(6.4260),   FML(4.7145),
   FML(6.4257),   FML(8.0757),   FML(9.0934),   FML(6.7013),
   FML(6.3555),   FML(9.0936),   FML(9.0599),   FML(7.7602),
   FML(5.3437),   FML(8.0170),   FML(8.1803),   FML(7.0442),
   FML(3.8218),   FML(5.9948),   FML(5.8713),   FML(6.1965),
   FML(2.9068),   FML(4.5955),   FML(5.4000),   FML(4.6691),
    };

    fml correctOutput_k57[] =
     {  FML(7.0020),    FML(7.5520),    FML(6.9139),    FML(7.0413),
    FML(8.6004),    FML(9.1073),    FML(8.8750),    FML(8.7273),
   FML(10.8400),    FML(9.9119),    FML(9.9738),    FML(8.7475),
   FML(10.4780),    FML(9.6040),    FML(8.5254),    FML(9.3461),
    FML(7.6759),    FML(6.4377),    FML(7.8250),    FML(6.0218),
    FML(6.2594),    FML(6.0091),    FML(5.3417),    FML(5.3377),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test125(const tTest& t)
{
    fml input[] =
     {  FML(0.939083),   FML(0.392853),   FML(0.415083),   FML(0.501440),   FML(0.772019),   FML(0.813773),   FML(0.722120),  FML(0.584345),   FML(0.768939),   FML(0.619559),
    FML(0.990579),   FML(0.526277),   FML(0.045488),   FML(0.255881),   FML(0.352494),   FML(0.338745),   FML(0.460591),  FML(0.939181),   FML(0.425291),   FML(0.679692),
    FML(0.847717),   FML(0.432573),   FML(0.653227),   FML(0.508407),   FML(0.372252),   FML(0.662422),   FML(0.935837),  FML(0.871778),   FML(0.025566),   FML(0.269299),
    FML(0.597406),   FML(0.137386),   FML(0.684270),   FML(0.616802),   FML(0.670327),   FML(0.177580),   FML(0.018953),  FML(0.189041),   FML(0.728687),   FML(0.683074),
    FML(0.167293),   FML(0.694330),   FML(0.635422),   FML(0.867514),   FML(0.426104),   FML(0.579302),   FML(0.671283),  FML(0.088039),   FML(0.459505),   FML(0.048274),
    FML(0.566214),   FML(0.408811),   FML(0.079426),   FML(0.956946),   FML(0.382278),   FML(0.733425),   FML(0.177252),  FML(0.101729),   FML(0.050639),   FML(0.845298),
    };
    u32 inputRows = 6;
    u32 inputCols = 5;

    fml correctOutput_k33[] =
    {  FML(2.1167),   FML(3.5956),   FML(3.4173),   FML(4.4335),   FML(3.1523),
   FML(3.3704),   FML(6.2322),   FML(5.8442),   FML(5.6412),   FML(4.6102),
   FML(3.2052),   FML(5.4476),   FML(4.3076),   FML(5.2545),   FML(4.3237),
   FML(3.5228),   FML(5.6590),   FML(5.7240),   FML(4.4866),   FML(3.2764),
   FML(2.9784),   FML(5.3960),   FML(4.6561),   FML(4.1742),   FML(2.6858),
   FML(1.7396),   FML(3.0970),   FML(3.1260),   FML(2.1311),   FML(1.5529),
    };

    fml correctOutput_k55[] =
     {  FML(3.9684),    FML(5.9752),    FML(7.0279),    FML(5.6964),    FML(5.0214),
    FML(5.9781),    FML(8.3965),   FML(10.1064),    FML(8.4383),    FML(8.1127),
    FML(6.9913),    FML(9.1451),   FML(11.2544),    FML(8.8395),    FML(7.8841),
    FML(6.8283),    FML(9.2150),    FML(9.8304),    FML(9.4343),    FML(7.7618),
    FML(5.8011),    FML(7.2540),    FML(9.8141),    FML(8.1447),    FML(5.8417),
    FML(4.1063),    FML(6.2048),    FML(7.9890),    FML(6.3781),    FML(5.3437),
    };

    fml correctOutput_k57[] =
     {  FML(7.8587),    FML(9.6139),    FML(8.8730),    FML(9.6991),    FML(7.6009),
    FML(9.9662),   FML(12.0863),   FML(10.8179),   FML(10.9729),    FML(8.9277),
   FML(12.4195),   FML(14.9932),   FML(13.7767),   FML(12.1596),   FML(11.2149),
   FML(11.2525),   FML(12.7181),   FML(13.0934),   FML(11.6307),    FML(9.2511),
    FML(9.0011),   FML(10.3667),   FML(10.4653),    FML(9.6584),    FML(7.3693),
    FML(6.4566),    FML(8.5194),    FML(8.5291),    FML(6.7024),    FML(4.7631),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test126(const tTest& t)
{
    fml input[] =
     {  FML(0.7696598),   FML(0.5809967),   FML(0.5833682),   FML(0.0669902),   FML(0.1645468),   FML(0.1505291),  FML(0.5280842),   FML(0.8457189),   FML(0.2584278),   FML(0.2751002),   FML(0.9786395),   FML(0.2283110),
    FML(0.9016448),   FML(0.6266332),   FML(0.9294598),   FML(0.0080765),   FML(0.7548543),   FML(0.6556199),  FML(0.8764767),   FML(0.3558066),   FML(0.5760555),   FML(0.2764507),   FML(0.4226294),   FML(0.0597144),
    FML(0.9957849),   FML(0.8970046),   FML(0.8028797),   FML(0.2398551),   FML(0.4060508),   FML(0.9165092),  FML(0.3785806),   FML(0.3650713),   FML(0.0556052),   FML(0.8018921),   FML(0.4399167),   FML(0.7620467),
    FML(0.0549258),   FML(0.2169278),   FML(0.4532107),   FML(0.7689895),   FML(0.0465404),   FML(0.3134089),  FML(0.8713403),   FML(0.2137956),   FML(0.7208623),   FML(0.3770862),   FML(0.3710540),   FML(0.3893199),
    FML(0.8921889),   FML(0.3085383),   FML(0.9016255),   FML(0.2453220),   FML(0.7981675),   FML(0.1921422),  FML(0.6996090),   FML(0.6044171),   FML(0.2666225),   FML(0.9791113),   FML(0.8327283),   FML(0.4579756),
    FML(0.5429937),   FML(0.8483480),   FML(0.0423416),   FML(0.1213352),   FML(0.4151281),   FML(0.2045422),  FML(0.5814771),   FML(0.8071222),   FML(0.3174864),   FML(0.1107829),   FML(0.6829798),   FML(0.8891957),
    };
    u32 inputRows = 6;
    u32 inputCols = 6;

    fml correctOutput_k33[] =
    {  FML(2.7454),   FML(4.2614),   FML(3.9210),   FML(3.5061),   FML(3.6458),   FML(1.8559),
   FML(4.0529),   FML(6.0811),   FML(5.1664),   FML(4.9849),   FML(5.0486),   FML(3.0226),
   FML(3.1466),   FML(5.0192),   FML(5.3019),   FML(5.0015),   FML(4.9542),   FML(3.3739),
   FML(3.6359),   FML(5.5171),   FML(6.2322),   FML(5.3504),   FML(5.9977),   FML(3.9948),
   FML(2.6807),   FML(4.1451),   FML(4.4974),   FML(4.0400),   FML(5.8954),   FML(4.2620),
   FML(1.3477),   FML(3.0076),   FML(2.4132),   FML(3.1298),   FML(3.8155),   FML(2.2058),
    };

    fml correctOutput_k55[] =
     {  FML(4.1061),    FML(5.5253),    FML(6.0688),    FML(5.3047),    FML(5.3946),    FML(3.9921),
    FML(5.5705),    FML(7.9894),    FML(8.9854),    FML(8.5929),    FML(7.3579),    FML(6.1653),
    FML(7.2385),   FML(10.0038),   FML(11.3608),   FML(11.8099),   FML(10.3110),    FML(7.9655),
    FML(6.8391),    FML(9.4751),   FML(10.9073),   FML(11.2265),    FML(8.3370),    FML(7.2887),
    FML(4.5422),    FML(7.3403),    FML(8.3103),    FML(9.1276),    FML(7.6715),    FML(8.3208),
    FML(4.0976),    FML(6.6814),    FML(7.7972),    FML(7.8684),    FML(6.9682),    FML(6.7338),
    };

    fml correctOutput_k57[] =
     {  FML(8.2953),    FML(8.6843),    FML(8.7747),    FML(8.8894),    FML(7.9599),    FML(6.3983),
    FML(9.6911),   FML(10.6151),   FML(12.9450),   FML(12.0692),   FML(11.0361),    FML(8.2315),
   FML(13.2002),   FML(14.6775),   FML(16.8946),   FML(15.2376),   FML(11.4856),   FML(10.2197),
   FML(11.7346),   FML(12.6731),   FML(16.3437),   FML(13.6916),   FML(12.1353),   FML(10.3598),
   FML(10.1739),   FML(11.7042),   FML(15.4567),   FML(11.9548),    FML(9.9729),    FML(8.3320),
    FML(7.5862),    FML(8.8474),    FML(9.1508),    FML(9.4973),    FML(7.9853),    FML(6.4841),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test127(const tTest& t)
{
    fml input[] =
     {  FML(0.260554),   FML(0.109733),   FML(0.383315),   FML(0.883732),   FML(0.835546),   FML(0.100609),   FML(0.763095),  FML(0.252321),   FML(0.785242),   FML(0.219556),   FML(0.805332),   FML(0.808003),   FML(0.889479),   FML(0.667159),
    FML(0.518100),   FML(0.281783),   FML(0.731262),   FML(0.196991),   FML(0.236067),   FML(0.606348),   FML(0.712873),  FML(0.636356),   FML(0.766216),   FML(0.721417),   FML(0.928648),   FML(0.739081),   FML(0.485670),   FML(0.946662),
    FML(0.419832),   FML(0.790561),   FML(0.742981),   FML(0.925224),   FML(0.916256),   FML(0.016916),   FML(0.267992),  FML(0.690724),   FML(0.299757),   FML(0.656049),   FML(0.905040),   FML(0.807654),   FML(0.290867),   FML(0.852668),
    FML(0.039172),   FML(0.825472),   FML(0.350863),   FML(0.262126),   FML(0.647378),   FML(0.065924),   FML(0.226720),  FML(0.108620),   FML(0.246787),   FML(0.717773),   FML(0.374438),   FML(0.652127),   FML(0.404085),   FML(0.812590),
    FML(0.659229),   FML(0.546708),   FML(0.852842),   FML(0.913675),   FML(0.712278),   FML(0.632662),   FML(0.463147),  FML(0.310379),   FML(0.204084),   FML(0.322994),   FML(0.290867),   FML(0.706692),   FML(0.680674),   FML(0.565554),
    FML(0.377432),   FML(0.439637),   FML(0.282836),   FML(0.890584),   FML(0.461267),   FML(0.391194),   FML(0.088576),  FML(0.020975),   FML(0.272321),   FML(0.409703),   FML(0.079205),   FML(0.078354),   FML(0.311301),   FML(0.957251),
    };
    u32 inputRows = 6;
    u32 inputCols = 7;

    fml correctOutput_k33[] =
    {  FML(1.9364),   FML(3.0969),   FML(4.1402),   FML(4.1119),   FML(4.8869),   FML(5.0795),   FML(3.9801),
   FML(3.6569),   FML(5.2562),   FML(5.0606),   FML(5.2552),   FML(6.2383),   FML(6.8166),   FML(5.7003),
   FML(2.6845),   FML(4.4901),   FML(4.6719),   FML(4.5273),   FML(5.6302),   FML(6.1853),   FML(5.4847),
   FML(4.3978),   FML(6.4406),   FML(5.3081),   FML(4.7401),   FML(4.4101),   FML(5.4320),   FML(4.8705),
   FML(3.4684),   FML(4.7839),   FML(4.0203),   FML(3.7953),   FML(3.1371),   FML(4.3899),   FML(3.8675),
   FML(1.7707),   FML(3.5726),   FML(3.4647),   FML(2.4936),   FML(1.8299),   FML(2.3476),   FML(1.9774),
    };

    fml correctOutput_k55[] =
     {  FML(4.1618),    FML(4.8692),    FML(6.0703),    FML(6.7304),    FML(7.1022),    FML(6.4950),    FML(5.7861),
    FML(5.2053),    FML(6.6115),    FML(8.9585),   FML(10.9045),   FML(10.8112),    FML(9.9409),    FML(9.3776),
    FML(6.4132),    FML(9.3366),   FML(11.4767),   FML(11.2083),   FML(11.8324),   FML(11.2202),   FML(10.4982),
    FML(6.9136),    FML(9.0905),   FML(10.2304),   FML(10.8335),   FML(10.2660),    FML(9.7901),    FML(9.1265),
    FML(5.2962),    FML(7.4322),    FML(8.9917),    FML(8.0124),    FML(8.2975),    FML(7.0532),    FML(7.0276),
    FML(4.6064),    FML(6.6490),    FML(7.7926),    FML(6.7979),    FML(6.3622),    FML(5.2610),    FML(5.5987),
    };

    fml correctOutput_k57[] =
     {  FML(6.6907),    FML(8.3367),   FML(11.0590),   FML(12.7890),   FML(10.8202),   FML(10.4687),    FML(9.2975),
    FML(8.7361),   FML(11.1802),   FML(14.0120),   FML(15.5024),   FML(13.8191),   FML(11.4420),   FML(10.5525),
   FML(11.0057),   FML(15.0561),   FML(17.4096),   FML(20.2546),   FML(16.7470),   FML(13.4503),   FML(11.4500),
   FML(10.5883),   FML(13.9968),   FML(16.3583),   FML(18.0662),   FML(14.6488),   FML(12.5741),    FML(9.9193),
    FML(8.6025),   FML(11.1894),   FML(13.3794),   FML(14.7966),   FML(11.4166),    FML(9.3549),    FML(7.3461),
    FML(5.8834),    FML(7.4236),    FML(9.8056),   FML(10.4532),    FML(8.0898),    FML(6.1087),    FML(4.9728),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test128(const tTest& t)
{
    fml input[] =
      {  FML(0.2420863),   FML(0.7749331),   FML(0.5872553),   FML(0.7639267),   FML(0.1842955),   FML(0.4602764),  FML(0.3746135),   FML(0.1843359),   FML(0.9905351),   FML(0.2147150),   FML(0.5687583),   FML(0.7240237),  FML(0.4828833),   FML(0.1738752),   FML(0.8541272),   FML(0.7610689),
     FML(0.0897938),   FML(0.4899255),   FML(0.3736659),   FML(0.1859425),   FML(0.0242267),   FML(0.0714571),  FML(0.2808377),   FML(0.2837080),   FML(0.5328624),   FML(0.9309819),   FML(0.1789953),   FML(0.8949817),  FML(0.6155349),   FML(0.5556532),   FML(0.5534074),   FML(0.7462056),
     FML(0.3590079),   FML(0.1827067),   FML(0.4150158),   FML(0.0218929),   FML(0.9828412),   FML(0.7276015),  FML(0.3990663),   FML(0.5773676),   FML(0.5333008),   FML(0.4515338),   FML(0.2387665),   FML(0.9059562),  FML(0.4282577),   FML(0.6108001),   FML(0.7289566),   FML(0.4472884),
     FML(0.6316260),   FML(0.4301613),   FML(0.3649155),   FML(0.4429061),   FML(0.4685529),   FML(0.0024069),  FML(0.5589899),   FML(0.2363173),   FML(0.6395419),   FML(0.6151861),   FML(0.4039951),   FML(0.6207181),  FML(0.2585325),   FML(0.7624559),   FML(0.4794019),   FML(0.9091713),
     FML(0.0159479),   FML(0.0563229),   FML(0.2758771),   FML(0.9325738),   FML(0.7203848),   FML(0.2953105),  FML(0.0713395),   FML(0.0107753),   FML(0.5867208),   FML(0.5769603),   FML(0.3709379),   FML(0.6182227),  FML(0.0825403),   FML(0.0857010),   FML(0.9846958),   FML(0.7661825),
     FML(0.6480292),   FML(0.0896834),   FML(0.0167639),   FML(0.1448162),   FML(0.5129896),   FML(0.2618646),  FML(0.9144188),   FML(0.0709889),   FML(0.3113017),   FML(0.0123145),   FML(0.7702485),   FML(0.6682258),  FML(0.4875019),   FML(0.4085583),   FML(0.4504341),   FML(0.6940337),
    };
    u32 inputRows = 6;
    u32 inputCols = 8;

    fml correctOutput_k33[] =
    {  FML(1.76103),   FML(2.20659),   FML(2.48291),   FML(2.92681),   FML(3.53795),   FML(4.27723),   FML(4.15334),   FML(2.87695),
   FML(2.55045),   FML(4.57750),   FML(4.76407),   FML(4.32048),   FML(4.62899),   FML(6.23796),   FML(5.47758),   FML(4.29462),
   FML(2.64999),   FML(3.41178),   FML(3.28759),   FML(4.75312),   FML(5.46313),   FML(6.01596),   FML(5.78994),   FML(4.52495),
   FML(2.06488),   FML(4.18619),   FML(4.11894),   FML(4.91576),   FML(4.61054),   FML(5.09997),   FML(5.22545),   FML(3.99810),
   FML(1.86650),   FML(3.65421),   FML(3.66538),   FML(3.82794),   FML(4.66914),   FML(5.35840),   FML(5.83223),   FML(3.80636),
   FML(0.94674),   FML(2.20575),   FML(1.96625),   FML(2.25973),   FML(2.29554),   FML(2.51756),   FML(2.83620),   FML(2.18531),
    };

    fml correctOutput_k55[] =
     {  FML(3.1473),    FML(4.5083),    FML(6.0469),    FML(5.7961),    FML(5.6676),    FML(6.5938),    FML(6.0863),    FML(4.8645),
    FML(4.4968),    FML(6.0021),    FML(8.0062),    FML(8.5759),    FML(8.7287),   FML(10.9479),    FML(9.7949),    FML(7.7679),
    FML(4.6313),    FML(7.2561),    FML(8.0345),    FML(8.6868),   FML(10.0826),   FML(11.1502),   FML(10.6666),    FML(8.7287),
    FML(4.8846),    FML(6.3307),    FML(8.3072),    FML(9.1363),    FML(9.8766),   FML(11.4106),   FML(10.4167),    FML(8.7337),
    FML(4.0790),    FML(5.9801),    FML(7.5279),    FML(8.3029),    FML(8.0959),    FML(8.6958),    FML(8.4928),    FML(7.3853),
    FML(3.1337),    FML(3.6712),    FML(5.9031),    FML(5.5208),    FML(5.7176),    FML(7.2678),    FML(6.2726),    FML(6.4638),
    };

    fml correctOutput_k57[] =
     {  FML(5.4261),    FML(7.1305),    FML(8.5497),   FML(10.5052),   FML(10.3880),   FML(10.2224),    FML(9.0730),    FML(8.2421),
    FML(7.1183),   FML(10.4845),   FML(11.7286),   FML(12.9451),   FML(13.8918),   FML(12.0814),   FML(11.5646),    FML(9.4898),
    FML(8.6700),   FML(12.3653),   FML(14.2114),   FML(16.2332),   FML(17.4353),   FML(14.9932),   FML(12.8049),   FML(11.7898),
    FML(7.4314),   FML(11.6915),   FML(13.5945),   FML(16.5592),   FML(18.2460),   FML(14.0868),   FML(13.5757),   FML(10.6422),
    FML(6.9588),    FML(9.3195),   FML(12.6095),   FML(11.4509),   FML(14.0483),   FML(11.3943),    FML(9.5604),    FML(8.4863),
    FML(4.9530),    FML(6.6947),    FML(9.2871),    FML(9.6663),   FML(10.9672),    FML(9.1210),    FML(8.5878),    FML(5.9353),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test129(const tTest& t)
{
    fml input[] =
    {  FML(0.748330),   FML(0.465066),
   FML(0.954748),   FML(0.526576),
   FML(0.315724),   FML(0.561770),
   FML(0.750846),   FML(0.842137),
   FML(0.583986),   FML(0.318377),
   FML(0.309570),   FML(0.055645),
   FML(0.111286),   FML(0.187218),
    };
    u32 inputRows = 7;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(1.67034),
   FML(1.87275),
   FML(2.39916),
   FML(1.93398),
   FML(1.67757),
   FML(1.01114),
   FML(0.49420),
    };

    fml correctOutput_k55[] =
    {  FML(2.0031),
   FML(3.1789),
   FML(3.9189),
   FML(3.2623),
   FML(2.9166),
   FML(2.6845),
   FML(1.8705),
    };

    fml correctOutput_k57[] =
    {  FML(2.7134),
   FML(3.6702),
   FML(3.7243),
   FML(3.6542),
   FML(2.8354),
   FML(2.4134),
   FML(1.6746),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test130(const tTest& t)
{
    fml input[] =
    {  FML(0.089436),   FML(0.265889),   FML(0.344868),   FML(0.251517),
   FML(0.038855),   FML(0.343838),   FML(0.086972),   FML(0.401413),
   FML(0.761868),   FML(0.592292),   FML(0.746300),   FML(0.076209),
   FML(0.318443),   FML(0.624837),   FML(0.780451),   FML(0.458793),
   FML(0.850298),   FML(0.558831),   FML(0.677204),   FML(0.454422),
   FML(0.085240),   FML(0.831942),   FML(0.819847),   FML(0.880061),
   FML(0.983042),   FML(0.634974),   FML(0.748589),   FML(0.985153),
    };
    u32 inputRows = 7;
    u32 inputCols = 2;

    fml correctOutput_k33[] =
    {  FML(1.2626),   FML(1.2733),
   FML(2.6081),   FML(2.6915),
   FML(3.1631),   FML(3.3952),
   FML(3.6579),   FML(4.2059),
   FML(4.0033),   FML(4.3427),
   FML(4.4997),   FML(5.3962),
   FML(2.2449),   FML(3.1949),
    };

    fml correctOutput_k55[] =
    {  FML(2.2833),   FML(2.2032),
   FML(3.0884),   FML(3.5935),
   FML(3.7037),   FML(4.3278),
   FML(5.1178),   FML(6.3210),
   FML(5.7840),   FML(6.8478),
   FML(4.7030),   FML(7.1322),
   FML(3.8961),   FML(5.5418),
    };

    fml correctOutput_k57[] =
    {  FML(3.0599),   FML(3.4239),
   FML(3.7906),   FML(3.8508),
   FML(5.6390),   FML(6.0615),
   FML(5.8626),   FML(5.9319),
   FML(7.5524),   FML(7.8942),
   FML(5.5014),   FML(5.7076),
   FML(5.6072),   FML(5.3634),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test131(const tTest& t)
{
    fml input[] =
    {  FML(0.269503),   FML(0.588035),   FML(0.720192),   FML(0.730447),   FML(0.082328),   FML(0.269591),
   FML(0.388686),   FML(0.969592),   FML(0.263488),   FML(0.623179),   FML(0.644451),   FML(0.189641),
   FML(0.265769),   FML(0.095569),   FML(0.778702),   FML(0.980053),   FML(0.691756),   FML(0.453755),
   FML(0.538041),   FML(0.131967),   FML(0.448400),   FML(0.977666),   FML(0.460686),   FML(0.995100),
   FML(0.766708),   FML(0.200711),   FML(0.874395),   FML(0.417353),   FML(0.431476),   FML(0.812936),
   FML(0.271553),   FML(0.601290),   FML(0.836284),   FML(0.234372),   FML(0.668338),   FML(0.773366),
   FML(0.466713),   FML(0.132675),   FML(0.120953),   FML(0.643438),   FML(0.870604),   FML(0.660369),
    };
    u32 inputRows = 7;
    u32 inputCols = 3;

    fml correctOutput_k33[] =
    {  FML(2.6306),   FML(3.4381),   FML(2.4738),
   FML(3.5350),   FML(5.7245),   FML(4.2953),
   FML(3.6397),   FML(5.6766),   FML(4.5018),
   FML(3.0752),   FML(5.6749),   FML(5.2696),
   FML(3.0848),   FML(5.8950),   FML(5.2498),
   FML(2.5103),   FML(5.4982),   FML(4.7124),
   FML(1.3397),   FML(2.5179),   FML(2.9229),
    };

    fml correctOutput_k55[] =
    {  FML(4.3052),   FML(4.5331),   FML(4.4410),
   FML(5.1431),   FML(6.9964),   FML(6.9852),
   FML(6.3516),   FML(7.9467),   FML(8.0686),
   FML(6.5148),   FML(7.8527),   FML(9.1843),
   FML(6.2575),   FML(7.2853),   FML(9.0722),
   FML(5.5010),   FML(6.4411),   FML(7.8333),
   FML(4.3766),   FML(5.2268),   FML(6.3313),
    };

    fml correctOutput_k57[] =
    {  FML(4.6955),   FML(5.9325),   FML(5.6525),
   FML(5.6916),   FML(7.4959),   FML(7.0726),
   FML(8.1080),   FML(9.1502),   FML(9.0263),
   FML(8.8321),   FML(7.7722),   FML(9.8828),
   FML(8.8555),   FML(9.0360),   FML(8.7643),
   FML(7.4264),   FML(7.2025),   FML(7.1375),
   FML(5.9372),   FML(5.8625),   FML(5.6355),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test132(const tTest& t)
{
    fml input[] =
     {  FML(0.3018522),   FML(0.8777775),   FML(0.7533985),   FML(0.9399330),   FML(0.7973152),   FML(0.1603518),  FML(0.5331519),   FML(0.7879976),
    FML(0.9282459),   FML(0.7483783),   FML(0.0363024),   FML(0.5623767),   FML(0.0498489),   FML(0.0025654),  FML(0.2751203),   FML(0.4179299),
    FML(0.6215147),   FML(0.8698136),   FML(0.7874700),   FML(0.9470927),   FML(0.3876044),   FML(0.6560880),  FML(0.5852818),   FML(0.4356839),
    FML(0.8972954),   FML(0.7519899),   FML(0.3367473),   FML(0.1773273),   FML(0.0680115),   FML(0.1344297),  FML(0.3124193),   FML(0.0616372),
    FML(0.0563768),   FML(0.8683212),   FML(0.2893152),   FML(0.0458760),   FML(0.3595497),   FML(0.5946741),  FML(0.9321202),   FML(0.1767183),
    FML(0.9579902),   FML(0.6636258),   FML(0.3302956),   FML(0.5907979),   FML(0.2266989),   FML(0.2800102),  FML(0.1375968),   FML(0.6760292),
    FML(0.6578323),   FML(0.4173316),   FML(0.9473743),   FML(0.9119208),   FML(0.5459390),   FML(0.2801335),  FML(0.3670130),   FML(0.3553704),
    };
    u32 inputRows = 7;
    u32 inputCols = 4;

    fml correctOutput_k33[] =
    {  FML(2.6790),   FML(3.4639),   FML(2.7431),   FML(1.8313),
   FML(4.7220),   FML(6.5905),   FML(5.4217),   FML(3.5493),
   FML(3.7420),   FML(5.2834),   FML(3.2987),   FML(2.1213),
   FML(3.2855),   FML(5.3192),   FML(4.8657),   FML(3.0241),
   FML(3.6378),   FML(4.8989),   FML(3.2940),   FML(2.3795),
   FML(4.0645),   FML(5.3713),   FML(4.6909),   FML(2.9281),
   FML(1.9719),   FML(3.5971),   FML(3.0378),   FML(2.0195),
    };

    fml correctOutput_k55[] =
    {  FML(4.6541),   FML(6.4676),   FML(5.8792),   FML(4.2239),
   FML(5.6113),   FML(8.0614),   FML(8.5037),   FML(6.0851),
   FML(6.2360),   FML(9.7908),   FML(9.5094),   FML(5.9686),
   FML(6.7936),   FML(8.8330),   FML(8.9824),   FML(6.1199),
   FML(6.8222),   FML(9.7774),   FML(8.5891),   FML(5.9136),
   FML(4.9663),   FML(7.3172),   FML(6.7996),   FML(6.0282),
   FML(4.4105),   FML(6.2271),   FML(6.5962),   FML(5.5578),
    };

    fml correctOutput_k57[] =
     {  FML(6.9179),    FML(8.1835),    FML(7.7998),    FML(7.1976),
    FML(9.6547),    FML(9.1059),    FML(8.0947),    FML(8.2667),
   FML(10.9541),   FML(10.2761),   FML(11.0167),    FML(9.6812),
   FML(10.8436),    FML(9.2544),    FML(9.7697),    FML(9.2104),
   FML(10.7163),   FML(10.2335),   FML(10.2823),    FML(9.8413),
    FML(8.8437),    FML(7.6534),    FML(7.8490),    FML(6.7433),
    FML(7.3656),    FML(6.8591),    FML(7.3555),    FML(5.8369),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test133(const tTest& t)
{
    fml input[] =
     {  FML(0.970007),   FML(0.616502),   FML(0.217931),   FML(0.880747),   FML(0.027744),   FML(0.969827),   FML(0.851066),  FML(0.084313),   FML(0.978754),   FML(0.227377),
    FML(0.678642),   FML(0.901369),   FML(0.637836),   FML(0.180514),   FML(0.956860),   FML(0.693803),   FML(0.972322),  FML(0.882357),   FML(0.351374),   FML(0.889469),
    FML(0.914142),   FML(0.717089),   FML(0.421027),   FML(0.312556),   FML(0.975192),   FML(0.031687),   FML(0.395912),  FML(0.250566),   FML(0.015437),   FML(0.081029),
    FML(0.321768),   FML(0.084367),   FML(0.740559),   FML(0.896079),   FML(0.017696),   FML(0.375626),   FML(0.789206),  FML(0.786907),   FML(0.537856),   FML(0.097622),
    FML(0.590666),   FML(0.384767),   FML(0.529333),   FML(0.738265),   FML(0.753574),   FML(0.799058),   FML(0.284293),  FML(0.360484),   FML(0.635617),   FML(0.141660),
    FML(0.307562),   FML(0.531326),   FML(0.957070),   FML(0.768401),   FML(0.847460),   FML(0.913163),   FML(0.235779),  FML(0.760799),   FML(0.515394),   FML(0.068959),
    FML(0.020842),   FML(0.399317),   FML(0.686612),   FML(0.737499),   FML(0.826159),   FML(0.512563),   FML(0.411384),  FML(0.415412),   FML(0.573934),   FML(0.037253),
    };
    u32 inputRows = 7;
    u32 inputCols = 5;

    fml correctOutput_k33[] =
    {  FML(2.6838),   FML(4.6607),   FML(5.0815),   FML(5.0908),   FML(3.3334),
   FML(3.8836),   FML(6.8393),   FML(4.9655),   FML(4.2984),   FML(3.6138),
   FML(3.6299),   FML(5.5142),   FML(5.7899),   FML(5.6161),   FML(4.4289),
   FML(3.4854),   FML(5.6484),   FML(5.2068),   FML(4.5579),   FML(3.0033),
   FML(3.4609),   FML(6.2805),   FML(6.8670),   FML(5.3059),   FML(3.0999),
   FML(3.2654),   FML(5.9492),   FML(6.4623),   FML(5.6100),   FML(2.7268),
   FML(1.7120),   FML(3.1431),   FML(4.2976),   FML(3.6388),   FML(1.6750),
    };

    fml correctOutput_k55[] =
     {  FML(4.9041),    FML(5.8356),    FML(7.0103),    FML(5.3977),    FML(4.0896),
    FML(6.0505),    FML(8.9167),   FML(10.5697),    FML(8.0142),    FML(7.0577),
    FML(7.2934),   FML(10.7533),   FML(11.7682),    FML(9.7570),    FML(8.1094),
    FML(7.9819),   FML(10.1612),   FML(11.7530),   FML(10.1406),    FML(7.0720),
    FML(6.8069),    FML(9.6564),   FML(11.1984),    FML(9.3066),    FML(6.7622),
    FML(5.8686),    FML(7.8182),   FML(10.2585),    FML(9.6244),    FML(7.3762),
    FML(4.8687),    FML(6.8000),    FML(8.8040),    FML(8.0090),    FML(6.0414),
    };

    fml correctOutput_k57[] =
     {  FML(8.8065),    FML(8.8238),    FML(8.3084),    FML(9.6881),    FML(6.3706),
    FML(8.9562),   FML(12.9106),   FML(12.2779),   FML(11.1742),    FML(8.7474),
   FML(13.6777),   FML(15.0004),   FML(13.2107),   FML(13.0971),   FML(10.8197),
   FML(12.8593),   FML(14.1312),   FML(14.5741),   FML(12.4990),   FML(10.8646),
   FML(11.9659),   FML(12.1349),   FML(13.4192),   FML(12.5446),    FML(9.9813),
   FML(10.4507),   FML(12.3668),    FML(9.7583),   FML(11.4617),    FML(8.0466),
    FML(8.4031),    FML(9.0130),    FML(8.6747),    FML(8.4064),    FML(7.1072),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test134(const tTest& t)
{
    fml input[] =
     {  FML(0.2520722),   FML(0.0448857),   FML(0.6184504),   FML(0.6877250),   FML(0.7055517),   FML(0.7787919),  FML(0.4093614),   FML(0.5487071),   FML(0.5960872),   FML(0.7372317),   FML(0.7479271),   FML(0.7239134),
    FML(0.0858312),   FML(0.7806431),   FML(0.0775957),   FML(0.3178113),   FML(0.5915392),   FML(0.0070315),  FML(0.6048925),   FML(0.6143809),   FML(0.1979820),   FML(0.1774743),   FML(0.7768829),   FML(0.0300902),
    FML(0.0608814),   FML(0.7064558),   FML(0.3450395),   FML(0.2851639),   FML(0.2730330),   FML(0.4429390),  FML(0.0514124),   FML(0.4504313),   FML(0.8197787),   FML(0.8588904),   FML(0.0883032),   FML(0.4007187),
    FML(0.4686521),   FML(0.4688672),   FML(0.6918641),   FML(0.9536455),   FML(0.6249292),   FML(0.3588262),  FML(0.3547659),   FML(0.6072399),   FML(0.7702174),   FML(0.5338083),   FML(0.8088898),   FML(0.7171097),
    FML(0.8926089),   FML(0.2829780),   FML(0.1888959),   FML(0.8862076),   FML(0.5423330),   FML(0.5224111),  FML(0.1392960),   FML(0.3647683),   FML(0.5383290),   FML(0.6143245),   FML(0.5489270),   FML(0.4629686),
    FML(0.6254033),   FML(0.7067338),   FML(0.4435206),   FML(0.5298766),   FML(0.4160238),   FML(0.9795911),  FML(0.8566623),   FML(0.2453691),   FML(0.4547552),   FML(0.9750512),   FML(0.5568966),   FML(0.6019196),
    FML(0.3078349),   FML(0.4105948),   FML(0.3188829),   FML(0.9884667),   FML(0.2671129),   FML(0.1857880),  FML(0.2653089),   FML(0.7535607),   FML(0.2697969),   FML(0.1964178),   FML(0.5261393),   FML(0.2806930),
    };
    u32 inputRows = 7;
    u32 inputCols = 6;

    fml correctOutput_k33[] =
    {  FML(1.6547),   FML(2.3143),   FML(3.3408),   FML(3.5300),   FML(3.3115),   FML(2.3543),
   FML(2.2298),   FML(4.0960),   FML(4.3338),   FML(5.2779),   FML(5.2262),   FML(3.8555),
   FML(3.6542),   FML(4.6421),   FML(4.1434),   FML(5.1402),   FML(5.1013),   FML(4.0324),
   FML(3.3983),   FML(5.1284),   FML(4.6616),   FML(4.9019),   FML(4.9263),   FML(4.5505),
   FML(3.5052),   FML(6.2263),   FML(5.8304),   FML(5.8461),   FML(5.7236),   FML(4.7748),
   FML(3.2522),   FML(5.6275),   FML(4.8217),   FML(4.6446),   FML(4.4060),   FML(3.7769),
   FML(1.6921),   FML(3.1784),   FML(3.3184),   FML(2.6335),   FML(3.5555),   FML(2.3026),
    };

    fml correctOutput_k55[] =
     {  FML(3.4525),    FML(4.2845),    FML(5.4508),    FML(6.7003),    FML(5.4980),    FML(4.3194),
    FML(5.3872),    FML(6.8043),    FML(9.4618),    FML(9.1318),    FML(8.0428),    FML(7.1626),
    FML(5.5401),    FML(8.3237),    FML(9.8145),   FML(10.9154),    FML(9.0070),    FML(8.2713),
    FML(5.7009),    FML(8.6063),   FML(10.4616),   FML(10.3477),    FML(9.2506),    FML(7.5845),
    FML(6.6176),    FML(8.9669),   FML(11.9395),   FML(11.2950),    FML(9.9067),    FML(8.0055),
    FML(5.5947),    FML(7.9619),   FML(10.4714),    FML(9.6458),    FML(8.0194),    FML(7.5901),
    FML(4.5568),    FML(6.6643),    FML(8.4372),    FML(7.9155),    FML(6.7189),    FML(6.3353),
    };

    fml correctOutput_k57[] =
     {  FML(5.7547),    FML(7.5078),    FML(9.1416),    FML(7.9350),    FML(7.9643),    FML(6.7020),
    FML(7.3494),   FML(11.4024),   FML(11.7968),   FML(11.6545),   FML(10.7954),    FML(9.0589),
   FML(10.6178),   FML(14.3253),   FML(14.8530),   FML(14.3457),   FML(12.8368),    FML(9.7347),
    FML(9.8714),   FML(13.9456),   FML(16.0117),   FML(14.0833),   FML(12.7742),   FML(10.8775),
   FML(10.5091),   FML(14.1318),   FML(16.9131),   FML(14.6758),   FML(12.1619),    FML(9.9514),
    FML(9.1608),   FML(12.3111),   FML(14.8649),   FML(13.2128),   FML(10.7939),    FML(8.2075),
    FML(7.5258),    FML(9.3524),   FML(10.8391),    FML(8.4222),    FML(7.9152),    FML(5.6589),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test135(const tTest& t)
{
    fml input[] =
      {  FML(5.7184e-01),   FML(3.3551e-01),   FML(7.7062e-01),   FML(3.8881e-01),   FML(2.3902e-01),   FML(2.2509e-01),  FML(9.6592e-01),   FML(6.5993e-01),   FML(9.9442e-01),   FML(3.5779e-01),   FML(1.5854e-01),   FML(2.4524e-01),  FML(4.3041e-01),   FML(8.4799e-01),
     FML(8.2023e-01),   FML(7.9798e-01),   FML(4.0444e-01),   FML(3.8754e-01),   FML(3.3158e-01),   FML(2.2111e-01),  FML(4.5057e-01),   FML(3.6995e-01),   FML(3.3376e-01),   FML(1.8085e-01),   FML(3.1993e-01),   FML(8.9497e-01),  FML(4.1343e-01),   FML(2.2374e-01),
     FML(5.5070e-01),   FML(3.0346e-01),   FML(8.9064e-01),   FML(6.0831e-01),   FML(3.8533e-01),   FML(7.9148e-01),  FML(5.3577e-01),   FML(8.1532e-01),   FML(2.2200e-01),   FML(6.2333e-01),   FML(4.3120e-01),   FML(4.7254e-01),  FML(7.4983e-01),   FML(3.5868e-01),
     FML(1.4939e-01),   FML(9.2017e-01),   FML(6.1307e-01),   FML(4.1954e-01),   FML(4.8385e-01),   FML(5.1160e-01),  FML(3.1879e-02),   FML(7.0326e-01),   FML(1.1403e-01),   FML(6.0900e-01),   FML(6.3740e-01),   FML(4.0043e-01),  FML(7.3631e-01),   FML(8.7260e-01),
     FML(9.5379e-01),   FML(1.6526e-01),   FML(8.9411e-01),   FML(1.7781e-01),   FML(8.4451e-01),   FML(6.7615e-01),  FML(7.3915e-01),   FML(2.9695e-01),   FML(7.2480e-01),   FML(4.5969e-01),   FML(2.1757e-01),   FML(4.1232e-01),  FML(3.0170e-01),   FML(9.9788e-01),
     FML(5.7260e-01),   FML(5.9377e-01),   FML(7.5413e-01),   FML(6.4607e-02),   FML(5.6524e-01),   FML(1.1403e-02),  FML(4.9470e-01),   FML(8.5986e-01),   FML(8.7028e-01),   FML(8.2422e-01),   FML(8.1968e-01),   FML(5.7300e-01),  FML(8.8178e-01),   FML(6.9256e-01),
     FML(4.1209e-01),   FML(8.1168e-01),   FML(8.5844e-01),   FML(1.2212e-01),   FML(9.1946e-01),   FML(5.4985e-01),  FML(2.1304e-01),   FML(5.6439e-01),   FML(9.6220e-01),   FML(5.6431e-04),   FML(6.4146e-01),   FML(1.2532e-02),  FML(2.8149e-01),   FML(3.1589e-01),
    };
    u32 inputRows = 7;
    u32 inputCols = 7;

    fml correctOutput_k33[] =
    {  FML(2.6776),   FML(3.3574),   FML(3.1458),   FML(2.7551),   FML(3.6702),   FML(3.3439),   FML(2.0402),
   FML(3.4180),   FML(5.7127),   FML(5.7901),   FML(4.8086),   FML(5.1824),   FML(4.9709),   FML(3.7999),
   FML(3.6665),   FML(5.1891),   FML(4.9313),   FML(4.3647),   FML(4.6404),   FML(5.3039),   FML(4.0701),
   FML(3.2889),   FML(6.1598),   FML(6.4679),   FML(5.8801),   FML(5.1816),   FML(4.9907),   FML(3.8537),
   FML(3.5649),   FML(4.6515),   FML(5.4193),   FML(6.3897),   FML(5.7743),   FML(6.1869),   FML(4.9380),
   FML(3.1451),   FML(5.7485),   FML(5.5923),   FML(5.5847),   FML(5.1367),   FML(5.6180),   FML(3.9733),
   FML(1.7042),   FML(2.7307),   FML(2.6223),   FML(3.7033),   FML(3.3391),   FML(3.4024),   FML(2.6815),
    };

    fml correctOutput_k55[] =
     {  FML(3.7225),    FML(5.3188),    FML(6.2806),    FML(6.0827),    FML(5.7017),    FML(5.3370),    FML(4.0392),
    FML(5.3912),    FML(8.2467),    FML(9.8639),    FML(9.0846),    FML(9.7046),    FML(8.3274),    FML(6.6227),
    FML(6.6176),    FML(9.4140),   FML(10.9873),   FML(11.1001),   FML(10.7600),    FML(8.8835),    FML(7.7236),
    FML(6.8200),    FML(9.4256),   FML(11.0777),   FML(11.4953),   FML(11.9566),    FML(9.7231),    FML(8.1924),
    FML(6.1453),    FML(8.5283),   FML(11.0243),   FML(11.0791),   FML(10.7013),    FML(9.1424),    FML(8.5086),
    FML(5.9453),    FML(8.2580),   FML(10.2844),    FML(9.4830),   FML(10.0122),    FML(8.8088),    FML(7.4669),
    FML(4.7348),    FML(6.7135),    FML(8.1440),    FML(8.1471),    FML(8.7378),    FML(6.8372),    FML(6.7123),
    };

    fml correctOutput_k57[] =
     {  FML(6.9462),    FML(8.2549),    FML(9.5516),   FML(11.8256),    FML(9.9304),    FML(7.5014),    FML(6.9779),
    FML(9.5917),   FML(10.3527),   FML(12.3960),   FML(14.7094),   FML(10.9011),   FML(10.4445),    FML(8.9147),
   FML(12.7666),   FML(15.5107),   FML(16.5405),   FML(18.3879),   FML(14.6953),   FML(12.7522),   FML(10.5476),
   FML(12.3140),   FML(13.3651),   FML(15.9431),   FML(18.7159),   FML(15.5626),   FML(13.3155),   FML(10.2846),
   FML(13.0795),   FML(15.5432),   FML(17.3026),   FML(21.2908),   FML(15.6939),   FML(13.4852),   FML(10.3778),
    FML(9.5879),   FML(12.8671),   FML(13.4230),   FML(16.3404),   FML(12.9176),   FML(10.8540),    FML(9.0268),
    FML(9.1031),   FML(10.5723),   FML(10.2131),   FML(12.2986),   FML(11.2229),    FML(8.7773),    FML(6.5089),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test136(const tTest& t)
{
    fml input[] =
      {  FML(0.3287286),   FML(0.8897324),   FML(0.1038683),   FML(0.1457825),   FML(0.8746950),   FML(0.3143893),  FML(0.5980132),   FML(0.8397281),   FML(0.9175915),   FML(0.1958516),   FML(0.5137353),   FML(0.2389795),  FML(0.9892917),   FML(0.2601638),   FML(0.0014361),   FML(0.8838970),
     FML(0.7234251),   FML(0.1773659),   FML(0.4671671),   FML(0.4353073),   FML(0.1697479),   FML(0.8908080),  FML(0.8549170),   FML(0.7436331),   FML(0.5625624),   FML(0.5667478),   FML(0.6512397),   FML(0.4624243),  FML(0.3534858),   FML(0.0023686),   FML(0.8127217),   FML(0.8575900),
     FML(0.1332090),   FML(0.1151396),   FML(0.5261102),   FML(0.9743773),   FML(0.6912397),   FML(0.7006647),  FML(0.6747461),   FML(0.1878007),   FML(0.8423566),   FML(0.0916160),   FML(0.6056457),   FML(0.8295492),  FML(0.6885479),   FML(0.3006800),   FML(0.9696342),   FML(0.0564159),
     FML(0.2716646),   FML(0.8203685),   FML(0.9708530),   FML(0.1519213),   FML(0.2863579),   FML(0.0782612),  FML(0.6903046),   FML(0.5132868),   FML(0.9700395),   FML(0.2637750),   FML(0.4396533),   FML(0.2996278),  FML(0.2810539),   FML(0.8851779),   FML(0.6067673),   FML(0.9810114),
     FML(0.5247843),   FML(0.1984696),   FML(0.4756581),   FML(0.1210879),   FML(0.0229701),   FML(0.0182981),  FML(0.1936739),   FML(0.6732820),   FML(0.4021302),   FML(0.3764212),   FML(0.7490707),   FML(0.2282904),  FML(0.2733526),   FML(0.6939743),   FML(0.1135159),   FML(0.5047825),
     FML(0.0476258),   FML(0.3164077),   FML(0.4846884),   FML(0.7420620),   FML(0.4214444),   FML(0.0044454),  FML(0.3276694),   FML(0.9879245),   FML(0.4843530),   FML(0.5997926),   FML(0.1584548),   FML(0.5757166),  FML(0.4432274),   FML(0.6738279),   FML(0.4916771),   FML(0.0604678),
     FML(0.7550247),   FML(0.3933718),   FML(0.2146525),   FML(0.8340832),   FML(0.0036209),   FML(0.2151806),  FML(0.8505774),   FML(0.5394368),   FML(0.6722720),   FML(0.8363393),   FML(0.3403032),   FML(0.5619332),  FML(0.2896529),   FML(0.5699523),   FML(0.4188451),   FML(0.1169359),
    };
    u32 inputRows = 7;
    u32 inputCols = 8;

    fml correctOutput_k33[] =
    {  FML(2.0783),   FML(3.7936),   FML(3.7590),   FML(4.4261),   FML(4.5313),   FML(3.3780),   FML(3.4990),   FML(2.9580),
   FML(3.0801),   FML(4.8229),   FML(5.2134),   FML(6.0076),   FML(5.8296),   FML(5.6615),   FML(5.2394),   FML(4.2379),
   FML(2.9488),   FML(4.4354),   FML(6.0044),   FML(5.6188),   FML(5.6360),   FML(5.3378),   FML(5.9407),   FML(4.2656),
   FML(2.5695),   FML(3.9450),   FML(4.4384),   FML(3.9352),   FML(4.7550),   FML(5.3708),   FML(4.8818),   FML(3.5376),
   FML(2.9840),   FML(3.3808),   FML(4.3581),   FML(4.2102),   FML(4.9746),   FML(5.1486),   FML(4.8968),   FML(3.8748),
   FML(2.7299),   FML(3.6412),   FML(4.1066),   FML(4.4239),   FML(5.5463),   FML(4.9867),   FML(4.6586),   FML(3.1036),
   FML(1.3548),   FML(2.3863),   FML(2.7250),   FML(2.7402),   FML(3.3673),   FML(3.3112),   FML(2.4950),   FML(1.8564),
    };

    fml correctOutput_k55[] =
     {  FML(3.8911),    FML(6.0707),    FML(6.3589),    FML(5.7713),    FML(6.8504),    FML(6.9548),    FML(4.7793),    FML(3.8564),
    FML(5.4998),    FML(6.8833),    FML(9.7252),   FML(10.1469),   FML(10.6563),   FML(10.1868),    FML(8.5617),    FML(6.6862),
    FML(5.3175),    FML(8.0787),   FML(10.3667),   FML(11.2585),   FML(10.9872),   FML(11.5356),    FML(9.2696),    FML(7.5311),
    FML(5.3332),    FML(7.3248),   FML(10.2044),   FML(11.5253),   FML(10.7577),   FML(11.7326),    FML(9.6816),    FML(8.4063),
    FML(4.4877),    FML(8.2510),   FML(10.3577),   FML(10.3566),   FML(10.3034),   FML(11.3572),    FML(9.0608),    FML(7.7336),
    FML(3.7648),    FML(7.1611),    FML(7.2783),    FML(7.9698),    FML(8.6259),    FML(8.8247),    FML(7.6922),    FML(6.8510),
    FML(3.1426),    FML(4.8737),    FML(6.6171),    FML(6.4751),    FML(7.2952),    FML(7.6862),    FML(6.3107),    FML(5.2809),
    };

    fml correctOutput_k57[] =
     {  FML(6.8858),    FML(8.8233),    FML(9.9920),   FML(11.1773),   FML(12.6683),   FML(12.1158),    FML(8.9051),    FML(7.4874),
   FML(10.1184),   FML(11.9617),   FML(13.1937),   FML(14.7302),   FML(16.2681),   FML(12.2298),   FML(10.6991),    FML(9.0575),
   FML(10.5290),   FML(12.4389),   FML(16.0065),   FML(17.5627),   FML(17.6908),   FML(14.0875),   FML(14.0752),    FML(9.5609),
   FML(10.6471),   FML(12.9609),   FML(15.2612),   FML(16.8082),   FML(18.0966),   FML(14.7120),   FML(12.8645),   FML(10.7199),
   FML(11.0376),   FML(13.7077),   FML(12.6645),   FML(17.1653),   FML(18.7453),   FML(12.5160),   FML(12.5939),   FML(10.4985),
    FML(8.0144),   FML(10.3954),   FML(12.2330),   FML(13.5482),   FML(13.6020),   FML(10.6370),   FML(10.1279),    FML(8.0305),
    FML(6.4132),    FML(8.0422),    FML(9.6565),   FML(10.2320),    FML(9.4427),    FML(8.7534),    FML(7.3803),    FML(5.3547),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test137(const tTest& t)
{
    fml input[] =
    {  FML(0.9309879),   FML(0.8764251),
   FML(0.4987210),   FML(0.9179333),
   FML(0.2377653),   FML(0.1238935),
   FML(0.1247597),   FML(0.0729347),
   FML(0.1756339),   FML(0.1829099),
   FML(0.0046785),   FML(0.0887603),
   FML(0.5397796),   FML(0.3772563),
   FML(0.0509277),   FML(0.8600020),
    };
    u32 inputRows = 8;
    u32 inputCols = 1;

    fml correctOutput_k33[] =
    {  FML(1.84011),
   FML(1.90893),
   FML(1.48469),
   FML(0.78874),
   FML(0.56870),
   FML(1.23009),
   FML(1.36059),
   FML(1.02139),
    };

    fml correctOutput_k55[] =
    {  FML(2.1320),
   FML(2.9026),
   FML(3.1184),
   FML(2.1744),
   FML(1.7313),
   FML(2.0437),
   FML(1.8270),
   FML(2.1068),
    };

    fml correctOutput_k57[] =
    {  FML(2.7803),
   FML(2.5506),
   FML(3.1065),
   FML(2.0249),
   FML(2.0634),
   FML(2.0970),
   FML(2.1665),
   FML(1.6023),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test138(const tTest& t)
{
    fml input[] =
    {  FML(0.7339305),   FML(0.3872899),   FML(0.8914217),   FML(0.0370066),
   FML(0.7607277),   FML(0.9264054),   FML(0.8795166),   FML(0.0051467),
   FML(0.6635081),   FML(0.0512477),   FML(0.2479868),   FML(0.0714974),
   FML(0.5221002),   FML(0.3286939),   FML(0.1384088),   FML(0.3723627),
   FML(0.8511522),   FML(0.3695398),   FML(0.3854161),   FML(0.5288354),
   FML(0.6411519),   FML(0.6281997),   FML(0.4236930),   FML(0.7892983),
   FML(0.2968695),   FML(0.0899147),   FML(0.1101307),   FML(0.1818021),
   FML(0.5583437),   FML(0.1744718),   FML(0.5668939),   FML(0.1725936),
    };
    u32 inputRows = 8;
    u32 inputCols = 2;

    fml correctOutput_k33[] =
    {  FML(2.89269),   FML(2.86149),
   FML(2.40078),   FML(3.46096),
   FML(2.71741),   FML(3.08800),
   FML(2.44290),   FML(3.36531),
   FML(3.26857),   FML(4.11151),
   FML(2.06640),   FML(3.51780),
   FML(2.67316),   FML(3.41346),
   FML(0.87911),   FML(1.44515),
    };

    fml correctOutput_k55[] =
    {  FML(2.3574),   FML(2.5784),
   FML(3.6455),   FML(4.8029),
   FML(4.5705),   FML(5.8155),
   FML(4.9445),   FML(5.6992),
   FML(3.6662),   FML(4.3298),
   FML(4.1980),   FML(5.1532),
   FML(3.3219),   FML(4.4799),
   FML(2.5947),   FML(3.1456),
    };

    fml correctOutput_k57[] =
    {  FML(4.2355),   FML(4.1140),
   FML(4.7203),   FML(4.7097),
   FML(5.6126),   FML(5.6443),
   FML(5.0807),   FML(5.1412),
   FML(4.4950),   FML(4.8168),
   FML(4.8042),   FML(5.1144),
   FML(4.1824),   FML(3.6450),
   FML(3.5760),   FML(2.9796),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test139(const tTest& t)
{
    fml input[] =
    {  FML(0.447846),   FML(0.271579),   FML(0.800608),   FML(0.628833),   FML(0.792346),   FML(0.229392),
   FML(0.813392),   FML(0.014767),   FML(0.767334),   FML(0.426128),   FML(0.478264),   FML(0.704823),
   FML(0.012300),   FML(0.034884),   FML(0.757974),   FML(0.220084),   FML(0.062455),   FML(0.622077),
   FML(0.367085),   FML(0.681447),   FML(0.194345),   FML(0.521632),   FML(0.016360),   FML(0.058053),
   FML(0.836097),   FML(0.236042),   FML(0.433995),   FML(0.123447),   FML(0.297407),   FML(0.468635),
   FML(0.951616),   FML(0.341532),   FML(0.032181),   FML(0.740720),   FML(0.298059),   FML(0.309637),
   FML(0.465058),   FML(0.824957),   FML(0.555324),   FML(0.955285),   FML(0.733056),   FML(0.545238),
   FML(0.188602),   FML(0.056537),   FML(0.643823),   FML(0.380587),   FML(0.463889),   FML(0.508999),
    };
    u32 inputRows = 8;
    u32 inputCols = 3;

    fml correctOutput_k33[] =
    {  FML(2.3380),   FML(3.9595),   FML(3.2236),
   FML(2.3015),   FML(4.0512),   FML(4.0681),
   FML(2.4471),   FML(3.4936),   FML(3.2423),
   FML(2.1487),   FML(3.7108),   FML(3.2667),
   FML(3.1992),   FML(4.5443),   FML(2.1369),
   FML(3.3723),   FML(5.8188),   FML(3.8056),
   FML(2.6864),   FML(5.5290),   FML(3.7764),
   FML(1.9243),   FML(2.9910),   FML(2.7851),
    };

    fml correctOutput_k55[] =
    {  FML(3.1089),   FML(3.6466),   FML(3.9582),
   FML(4.8371),   FML(4.7203),   FML(6.2364),
   FML(5.1222),   FML(6.4119),   FML(6.9458),
   FML(4.7156),   FML(5.8453),   FML(6.0187),
   FML(5.1279),   FML(6.4334),   FML(5.8255),
   FML(5.8968),   FML(6.8224),   FML(7.3902),
   FML(5.4271),   FML(5.8125),   FML(6.9509),
   FML(4.4689),   FML(4.9132),   FML(6.1895),
    };

    fml correctOutput_k57[] =
    {  FML(4.1318),   FML(4.7506),   FML(5.4090),
   FML(6.2771),   FML(5.9836),   FML(5.5204),
   FML(6.8830),   FML(6.5777),   FML(6.9499),
   FML(6.7243),   FML(6.2920),   FML(5.8533),
   FML(6.9600),   FML(7.8612),   FML(7.0939),
   FML(6.3765),   FML(6.9681),   FML(7.6017),
   FML(5.9383),   FML(6.5857),   FML(6.7965),
   FML(5.7018),   FML(5.3218),   FML(5.8508),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test140(const tTest& t)
{
    fml input[] =
     {  FML(6.4814e-01),   FML(1.6056e-01),   FML(1.2771e-01),   FML(8.3716e-01),   FML(8.4979e-01),   FML(4.4448e-02),  FML(6.5810e-01),   FML(1.9808e-01),
    FML(5.6962e-01),   FML(7.7263e-01),   FML(2.6921e-01),   FML(5.0512e-01),   FML(7.2194e-01),   FML(3.4144e-01),  FML(1.4582e-01),   FML(9.6562e-01),
    FML(5.1202e-01),   FML(5.7569e-02),   FML(7.0919e-01),   FML(5.8951e-01),   FML(9.6675e-01),   FML(7.4592e-01),  FML(8.7130e-02),   FML(6.5051e-02),
    FML(4.3237e-01),   FML(4.0421e-01),   FML(5.3190e-01),   FML(8.9877e-01),   FML(9.2115e-01),   FML(9.8150e-01),  FML(2.1500e-01),   FML(2.4432e-01),
    FML(3.5294e-01),   FML(8.4346e-01),   FML(6.4776e-01),   FML(6.0203e-01),   FML(2.6676e-01),   FML(3.0367e-01),  FML(2.9559e-02),   FML(2.7185e-01),
    FML(9.7447e-01),   FML(1.1309e-01),   FML(7.7917e-01),   FML(9.7019e-01),   FML(9.2012e-01),   FML(5.9410e-01),  FML(8.7311e-01),   FML(5.4752e-01),
    FML(1.8043e-01),   FML(6.6371e-01),   FML(7.9476e-02),   FML(6.5738e-01),   FML(9.6107e-01),   FML(9.9055e-01),  FML(3.3499e-04),   FML(9.5226e-02),
    FML(9.9363e-01),   FML(1.4672e-01),   FML(2.0317e-01),   FML(1.4468e-01),   FML(7.2330e-01),   FML(5.0222e-01),  FML(4.4480e-01),   FML(4.1763e-01),
    };
    u32 inputRows = 8;
    u32 inputCols = 4;

    fml correctOutput_k33[] =
    {  FML(2.2421),   FML(3.7094),   FML(3.5337),   FML(2.7212),
   FML(2.8610),   FML(6.2023),   FML(4.2715),   FML(3.6996),
   FML(3.5841),   FML(6.2261),   FML(5.5968),   FML(4.8160),
   FML(3.2134),   FML(5.2994),   FML(5.1404),   FML(3.6246),
   FML(3.9970),   FML(7.0970),   FML(6.8103),   FML(4.4453),
   FML(3.3086),   FML(5.9695),   FML(5.3684),   FML(3.5955),
   FML(2.4708),   FML(6.0820),   FML(5.3913),   FML(4.9256),
   FML(1.5361),   FML(3.0218),   FML(2.4622),   FML(2.7540),
    };

    fml correctOutput_k55[] =
     {  FML(3.8413),    FML(5.1170),    FML(4.9492),    FML(3.8234),
    FML(6.2683),    FML(7.9119),    FML(8.2600),    FML(6.6278),
    FML(7.0922),    FML(7.9472),    FML(9.2625),    FML(7.4739),
    FML(7.6739),    FML(9.3696),   FML(10.7382),    FML(8.7068),
    FML(7.2839),    FML(9.7289),   FML(10.4124),    FML(8.6337),
    FML(6.3396),    FML(8.6424),   FML(10.3957),    FML(8.1821),
    FML(6.3061),    FML(9.0562),    FML(8.8040),    FML(7.7707),
    FML(4.4842),    FML(5.0802),    FML(7.2847),    FML(5.9786),
    };

    fml correctOutput_k57[] =
     {  FML(5.5114),    FML(6.9482),    FML(7.0505),    FML(6.7711),
    FML(8.1536),    FML(9.5422),    FML(8.5716),    FML(7.8887),
   FML(10.8014),   FML(10.0134),   FML(10.3205),    FML(9.8419),
   FML(11.1892),   FML(11.9058),   FML(11.9739),   FML(11.7550),
   FML(10.9134),   FML(10.5560),   FML(10.8928),    FML(9.8036),
   FML(11.5962),   FML(11.6518),   FML(11.3630),   FML(11.2094),
    FML(8.6100),    FML(7.9444),    FML(8.9703),    FML(7.8680),
    FML(8.0955),    FML(7.9709),    FML(7.1938),    FML(7.3738),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test141(const tTest& t)
{
    fml input[] =
     {  FML(0.1174471),   FML(0.0491726),   FML(0.5360838),   FML(0.2338143),   FML(0.6835811),   FML(0.2231389),  FML(0.7286393),   FML(0.1750022),   FML(0.8171081),   FML(0.7729126),
    FML(0.1743623),   FML(0.7377903),   FML(0.6339119),   FML(0.7847393),   FML(0.8425838),   FML(0.7201381),  FML(0.2958354),   FML(0.1232607),   FML(0.2109669),   FML(0.7096591),
    FML(0.9152635),   FML(0.6648947),   FML(0.5295745),   FML(0.2700562),   FML(0.7923928),   FML(0.2795375),  FML(0.0471151),   FML(0.0626835),   FML(0.3686322),   FML(0.4558116),
    FML(0.0812190),   FML(0.4454094),   FML(0.3589255),   FML(0.9154153),   FML(0.3145142),   FML(0.2485998),  FML(0.0063162),   FML(0.2195177),   FML(0.0686083),   FML(0.3162059),
    FML(0.1574295),   FML(0.3620538),   FML(0.2675673),   FML(0.9846180),   FML(0.5344579),   FML(0.1271663),  FML(0.9585057),   FML(0.8961989),   FML(0.5991765),   FML(0.4413141),
    FML(0.6835144),   FML(0.4783788),   FML(0.1978258),   FML(0.7954950),   FML(0.4864704),   FML(0.4981699),  FML(0.6070209),   FML(0.1630906),   FML(0.0477677),   FML(0.2216157),
    FML(0.5611259),   FML(0.4747233),   FML(0.2478722),   FML(0.0456438),   FML(0.1268797),   FML(0.0207109),  FML(0.0674895),   FML(0.7075874),   FML(0.5324109),   FML(0.5367820),
    FML(0.6813624),   FML(0.7259337),   FML(0.9800450),   FML(0.8970432),   FML(0.2608445),   FML(0.8391401),  FML(0.3260012),   FML(0.1402354),   FML(0.1838871),   FML(0.9769675),
    };
    u32 inputRows = 8;
    u32 inputCols = 5;

    fml correctOutput_k33[] =
    {  FML(2.4731),   FML(3.6912),   FML(3.6553),   FML(3.4881),   FML(2.3054),
   FML(2.9086),   FML(4.7988),   FML(4.2757),   FML(4.5938),   FML(3.1592),
   FML(3.4951),   FML(5.2978),   FML(4.2292),   FML(3.6649),   FML(2.0361),
   FML(3.0494),   FML(4.7199),   FML(5.0606),   FML(4.7750),   FML(3.1278),
   FML(3.1506),   FML(4.9535),   FML(4.6922),   FML(3.5407),   FML(3.0056),
   FML(2.5155),   FML(3.9627),   FML(3.8284),   FML(4.6427),   FML(3.8638),
   FML(4.2055),   FML(6.2588),   FML(4.3245),   FML(3.7507),   FML(3.2633),
   FML(1.5999),   FML(2.6608),   FML(2.5714),   FML(2.3953),   FML(1.8980),
    };

    fml correctOutput_k55[] =
     {  FML(3.8873),    FML(4.0141),    FML(5.0689),    FML(4.7446),    FML(3.7634),
    FML(5.5688),    FML(6.6893),    FML(8.5386),    FML(7.2701),    FML(6.1145),
    FML(6.0106),    FML(7.7132),   FML(10.4769),    FML(8.8948),    FML(6.6872),
    FML(6.7691),    FML(9.6143),   FML(10.5939),    FML(7.8868),    FML(5.7244),
    FML(5.2958),    FML(7.3014),    FML(8.8990),    FML(7.3632),    FML(5.5916),
    FML(5.6163),    FML(8.9844),   FML(10.1583),    FML(7.8945),    FML(6.9412),
    FML(4.6495),    FML(7.0611),    FML(9.0905),    FML(7.0837),    FML(6.4996),
    FML(4.2985),    FML(5.6908),    FML(7.5932),    FML(5.6872),    FML(4.4911),
    };

    fml correctOutput_k57[] =
     {  FML(5.9292),    FML(8.1097),    FML(7.7131),    FML(8.1175),    FML(6.8623),
    FML(6.8038),   FML(10.1944),    FML(8.9042),    FML(8.3474),    FML(6.7536),
    FML(9.2813),   FML(13.7389),   FML(11.4886),   FML(11.8932),    FML(9.6827),
   FML(10.0763),   FML(12.7205),   FML(11.0075),   FML(10.3788),    FML(8.5479),
    FML(8.7660),   FML(10.2449),   FML(11.3131),    FML(9.6441),    FML(8.7526),
    FML(9.9867),   FML(12.2151),   FML(11.1679),   FML(10.5669),    FML(8.6235),
    FML(9.1297),   FML(10.9322),    FML(9.8572),    FML(9.7903),    FML(7.2010),
    FML(7.8064),    FML(8.0828),    FML(7.8477),    FML(6.7345),    FML(5.2829),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test142(const tTest& t)
{
    fml input[] =
     {  FML(0.6479722),   FML(0.0050898),   FML(0.9145420),   FML(0.8320855),   FML(0.3490256),   FML(0.2131240),  FML(0.7978667),   FML(0.2919056),   FML(0.1491697),   FML(0.2296099),   FML(0.9061480),   FML(0.7798139),
    FML(0.7735448),   FML(0.7921682),   FML(0.1157976),   FML(0.5700542),   FML(0.7259285),   FML(0.5494189),  FML(0.6264677),   FML(0.1142317),   FML(0.1340237),   FML(0.8571387),   FML(0.9527654),   FML(0.1179603),
    FML(0.9238461),   FML(0.0590407),   FML(0.5247678),   FML(0.8962817),   FML(0.8514310),   FML(0.1239982),  FML(0.0693601),   FML(0.8963178),   FML(0.9461992),   FML(0.6021507),   FML(0.4042497),   FML(0.2873781),
    FML(0.6192877),   FML(0.8166263),   FML(0.3770142),   FML(0.0616419),   FML(0.6360734),   FML(0.4588986),  FML(0.3684406),   FML(0.5241618),   FML(0.0285759),   FML(0.4932454),   FML(0.3163443),   FML(0.1900441),
    FML(0.4599539),   FML(0.7950079),   FML(0.8953123),   FML(0.8708785),   FML(0.8969838),   FML(0.8143327),  FML(0.8851332),   FML(0.8302579),   FML(0.5140395),   FML(0.1491665),   FML(0.8698042),   FML(0.6178247),
    FML(0.0608057),   FML(0.2822471),   FML(0.9336732),   FML(0.0903955),   FML(0.6906229),   FML(0.7644800),  FML(0.9975438),   FML(0.2617335),   FML(0.5116078),   FML(0.7285861),   FML(0.5203575),   FML(0.7517502),
    FML(0.3316341),   FML(0.9908309),   FML(0.8523059),   FML(0.1912840),   FML(0.1556309),   FML(0.6177282),  FML(0.1025808),   FML(0.7999560),   FML(0.2852351),   FML(0.3704255),   FML(0.8365932),   FML(0.4368881),
    FML(0.9898537),   FML(0.5486924),   FML(0.3878829),   FML(0.8366498),   FML(0.5705617),   FML(0.5026977),  FML(0.6400548),   FML(0.5754522),   FML(0.0861908),   FML(0.5142888),   FML(0.4449385),   FML(0.3389047),
    };
    u32 inputRows = 8;
    u32 inputCols = 6;

    fml correctOutput_k33[] =
    {  FML(2.5320),   FML(3.8811),   FML(3.8042),   FML(3.1774),   FML(3.7548),   FML(2.1829),
   FML(3.2067),   FML(6.2590),   FML(5.3413),   FML(5.2171),   FML(5.1388),   FML(3.9119),
   FML(3.3001),   FML(5.6504),   FML(4.5789),   FML(4.9239),   FML(4.5542),   FML(2.8552),
   FML(3.9060),   FML(7.1287),   FML(6.3752),   FML(6.3801),   FML(5.3782),   FML(3.9057),
   FML(3.0593),   FML(5.3628),   FML(6.4224),   FML(6.4477),   FML(6.0776),   FML(3.2730),
   FML(3.9798),   FML(5.3100),   FML(6.4453),   FML(5.6934),   FML(5.4471),   FML(4.0759),
   FML(3.5727),   FML(5.2991),   FML(5.8051),   FML(5.1140),   FML(5.5706),   FML(3.4979),
   FML(2.0057),   FML(3.1802),   FML(3.4828),   FML(2.7417),   FML(2.6004),   FML(1.8879),
    };

    fml correctOutput_k55[] =
     {  FML(4.2535),    FML(4.8596),    FML(6.1876),    FML(6.8430),    FML(4.8004),    FML(3.8900),
    FML(5.3065),    FML(8.2433),    FML(9.6342),    FML(9.8471),    FML(6.7053),    FML(6.6482),
    FML(7.5271),    FML(9.9093),   FML(12.0167),   FML(11.1352),    FML(9.0388),    FML(7.9281),
    FML(7.3279),    FML(9.8776),   FML(12.9192),   FML(11.1240),   FML(10.1873),    FML(8.4099),
    FML(7.3577),    FML(9.5101),   FML(11.9565),   FML(11.1261),   FML(10.5509),    FML(7.2428),
    FML(7.5478),   FML(10.3962),   FML(11.8168),   FML(12.9438),   FML(10.1959),    FML(8.4082),
    FML(5.6289),    FML(9.3752),   FML(10.3055),   FML(11.6886),    FML(9.2070),    FML(7.5397),
    FML(3.9858),    FML(6.6292),    FML(8.0493),    FML(8.3040),    FML(6.2400),    FML(6.1965),
    };

    fml correctOutput_k57[] =
     {  FML(6.7544),    FML(8.5471),   FML(11.1779),    FML(9.7107),    FML(8.8375),    FML(7.4325),
    FML(8.4048),   FML(11.0253),   FML(12.9818),   FML(10.9490),   FML(10.2014),    FML(7.7923),
   FML(12.4114),   FML(15.7834),   FML(17.0428),   FML(16.3096),   FML(13.1057),   FML(11.1924),
   FML(13.0119),   FML(14.6313),   FML(18.1369),   FML(15.3905),   FML(13.9337),   FML(10.7159),
   FML(13.0000),   FML(14.7404),   FML(16.8800),   FML(15.1814),   FML(13.6250),   FML(10.7308),
   FML(13.3867),   FML(14.2586),   FML(18.4321),   FML(16.4512),   FML(13.2763),   FML(10.1721),
   FML(12.3144),   FML(12.6970),   FML(14.8570),   FML(12.7880),   FML(11.6305),    FML(9.3579),
    FML(8.6134),    FML(9.7289),   FML(11.2756),   FML(10.0473),    FML(8.4684),    FML(5.8764),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test143(const tTest& t)
{
    fml input[] =
     {  FML(0.108213),   FML(0.547488),   FML(0.752061),   FML(0.652230),   FML(0.671781),   FML(0.259718),   FML(0.933649),  FML(0.191397),   FML(0.558984),   FML(0.567240),   FML(0.234724),   FML(0.641292),   FML(0.416100),   FML(0.666628),
    FML(0.737467),   FML(0.389480),   FML(0.071919),   FML(0.923812),   FML(0.440343),   FML(0.146761),   FML(0.692341),  FML(0.529935),   FML(0.571259),   FML(0.426775),   FML(0.625177),   FML(0.247698),   FML(0.923647),   FML(0.523765),
    FML(0.479082),   FML(0.927943),   FML(0.465577),   FML(0.026451),   FML(0.020033),   FML(0.442281),   FML(0.315648),  FML(0.075095),   FML(0.866174),   FML(0.176653),   FML(0.045896),   FML(0.715413),   FML(0.062582),   FML(0.210334),
    FML(0.611169),   FML(0.016159),   FML(0.736017),   FML(0.746616),   FML(0.765979),   FML(0.314717),   FML(0.305119),  FML(0.571309),   FML(0.675194),   FML(0.779506),   FML(0.344947),   FML(0.789838),   FML(0.726885),   FML(0.312332),
    FML(0.271620),   FML(0.036198),   FML(0.115384),   FML(0.020721),   FML(0.780534),   FML(0.901964),   FML(0.307162),  FML(0.669678),   FML(0.704189),   FML(0.085397),   FML(0.950591),   FML(0.373679),   FML(0.474569),   FML(0.355302),
    FML(0.489291),   FML(0.859658),   FML(0.174218),   FML(0.240628),   FML(0.161529),   FML(0.136322),   FML(0.852143),  FML(0.053910),   FML(0.757716),   FML(0.311938),   FML(0.713455),   FML(0.392868),   FML(0.927574),   FML(0.568020),
    FML(0.631414),   FML(0.791768),   FML(0.961919),   FML(0.086020),   FML(0.242594),   FML(0.730569),   FML(0.647447),  FML(0.875377),   FML(0.500526),   FML(0.018575),   FML(0.133437),   FML(0.118742),   FML(0.388908),   FML(0.848580),
    FML(0.882700),   FML(0.215937),   FML(0.748377),   FML(0.567088),   FML(0.295346),   FML(0.358230),   FML(0.011045),  FML(0.180971),   FML(0.281431),   FML(0.525684),   FML(0.684623),   FML(0.100111),   FML(0.400964),   FML(0.135154),
    };
    u32 inputRows = 8;
    u32 inputCols = 7;

    fml correctOutput_k33[] =
    {  FML(2.4381),   FML(3.4481),   FML(3.6659),   FML(3.5410),   FML(3.7081),   FML(3.9974),   FML(2.9005),
   FML(3.0614),   FML(4.3052),   FML(4.0877),   FML(3.9365),   FML(5.1344),   FML(4.9172),   FML(2.9473),
   FML(3.3771),   FML(5.8459),   FML(4.3371),   FML(5.0723),   FML(4.9604),   FML(5.3356),   FML(3.6615),
   FML(2.0629),   FML(4.2154),   FML(4.8282),   FML(4.5270),   FML(4.5644),   FML(5.8179),   FML(3.4681),
   FML(2.3477),   FML(3.8489),   FML(3.8948),   FML(5.5327),   FML(5.6797),   FML(6.1325),   FML(4.3227),
   FML(2.8065),   FML(4.6253),   FML(5.2852),   FML(5.0224),   FML(3.9226),   FML(4.6710),   FML(3.9786),
   FML(3.8786),   FML(5.1318),   FML(3.6092),   FML(3.4089),   FML(4.3875),   FML(4.0480),   FML(3.3448),
   FML(1.8277),   FML(2.9347),   FML(3.4948),   FML(2.4874),   FML(2.0419),   FML(2.1765),   FML(1.9611),
    };

    fml correctOutput_k55[] =
     {  FML(3.4876),    FML(4.9753),    FML(5.4489),    FML(5.5061),    FML(5.6015),    FML(5.1140),    FML(4.1866),
    FML(5.0885),    FML(7.4618),    FML(9.9104),    FML(8.9300),    FML(9.0357),    FML(8.0945),    FML(5.8972),
    FML(5.2071),    FML(8.7463),   FML(10.7361),   FML(10.3079),    FML(9.9903),    FML(8.8680),    FML(8.3067),
    FML(5.4742),    FML(7.2985),    FML(9.4516),    FML(9.4129),    FML(9.8973),    FML(9.0134),    FML(7.8573),
    FML(5.9446),    FML(8.1695),   FML(10.2173),    FML(9.6751),   FML(10.6517),    FML(8.1861),    FML(7.7124),
    FML(5.8662),    FML(7.5204),    FML(9.6866),    FML(9.2791),   FML(10.4278),    FML(8.7733),    FML(7.7123),
    FML(4.2297),    FML(6.6869),    FML(8.3600),    FML(8.5154),    FML(8.9662),    FML(7.4977),    FML(6.8630),
    FML(4.3136),    FML(6.4286),    FML(7.0156),    FML(6.8943),    FML(6.6040),    FML(5.3707),    FML(4.7228),
    };

    fml correctOutput_k57[] =
     {  FML(5.7823),    FML(7.9899),    FML(9.8391),   FML(10.2342),    FML(8.9696),    FML(7.4202),    FML(5.8737),
    FML(8.4426),   FML(11.4811),   FML(11.9783),   FML(13.9171),   FML(10.6433),   FML(10.6121),    FML(8.7407),
    FML(9.8531),   FML(12.6066),   FML(14.8743),   FML(18.5120),   FML(13.5675),   FML(12.3454),    FML(9.7182),
   FML(10.2932),   FML(13.2263),   FML(15.4310),   FML(16.6960),   FML(13.2363),   FML(12.4684),   FML(10.9000),
   FML(10.4111),   FML(12.3624),   FML(14.5969),   FML(16.9065),   FML(13.2833),   FML(11.2684),    FML(9.4798),
   FML(10.3831),   FML(12.2989),   FML(13.9278),   FML(18.6413),   FML(14.1889),   FML(11.4163),   FML(10.3496),
    FML(9.2891),   FML(10.9362),   FML(12.1807),   FML(12.8388),   FML(11.5403),    FML(8.7408),    FML(6.6603),
    FML(7.8496),    FML(7.7426),    FML(9.5427),   FML(10.5354),    FML(7.7964),    FML(6.6076),    FML(5.2235),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


void test144(const tTest& t)
{
    fml input[] =
      {  FML(0.038513),   FML(0.987480),   FML(0.372836),   FML(0.862104),   FML(0.064520),   FML(0.222306),   FML(0.204276),  FML(0.143618),   FML(0.401578),   FML(0.262270),   FML(0.814799),   FML(0.124541),   FML(0.205079),   FML(0.592483),  FML(0.172371),   FML(0.088782),
     FML(0.347269),   FML(0.980642),   FML(0.213071),   FML(0.375460),   FML(0.414619),   FML(0.867204),   FML(0.110290),  FML(0.682969),   FML(0.251021),   FML(0.083867),   FML(0.099981),   FML(0.368089),   FML(0.404040),   FML(0.492542),  FML(0.373708),   FML(0.788353),
     FML(0.722120),   FML(0.216007),   FML(0.786049),   FML(0.459809),   FML(0.770561),   FML(0.782852),   FML(0.291083),  FML(0.430205),   FML(0.679763),   FML(0.295356),   FML(0.050205),   FML(0.059266),   FML(0.885263),   FML(0.510471),  FML(0.281186),   FML(0.450273),
     FML(0.891268),   FML(0.691647),   FML(0.705752),   FML(0.521336),   FML(0.192735),   FML(0.591195),   FML(0.496736),  FML(0.339394),   FML(0.920939),   FML(0.106722),   FML(0.234950),   FML(0.743848),   FML(0.885313),   FML(0.642425),  FML(0.338031),   FML(0.503471),
     FML(0.780092),   FML(0.983957),   FML(0.215821),   FML(0.742230),   FML(0.541651),   FML(0.463810),   FML(0.652188),  FML(0.426651),   FML(0.417412),   FML(0.423468),   FML(0.864441),   FML(0.466003),   FML(0.989850),   FML(0.758690),  FML(0.184832),   FML(0.356247),
     FML(0.742955),   FML(0.862745),   FML(0.074809),   FML(0.155753),   FML(0.797805),   FML(0.254569),   FML(0.551565),  FML(0.129860),   FML(0.709579),   FML(0.114414),   FML(0.294760),   FML(0.373436),   FML(0.717654),   FML(0.165222),  FML(0.558968),   FML(0.551147),
     FML(0.253863),   FML(0.639777),   FML(0.452928),   FML(0.939968),   FML(0.729641),   FML(0.535978),   FML(0.666754),  FML(0.651658),   FML(0.159828),   FML(0.130818),   FML(0.607214),   FML(0.641919),   FML(0.175846),   FML(0.764214),  FML(0.891842),   FML(0.550374),
     FML(0.795263),   FML(0.837819),   FML(0.632538),   FML(0.470457),   FML(0.676606),   FML(0.238664),   FML(0.379584),  FML(0.437875),   FML(0.717894),   FML(0.428508),   FML(0.825738),   FML(0.971942),   FML(0.621622),   FML(0.999688),  FML(0.258616),   FML(0.694647),
    };
    u32 inputRows = 8;
    u32 inputCols = 8;

    fml correctOutput_k33[] =
    {  FML(2.3339),   FML(3.6546),   FML(3.3385),   FML(2.3967),   FML(1.9854),   FML(2.4070),   FML(3.0923),   FML(2.4000),
   FML(3.8262),   FML(5.9367),   FML(4.8073),   FML(4.3780),   FML(2.9723),   FML(3.8025),   FML(4.2122),   FML(3.3249),
   FML(4.1681),   FML(5.6713),   FML(5.3159),   FML(5.1827),   FML(3.8195),   FML(5.1049),   FML(4.3328),   FML(4.4984),
   FML(3.7115),   FML(6.2129),   FML(5.7106),   FML(5.2697),   FML(4.5497),   FML(5.7209),   FML(5.0950),   FML(4.6617),
   FML(3.3796),   FML(6.0598),   FML(4.6066),   FML(4.3738),   FML(4.4026),   FML(5.4998),   FML(5.1743),   FML(4.7880),
   FML(3.9966),   FML(6.5584),   FML(5.0388),   FML(4.9202),   FML(4.7335),   FML(4.7932),   FML(5.9862),   FML(4.3138),
   FML(3.9214),   FML(5.6531),   FML(4.5355),   FML(5.0259),   FML(5.3675),   FML(5.7359),   FML(5.9359),   FML(4.3649),
   FML(2.0250),   FML(3.6952),   FML(3.1998),   FML(3.0773),   FML(2.8485),   FML(2.9945),   FML(4.0080),   FML(2.8238),
    };

    fml correctOutput_k55[] =
     {  FML(4.3948),    FML(5.5538),    FML(6.2222),    FML(5.1515),    FML(4.4549),    FML(4.5682),    FML(3.6829),    FML(3.8458),
    FML(5.3479),    FML(7.6622),    FML(8.9177),    FML(7.6566),    FML(7.8538),    FML(7.6305),    FML(6.2448),    FML(5.8155),
    FML(6.7756),    FML(9.3461),   FML(10.1995),    FML(9.1599),    FML(9.8686),    FML(8.8094),    FML(7.0771),    FML(6.7164),
    FML(7.3485),    FML(9.5837),   FML(11.5327),   FML(10.6093),   FML(10.5904),    FML(8.3835),    FML(8.7483),    FML(7.7430),
    FML(7.5938),   FML(10.5371),   FML(12.4914),   FML(10.1386),   FML(10.6167),   FML(10.0768),    FML(9.3862),    FML(8.2264),
    FML(7.6898),   FML(10.1104),   FML(10.8477),   FML(10.7235),   FML(11.1021),   FML(10.8634),   FML(10.6199),    FML(8.4959),
    FML(6.4017),    FML(8.5911),   FML(10.1799),    FML(9.2067),    FML(9.7447),    FML(9.6903),    FML(8.6172),    FML(8.0195),
    FML(4.8064),    FML(6.6571),    FML(7.8112),    FML(8.0650),    FML(7.6065),    FML(7.8426),    FML(7.1592),    FML(7.4098),
    };

    fml correctOutput_k57[] =
     {  FML(5.9606),    FML(7.7235),    FML(7.7817),    FML(9.8144),    FML(9.5722),    FML(7.7050),    FML(6.1662),    FML(5.7413),
    FML(8.8200),   FML(11.2173),   FML(10.5776),   FML(13.4995),   FML(11.9195),    FML(9.7291),    FML(8.1137),    FML(7.5617),
   FML(11.6143),   FML(13.1398),   FML(14.4082),   FML(19.2992),   FML(15.4443),   FML(12.7738),   FML(10.9440),    FML(9.1353),
   FML(12.3009),   FML(13.5648),   FML(15.4552),   FML(19.5379),   FML(16.5648),   FML(14.3992),   FML(10.9154),   FML(10.0523),
   FML(11.6945),   FML(14.7459),   FML(16.1210),   FML(18.6451),   FML(16.5676),   FML(14.1675),   FML(12.9589),    FML(9.0159),
   FML(12.2111),   FML(14.7831),   FML(16.7885),   FML(19.8765),   FML(18.6367),   FML(15.5485),   FML(13.3976),   FML(10.9009),
    FML(9.9459),   FML(11.7673),   FML(14.8742),   FML(16.2403),   FML(15.3194),   FML(11.8362),   FML(10.6706),    FML(8.6330),
    FML(8.1102),    FML(8.9693),   FML(11.8548),   FML(11.8595),   FML(12.7328),    FML(9.4582),    FML(8.7455),    FML(6.8711),
    };

    TEST_ALL_2_COMPONENT_1_COUNT_KERNELS
}


int main()
{
    tCrashReporter::init();

    srand(time(0));

    tTest("conv2d() golden test 0", test0);

    tTest("conv2d() golden test 1", test1);
    tTest("conv2d() golden test 2", test2);
    tTest("conv2d() golden test 3", test3);
    tTest("conv2d() golden test 4", test4);
    tTest("conv2d() golden test 5", test5);
    tTest("conv2d() golden test 6", test6);
    tTest("conv2d() golden test 7", test7);
    tTest("conv2d() golden test 8", test8);
    tTest("conv2d() golden test 9", test9);
    tTest("conv2d() golden test 10", test10);
    tTest("conv2d() golden test 11", test11);
    tTest("conv2d() golden test 12", test12);
    tTest("conv2d() golden test 13", test13);
    tTest("conv2d() golden test 14", test14);
    tTest("conv2d() golden test 15", test15);
    tTest("conv2d() golden test 16", test16);
    tTest("conv2d() golden test 17", test17);
    tTest("conv2d() golden test 18", test18);
    tTest("conv2d() golden test 19", test19);
    tTest("conv2d() golden test 20", test20);
    tTest("conv2d() golden test 21", test21);
    tTest("conv2d() golden test 22", test22);
    tTest("conv2d() golden test 23", test23);
    tTest("conv2d() golden test 24", test24);
    tTest("conv2d() golden test 25", test25);
    tTest("conv2d() golden test 26", test26);
    tTest("conv2d() golden test 27", test27);
    tTest("conv2d() golden test 28", test28);
    tTest("conv2d() golden test 29", test29);
    tTest("conv2d() golden test 30", test30);
    tTest("conv2d() golden test 31", test31);
    tTest("conv2d() golden test 32", test32);
    tTest("conv2d() golden test 33", test33);
    tTest("conv2d() golden test 34", test34);
    tTest("conv2d() golden test 35", test35);
    tTest("conv2d() golden test 36", test36);
    tTest("conv2d() golden test 37", test37);
    tTest("conv2d() golden test 38", test38);
    tTest("conv2d() golden test 39", test39);
    tTest("conv2d() golden test 40", test40);
    tTest("conv2d() golden test 41", test41);
    tTest("conv2d() golden test 42", test42);
    tTest("conv2d() golden test 43", test43);
    tTest("conv2d() golden test 44", test44);
    tTest("conv2d() golden test 45", test45);
    tTest("conv2d() golden test 46", test46);
    tTest("conv2d() golden test 47", test47);
    tTest("conv2d() golden test 48", test48);
    tTest("conv2d() golden test 49", test49);
    tTest("conv2d() golden test 50", test50);
    tTest("conv2d() golden test 51", test51);
    tTest("conv2d() golden test 52", test52);
    tTest("conv2d() golden test 53", test53);
    tTest("conv2d() golden test 54", test54);
    tTest("conv2d() golden test 55", test55);
    tTest("conv2d() golden test 56", test56);
    tTest("conv2d() golden test 57", test57);
    tTest("conv2d() golden test 58", test58);
    tTest("conv2d() golden test 59", test59);
    tTest("conv2d() golden test 60", test60);
    tTest("conv2d() golden test 61", test61);
    tTest("conv2d() golden test 62", test62);
    tTest("conv2d() golden test 63", test63);
    tTest("conv2d() golden test 64", test64);

    tTest("conv2d() golden test 65", test65);
    tTest("conv2d() golden test 66", test66);
    tTest("conv2d() golden test 67", test67);
    tTest("conv2d() golden test 68", test68);
    tTest("conv2d() golden test 69", test69);
    tTest("conv2d() golden test 70", test70);
    tTest("conv2d() golden test 71", test71);
    tTest("conv2d() golden test 72", test72);
    tTest("conv2d() golden test 73", test73);
    tTest("conv2d() golden test 74", test74);
    tTest("conv2d() golden test 75", test75);
    tTest("conv2d() golden test 76", test76);
    tTest("conv2d() golden test 77", test77);
    tTest("conv2d() golden test 78", test78);
    tTest("conv2d() golden test 79", test79);
    tTest("conv2d() golden test 80", test80);

    tTest("conv2d() golden test 81", test81);
    tTest("conv2d() golden test 82", test82);
    tTest("conv2d() golden test 83", test83);
    tTest("conv2d() golden test 84", test84);
    tTest("conv2d() golden test 85", test85);
    tTest("conv2d() golden test 86", test86);
    tTest("conv2d() golden test 87", test87);
    tTest("conv2d() golden test 88", test88);
    tTest("conv2d() golden test 89", test89);
    tTest("conv2d() golden test 90", test90);
    tTest("conv2d() golden test 91", test91);
    tTest("conv2d() golden test 92", test92);
    tTest("conv2d() golden test 93", test93);
    tTest("conv2d() golden test 94", test94);
    tTest("conv2d() golden test 95", test95);
    tTest("conv2d() golden test 96", test96);
    tTest("conv2d() golden test 97", test97);
    tTest("conv2d() golden test 98", test98);
    tTest("conv2d() golden test 99", test99);
    tTest("conv2d() golden test 100", test100);
    tTest("conv2d() golden test 101", test101);
    tTest("conv2d() golden test 102", test102);
    tTest("conv2d() golden test 103", test103);
    tTest("conv2d() golden test 104", test104);
    tTest("conv2d() golden test 105", test105);
    tTest("conv2d() golden test 106", test106);
    tTest("conv2d() golden test 107", test107);
    tTest("conv2d() golden test 108", test108);
    tTest("conv2d() golden test 109", test109);
    tTest("conv2d() golden test 110", test110);
    tTest("conv2d() golden test 111", test111);
    tTest("conv2d() golden test 112", test112);
    tTest("conv2d() golden test 113", test113);
    tTest("conv2d() golden test 114", test114);
    tTest("conv2d() golden test 115", test115);
    tTest("conv2d() golden test 116", test116);
    tTest("conv2d() golden test 117", test117);
    tTest("conv2d() golden test 118", test118);
    tTest("conv2d() golden test 119", test119);
    tTest("conv2d() golden test 120", test120);
    tTest("conv2d() golden test 121", test121);
    tTest("conv2d() golden test 122", test122);
    tTest("conv2d() golden test 123", test123);
    tTest("conv2d() golden test 124", test124);
    tTest("conv2d() golden test 125", test125);
    tTest("conv2d() golden test 126", test126);
    tTest("conv2d() golden test 127", test127);
    tTest("conv2d() golden test 128", test128);
    tTest("conv2d() golden test 129", test129);
    tTest("conv2d() golden test 130", test130);
    tTest("conv2d() golden test 131", test131);
    tTest("conv2d() golden test 132", test132);
    tTest("conv2d() golden test 133", test133);
    tTest("conv2d() golden test 134", test134);
    tTest("conv2d() golden test 135", test135);
    tTest("conv2d() golden test 136", test136);
    tTest("conv2d() golden test 137", test137);
    tTest("conv2d() golden test 138", test138);
    tTest("conv2d() golden test 139", test139);
    tTest("conv2d() golden test 140", test140);
    tTest("conv2d() golden test 141", test141);
    tTest("conv2d() golden test 142", test142);
    tTest("conv2d() golden test 143", test143);
    tTest("conv2d() golden test 144", test144);

    return 0;
}
