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
 * convolution like we do, but I'm not using that above because I want to draw
 * explicit attention to this fact that conv2() and xcorr2() are different!
 */


fml kKernel33[] = {  FML(0.9522223),   FML(0.6157124),   FML(0.1143837),   // <--- This is a column! (Don't be fooled by the fact that it looks like a row here; all these matrices are interpreted as column-major.)
                     FML(0.4610145),   FML(0.0780320),   FML(0.0066089),
                     FML(0.6854424),   FML(0.7684590),   FML(0.7727028)  };
fml kBias33     = FML(0.30678);

fml kKernel55[] = {  FML(0.407211),   FML(0.332282),   FML(0.042351),   FML(0.853344),   FML(0.857271),
                     FML(0.163832),   FML(0.443431),   FML(0.178985),   FML(0.452883),   FML(0.529514),
                     FML(0.612710),   FML(0.543656),   FML(0.715227),   FML(0.500823),   FML(0.602494),
                     FML(0.976182),   FML(0.424915),   FML(0.845589),   FML(0.179422),   FML(0.769882),
                     FML(0.060206),   FML(0.626647),   FML(0.932404),   FML(0.154073),   FML(0.879106)  };
fml kBias55     = FML(0.65545);

fml kKernel57[] = {  FML(0.323413),   FML(0.255111),   FML(0.389326),   FML(0.279595),   FML(0.829499),   // <-- Again, note that this is a column!
                     FML(0.382419),   FML(0.392395),   FML(0.033404),   FML(0.151718),   FML(0.775017),
                     FML(0.295482),   FML(0.478754),   FML(0.953186),   FML(0.692873),   FML(0.525434),
                     FML(0.593704),   FML(0.301498),   FML(0.770169),   FML(0.112731),   FML(0.478316),
                     FML(0.172259),   FML(0.050867),   FML(0.688015),   FML(0.040391),   FML(0.080661),
                     FML(0.430828),   FML(0.730764),   FML(0.707751),   FML(0.032500),   FML(0.232391),
                     FML(0.332616),   FML(0.140028),   FML(0.653501),   FML(0.245474),   FML(0.752484)  };
fml kBias57     = FML(0.94145);


#define TEST_ALL_KERNELS \
    fml* output = new fml[inputRows*inputCols]; \
    for (u32 i = 0; i < inputRows*inputCols; i++) \
        output[i] = FML(0.0); \
 \
    { \
        s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel33, 3, 3, \
                            1, 1, \
                            1, \
                 &kBias33, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k33, inputRows*inputCols); \
    } \
 \
    { \
        s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel55, 5, 5, \
                            1, 1, \
                            1, \
                 &kBias55, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k55, inputRows*inputCols); \
    } \
 \
    { \
        s_conv2d(input, inputRows, inputCols, 1, \
                 kKernel57, 5, 7, \
                            1, 1, \
                            1, \
                 &kBias57, FML(1.0), \
                 output); \
 \
        s_checkOutput(t, output, correctOutput_k57, inputRows*inputCols); \
    } \
 \
    delete [] output; \


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


void test1(const tTest& t)
{
    fml input[] = {  FML(0.33510)  };
    u32 inputRows = 1;
    u32 inputCols = 1;

    fml correctOutput_k33[] = {  FML(0.33293)  };

    fml correctOutput_k55[] = {  FML(0.89513)  };

    fml correctOutput_k57[] = {  FML(1.1995)  };

    TEST_ALL_KERNELS
}


void test2(const tTest& t)
{
    fml input[] = {  FML(0.26893),   FML(0.94183),  };
    u32 inputRows = 1;
    u32 inputCols = 2;

    fml correctOutput_k33[] = {  FML(1.05152),   FML(0.54586),  };

    fml correctOutput_k55[] = {  FML(1.6442),   FML(1.3772),  };

    fml correctOutput_k57[] = {  FML(1.7966),   FML(1.9232),  };

    TEST_ALL_KERNELS
}


void test3(const tTest& t)
{
    fml input[] = {  FML(0.65274),  FML(0.71119), };
    u32 inputRows = 2;
    u32 inputCols = 1;

    fml correctOutput_k33[] = {  FML(0.36242),  FML(0.66320),  };

    fml correctOutput_k55[] = { FML(1.4785),  FML(1.5190),  };

    fml correctOutput_k57[] = { FML(1.5243),  FML(1.6860),  };

    TEST_ALL_KERNELS
}


void test4(const tTest& t)
{
    fml input[] = {     FML(0.11911),   FML(0.26430),
           FML(0.23334),   FML(0.98844),
    };
    u32 inputRows = 2;
    u32 inputCols = 2;

    fml correctOutput_k33[] = {     FML(1.26090),   FML(0.43509),
           FML(1.30183),   FML(0.76763),   };

    fml correctOutput_k55[] = {      FML(1.2477),   FML(1.4584),
           FML(1.8442),   FML(1.5894),   };

    fml correctOutput_k57[] = {     FML(1.2634),   FML(1.5292),
           FML(1.8728),   FML(2.0820),   };

    TEST_ALL_KERNELS
}


void test5(const tTest& t)
{
    fml input[] = {  FML(0.88967),   FML(0.76575),   FML(0.21727),  };
    u32 inputRows = 3;
    u32 inputCols = 1;

    fml correctOutput_k33[] = {        FML(0.38127),
       FML(0.77812),
       FML(0.67676),
    };

    fml correctOutput_k55[] = {        FML(1.8062),
       FML(1.7956),
       FML(1.7723),
    };

    fml correctOutput_k57[] = {         FML(1.8169),
       FML(1.8239),
       FML(1.8679),
    };

    TEST_ALL_KERNELS
}


void test6(const tTest& t)
{
    fml input[] = {     FML(0.0050072),
       FML(0.2020735),
       FML(0.0744064),
    };
    u32 inputRows = 1;
    u32 inputCols = 3;

    fml correctOutput_k33[] = {  FML(0.46246),   FML(0.38281),   FML(0.43701),  };

    fml correctOutput_k55[] = {  FML(0.89928),   FML(0.86379),   FML(0.74505),  };

    fml correctOutput_k57[] = {  FML(1.1370),   FML(1.1530),   FML(1.1915),  };

    TEST_ALL_KERNELS
}


void test7(const tTest& t)
{
    fml input[] = {      FML(0.910633),   FML(0.303282),   FML(0.091239),
       FML(0.212374),   FML(0.522718),   FML(0.723486),
    };
    u32 inputRows = 3;
    u32 inputCols = 2;

    fml correctOutput_k33[] = {     FML(0.94695),   FML(0.92219),
       FML(1.85716),   FML(1.51455),
       FML(1.36798),   FML(0.94919),
    };

    fml correctOutput_k55[] = {     FML(2.3440),   FML(1.8537),
       FML(2.0752),   FML(2.0065),
       FML(2.4847),   FML(1.8872),
    };

    fml correctOutput_k57[] = {     FML(1.9462),   FML(2.6361),
       FML(1.8595),   FML(2.2779),
       FML(2.2047),   FML(2.2836),
    };

    TEST_ALL_KERNELS
}


void test8(const tTest& t)
{
    fml input[] = {     FML(0.23586),   FML(0.87195),
       FML(0.13268),   FML(0.14731),
       FML(0.56936),   FML(0.80738),
    };
    u32 inputRows = 2;
    u32 inputCols = 3;

    fml correctOutput_k33[] = {     FML(0.54674),   FML(1.62446),   FML(0.45509),
       FML(0.68771),   FML(2.15161),   FML(0.84931),
    };

    fml correctOutput_k55[] = {     FML(2.0547),   FML(1.8875),   FML(2.3116),
       FML(2.6979),   FML(2.0182),   FML(1.7429),
    };

    fml correctOutput_k57[] = {      FML(1.7478),   FML(2.3136),   FML(1.8397),
       FML(2.7797),   FML(2.6234),   FML(2.0605),
    };

    TEST_ALL_KERNELS
}


void test9(const tTest& t)
{
    fml input[] = {     FML(0.16747),   FML(0.84134),   FML(0.80023),
       FML(0.62639),   FML(0.63127),   FML(0.87334),
       FML(0.38374),   FML(0.79613),   FML(0.93436),
    };
    u32 inputRows = 3;
    u32 inputCols = 3;

    fml correctOutput_k33[] = {     FML(1.29455),   FML(1.46924),   FML(0.79987),
       FML(2.04422),   FML(3.01642),   FML(1.63703),
       FML(1.86092),   FML(3.22353),   FML(1.88556),
    };

    fml correctOutput_k55[] = {     FML(4.2959),   FML(3.9672),   FML(4.1631),
       FML(3.8324),   FML(3.4760),   FML(3.4619),
       FML(4.7992),   FML(4.0538),   FML(2.9122),
    };

    fml correctOutput_k57[] = {     FML(2.5896),   FML(3.4474),   FML(4.0204),
       FML(3.1058),   FML(3.7565),   FML(3.4976),
       FML(4.0601),   FML(4.1408),   FML(3.8696),
    };

    TEST_ALL_KERNELS
}


void test10(const tTest& t)
{
    fml input[] = {  FML(0.129483),   FML(0.017115),   FML(0.065013),   FML(0.282924),  };
    u32 inputRows = 4;
    u32 inputCols = 1;

    fml correctOutput_k33[] = {  FML(0.31700),
       FML(0.36824),
       FML(0.32161),
       FML(0.35883),};

    fml correctOutput_k55[] = {  FML(0.79580),
       FML(0.94111),
       FML(0.93229),
       FML(0.90364),};

    fml correctOutput_k57[] = {  FML(1.0742),
       FML(1.1363),
       FML(1.1055),
       FML(1.1891),};

    TEST_ALL_KERNELS
}


void test11(const tTest& t)
{
    fml input[] = {  FML(0.15208),
   FML(0.99438),
   FML(0.84966),
   FML(0.71036),  };
    u32 inputRows = 1;
    u32 inputCols = 4;

    fml correctOutput_k33[] = {     FML(1.08279),   FML(1.13094),   FML(1.53122),   FML(0.88536),
  };

    fml correctOutput_k55[] = {     FML(2.3973),   FML(2.7747),   FML(2.0482),   FML(1.3577),
  };

    fml correctOutput_k57[] = {     FML(2.8083),   FML(2.9396),   FML(3.0375),   FML(2.3909),
  };

    TEST_ALL_KERNELS
}


void test12(const tTest& t)
{
    fml input[] = {     FML(0.83547),   FML(0.94485),   FML(0.38129),   FML(0.95135),
   FML(0.29404),   FML(0.13849),   FML(0.88791),   FML(0.21156),
  };
    u32 inputRows = 4;
    u32 inputCols = 2;

    fml correctOutput_k33[] = {     FML(0.71119),   FML(0.95313),
   FML(1.76226),   FML(1.87994),
   FML(1.71913),   FML(1.68460),
   FML(1.32798),   FML(1.68146),
  };

    fml correctOutput_k55[] = {     FML(2.9130),   FML(2.2494),
   FML(3.1138),   FML(2.7025),
   FML(3.5648),   FML(2.7069),
   FML(2.8135),   FML(1.8685),
  };

    fml correctOutput_k57[] = {     FML(2.1533),   FML(3.2596),
   FML(2.5822),   FML(3.4027),
   FML(2.8004),   FML(3.5873),
   FML(2.5646),   FML(2.8229),
  };

    TEST_ALL_KERNELS
}


void test13(const tTest& t)
{
    fml input[] = {     FML(0.83401),   FML(0.40772),
   FML(0.64237),   FML(0.55419),
   FML(0.94360),   FML(0.98256),
   FML(0.38225),   FML(0.98305),
  };
    u32 inputRows = 2;
    u32 inputCols = 4;

    fml correctOutput_k33[] = {     FML(1.2964),   FML(2.4051),   FML(1.8992),   FML(1.0365),
   FML(1.5893),   FML(3.0932),   FML(2.7888),   FML(2.0632),
  };

    fml correctOutput_k55[] = {     FML(3.1300),   FML(3.2084),   FML(3.0712),   FML(2.5352),
   FML(3.6495),   FML(4.2318),   FML(3.5433),   FML(2.3976),
  };

    fml correctOutput_k57[] = {      FML(3.2850),   FML(3.5675),   FML(3.1676),   FML(3.4711),
   FML(4.0018),   FML(4.0490),   FML(3.8551),   FML(3.8442),
 };

    TEST_ALL_KERNELS
}


void test14(const tTest& t)
{
    fml input[] = {    FML(0.0053767),   FML(0.5394381),   FML(0.7707835),   FML(0.0895143),
   FML(0.9633067),   FML(0.7150167),   FML(0.5405788),   FML(0.0043170),
   FML(0.7212126),   FML(0.4241238),   FML(0.0188700),   FML(0.9632647),
   };
    u32 inputRows = 4;
    u32 inputCols = 3;

    fml correctOutput_k33[] = {     FML(1.60352),   FML(1.33363),   FML(1.04077),
   FML(1.98391),   FML(2.07052),   FML(2.09185),
   FML(1.52506),   FML(2.72664),   FML(1.52434),
   FML(1.04296),   FML(2.09857),   FML(0.90806),
  };

    fml correctOutput_k55[] = {    FML(3.5073),   FML(3.3821),   FML(3.2989),
   FML(4.2955),   FML(3.8694),   FML(3.5021),
   FML(3.7254),   FML(3.5148),   FML(2.6879),
   FML(3.3359),   FML(3.0733),   FML(2.4517),
   };

    fml correctOutput_k57[] = {    FML(2.6389),   FML(3.3213),   FML(3.9308),
   FML(3.1030),   FML(3.3504),   FML(3.6744),
   FML(2.9507),   FML(3.4000),   FML(3.0189),
   FML(2.5950),   FML(2.8828),   FML(2.9268),
   };

    TEST_ALL_KERNELS
}


void test15(const tTest& t)
{
    fml input[] = {    FML(0.790240),   FML(0.439315),   FML(0.380394),
   FML(0.586823),   FML(0.184325),   FML(0.782359),
   FML(0.134491),   FML(0.917011),   FML(0.174519),
   FML(0.995813),   FML(0.029755),   FML(0.985472),
   };
    u32 inputRows = 3;
    u32 inputCols = 4;

    fml correctOutput_k33[] = {      FML(0.96473),   FML(1.70253),   FML(1.49397),   FML(0.57238),
   FML(1.85630),   FML(2.59508),   FML(2.67017),   FML(1.48734),
   FML(1.26655),   FML(1.86801),   FML(2.17807),   FML(1.37805),
 };

    fml correctOutput_k55[] = {     FML(3.2216),   FML(4.3927),   FML(4.2594),   FML(3.3609),
   FML(3.1016),   FML(3.7669),   FML(3.3503),   FML(2.8849),
   FML(3.7087),   FML(3.7331),   FML(3.9810),   FML(2.7799),
  };

    fml correctOutput_k57[] = {    FML(3.8210),   FML(4.1243),   FML(3.4837),   FML(4.4383),
   FML(2.9030),   FML(3.8348),   FML(3.2000),   FML(3.2692),
   FML(4.3158),   FML(4.0925),   FML(3.7779),   FML(3.7844),
   };

    TEST_ALL_KERNELS
}


void test16(const tTest& t)
{
    fml input[] = {     FML(0.8409273),   FML(0.5840855),   FML(0.0079345),   FML(0.6151121),
   FML(0.3644661),   FML(0.8707131),   FML(0.5948696),   FML(0.2140174),
   FML(0.1441759),   FML(0.7902441),   FML(0.1294370),   FML(0.6167350),
   FML(0.0255033),   FML(0.4408352),   FML(0.8814614),   FML(0.2018693),
  };
    u32 inputRows = 4;
    u32 inputCols = 4;

    fml correctOutput_k33[] = {      FML(1.32914),   FML(1.64697),   FML(1.00749),   FML(0.49085),
   FML(2.11868),   FML(2.51408),   FML(2.42433),   FML(0.99742),
   FML(1.80007),   FML(2.50514),   FML(3.04065),   FML(1.48286),
   FML(0.93065),   FML(1.54667),   FML(1.87212),   FML(1.23188),
 };

    fml correctOutput_k55[] = {     FML(2.8466),   FML(3.3598),   FML(3.3270),   FML(3.1462),
   FML(4.4569),   FML(4.6778),   FML(3.9711),   FML(2.9873),
   FML(3.7887),   FML(4.3889),   FML(4.2796),   FML(2.9644),
   FML(3.4450),   FML(3.9898),   FML(3.3380),   FML(2.4077),
  };

    fml correctOutput_k57[] = {     FML(2.9386),   FML(3.1940),   FML(2.6961),   FML(3.2880),
   FML(4.0710),   FML(4.1916),   FML(4.6161),   FML(4.1032),
   FML(3.6643),   FML(4.0071),   FML(3.9602),   FML(3.8922),
   FML(3.3654),   FML(4.1093),   FML(3.1774),   FML(3.5117),
  };

    TEST_ALL_KERNELS
}


void test17(const tTest& t)
{
    fml input[] = {  FML(0.925719),   FML(0.044776),   FML(0.367802),   FML(0.808171),   FML(0.112447),  };
    u32 inputRows = 5;
    u32 inputCols = 1;

    fml correctOutput_k33[] = {     FML(0.37931),
   FML(0.73948),
   FML(0.36147),
   FML(0.54015),
   FML(0.68813),
  };

    fml correctOutput_k55[] = {     FML(1.5616),
   FML(1.8619),
   FML(1.9826),
   FML(1.5172),
   FML(1.4006),
  };

    fml correctOutput_k57[] = {    FML(1.8354),
   FML(1.6831),
   FML(1.9327),
   FML(1.7140),
   FML(1.4901),
   };

    TEST_ALL_KERNELS
}


void test18(const tTest& t)
{
    fml input[] = {      FML(0.11779),
   FML(0.44911),
   FML(0.67194),
   FML(0.66061),
   FML(0.98855),
 };
    u32 inputRows = 1;
    u32 inputCols = 5;

    fml correctOutput_k33[] = {    FML(0.66110),   FML(0.93071),   FML(1.14339),   FML(1.53171),   FML(0.79066),
   };

    fml correctOutput_k55[] = {    FML(1.7460),   FML(2.1819),   FML(2.7017),   FML(2.1031),   FML(1.5092),
   };

    fml correctOutput_k57[] = {     FML(2.2484),   FML(2.9755),   FML(3.0451),   FML(2.8317),   FML(2.5298),
  };

    TEST_ALL_KERNELS
}


void test19(const tTest& t)
{
    fml input[] = {     FML(0.182864),   FML(0.640849),   FML(0.408931),   FML(0.315291),   FML(0.651862),
   FML(0.721770),   FML(0.996048),   FML(0.097393),   FML(0.136901),   FML(0.782035),
  };
    u32 inputRows = 5;
    u32 inputCols = 2;

    fml correctOutput_k33[] = {     FML(1.64959),   FML(0.55558),
   FML(1.77920),   FML(1.33338),
   FML(1.49958),   FML(1.67256),
   FML(1.30046),   FML(1.02562),
   FML(1.19780),   FML(1.13251),
  };

    fml correctOutput_k55[] = {      FML(2.2176),   FML(2.2687),
   FML(2.8798),   FML(2.4394),
   FML(3.7958),   FML(3.1239),
   FML(3.0922),   FML(2.4462),
   FML(2.3582),   FML(1.6724),
 };

    fml correctOutput_k57[] = {    FML(1.8948),   FML(2.4894),
   FML(2.4240),   FML(3.1500),
   FML(2.2161),   FML(3.4464),
   FML(2.0638),   FML(2.8931),
   FML(2.3431),   FML(2.5360),
   };

    TEST_ALL_KERNELS
}


void test20(const tTest& t)
{
    fml input[] = {    FML(0.714691),   FML(0.327563),
   FML(0.076002),   FML(0.859363),
   FML(0.974600),   FML(0.332810),
   FML(0.837369),   FML(0.139951),
   FML(0.753754),   FML(0.110792),
   };
    u32 inputRows = 2;
    u32 inputCols = 5;

    fml correctOutput_k33[] = {    FML(1.08715),   FML(1.80201),   FML(1.28175),   FML(1.67603),   FML(0.89792),
   FML(1.37430),   FML(2.21489),   FML(2.06506),   FML(2.43849),   FML(1.54645),
   };

    fml correctOutput_k55[] = {    FML(2.5091),   FML(3.1026),   FML(3.6848),   FML(3.0434),   FML(1.7886),
   FML(2.9583),   FML(3.0377),   FML(2.9120),   FML(2.1781),   FML(1.8788),
   };

    fml correctOutput_k57[] = {      FML(2.8980),   FML(3.8060),   FML(3.5899),   FML(3.7875),   FML(2.7825),
   FML(3.1608),   FML(3.4481),   FML(3.4066),   FML(2.5685),   FML(2.5358),
 };

    TEST_ALL_KERNELS
}


void test21(const tTest& t)
{
    fml input[] = {    FML(0.587346),   FML(0.854598),   FML(0.511826),   FML(0.546104),   FML(0.153540),
   FML(0.187641),   FML(0.586582),   FML(0.830197),   FML(0.531662),   FML(0.261282),
   FML(0.443431),   FML(0.789160),   FML(0.091968),   FML(0.838380),   FML(0.990026),
   };
    u32 inputRows = 5;
    u32 inputCols = 3;

    fml correctOutput_k33[] = {     FML(0.95571),   FML(1.73523),   FML(0.52923),
   FML(1.86850),   FML(2.57001),   FML(1.20820),
   FML(2.19517),   FML(3.09629),   FML(1.81385),
   FML(1.76587),   FML(3.04620),   FML(1.56891),
   FML(1.13573),   FML(2.52228),   FML(1.43767),
  };

    fml correctOutput_k55[] = {     FML(3.3309),   FML(2.9342),   FML(3.3550),
   FML(5.0703),   FML(4.3652),   FML(3.9941),
   FML(5.2495),   FML(4.7608),   FML(4.3994),
   FML(4.3860),   FML(4.2076),   FML(3.6344),
   FML(4.0870),   FML(3.2767),   FML(2.6906),
  };

    fml correctOutput_k57[] = {     FML(2.3156),   FML(3.3145),   FML(2.9834),
   FML(3.6656),   FML(4.1719),   FML(4.3578),
   FML(3.8431),   FML(3.8771),   FML(4.4937),
   FML(3.5934),   FML(3.8608),   FML(4.0228),
   FML(3.2310),   FML(3.0945),   FML(3.1754),
  };

    TEST_ALL_KERNELS
}


void test22(const tTest& t)
{
    fml input[] = {     FML(0.095766),   FML(0.707168),   FML(0.776445),
   FML(0.153225),   FML(0.063558),   FML(0.066990),
   FML(0.967620),   FML(0.282185),   FML(0.531506),
   FML(0.289289),   FML(0.146912),   FML(0.633923),
   FML(0.078587),   FML(0.064247),   FML(0.116463),
  };
    u32 inputRows = 3;
    u32 inputCols = 5;

    fml correctOutput_k33[] = {     FML(0.48579),   FML(1.42063),   FML(0.82159),   FML(1.06841),   FML(0.50826),
   FML(0.61688),   FML(2.28903),   FML(1.77212),   FML(1.80496),   FML(0.78723),
   FML(0.78843),   FML(2.09462),   FML(1.16796),   FML(1.15347),   FML(0.87569),
  };

    fml correctOutput_k55[] = {     FML(3.1514),   FML(3.7135),   FML(4.1185),   FML(2.1861),   FML(2.0054),
   FML(2.6844),   FML(2.4995),   FML(2.9715),   FML(2.1750),   FML(2.0312),
   FML(2.6178),   FML(3.5145),   FML(3.1885),   FML(2.0336),   FML(1.5581),
  };

    fml correctOutput_k57[] = {    FML(3.0994),   FML(3.3195),   FML(3.2511),   FML(3.8929),   FML(2.3956),
   FML(2.8729),   FML(2.9844),   FML(2.1179),   FML(2.9559),   FML(2.2978),
   FML(3.4156),   FML(3.5487),   FML(3.1467),   FML(3.2697),   FML(2.4476),
   };

    TEST_ALL_KERNELS
}


void test23(const tTest& t)
{
    fml input[] = {     FML(0.44533),   FML(0.84106),   FML(0.69988),   FML(0.84110),   FML(0.33373),
   FML(0.76492),   FML(0.54171),   FML(0.53141),   FML(0.75282),   FML(0.76615),
   FML(0.95980),   FML(0.59822),   FML(0.21656),   FML(0.20624),   FML(0.21789),
   FML(0.65021),   FML(0.84997),   FML(0.47749),   FML(0.27931),   FML(0.64980),
  };
    u32 inputRows = 5;
    u32 inputCols = 4;

    fml correctOutput_k33[] = {    FML(1.35348),   FML(1.94027),   FML(2.07500),   FML(1.02252),
   FML(1.93355),   FML(3.01210),   FML(3.38788),   FML(1.98306),
   FML(2.11608),   FML(2.66679),   FML(2.69532),   FML(1.46430),
   FML(2.23204),   FML(2.31335),   FML(2.52536),   FML(0.91112),
   FML(1.82535),   FML(2.02883),   FML(2.29824),   FML(0.81680),
   };

    fml correctOutput_k55[] = {     FML(4.1475),   FML(4.8679),   FML(4.8421),   FML(3.3415),
   FML(5.1883),   FML(5.9192),   FML(5.7109),   FML(4.1402),
   FML(5.5175),   FML(6.9764),   FML(6.6098),   FML(4.8721),
   FML(4.2462),   FML(4.5487),   FML(4.4833),   FML(3.3735),
   FML(3.6122),   FML(3.7786),   FML(3.3270),   FML(2.2295),
  };

    fml correctOutput_k57[] = {    FML(4.0470),   FML(4.5216),   FML(4.4405),   FML(4.7183),
   FML(4.8521),   FML(5.5940),   FML(5.0815),   FML(5.5844),
   FML(5.1362),   FML(6.1490),   FML(5.3594),   FML(5.7015),
   FML(4.2671),   FML(4.9047),   FML(4.2723),   FML(3.9348),
   FML(3.5454),   FML(4.0690),   FML(3.7001),   FML(3.2751),
   };

    TEST_ALL_KERNELS
}


void test24(const tTest& t)
{
    fml input[] = {     FML(0.39433),   FML(0.85712),   FML(0.26532),   FML(0.66932),
   FML(0.52408),   FML(0.81695),   FML(0.33451),   FML(0.44357),
   FML(0.53334),   FML(0.94254),   FML(0.41904),   FML(0.77385),
   FML(0.67763),   FML(0.27004),   FML(0.56726),   FML(0.37769),
   FML(0.49249),   FML(0.65421),   FML(0.50793),   FML(0.77860),
  };
    u32 inputRows = 4;
    u32 inputCols = 5;

    fml correctOutput_k33[] = {      FML(1.37721),   FML(1.83206),   FML(1.50014),   FML(1.68160),   FML(0.79765),
   FML(1.80270),   FML(2.96160),   FML(2.77959),   FML(3.01291),   FML(1.46463),
   FML(1.88683),   FML(3.33456),   FML(2.72658),   FML(3.16244),   FML(1.30277),
   FML(1.05147),   FML(2.04226),   FML(1.83104),   FML(2.41974),   FML(1.37440),
 };

    fml correctOutput_k55[] = {    FML(3.3848),   FML(4.3550),   FML(5.4426),   FML(4.2923),   FML(3.3714),
   FML(5.2919),   FML(5.5527),   FML(6.6183),   FML(5.0608),   FML(4.1584),
   FML(4.2419),   FML(4.8888),   FML(6.2754),   FML(4.9486),   FML(3.7779),
   FML(4.1592),   FML(4.5094),   FML(4.7490),   FML(3.8536),   FML(2.8085),
   };

    fml correctOutput_k57[] = {     FML(3.3307),   FML(4.6300),   FML(4.2598),   FML(4.5669),   FML(3.9640),
   FML(4.6536),   FML(6.2744),   FML(5.7763),   FML(5.6987),   FML(5.0141),
   FML(4.0606),   FML(5.1909),   FML(5.3330),   FML(5.1842),   FML(4.3793),
   FML(4.1848),   FML(5.1990),   FML(5.0413),   FML(4.5186),   FML(3.8670),
  };

    TEST_ALL_KERNELS
}


void test25(const tTest& t)
{
    fml input[] = {     FML(0.558359),   FML(0.249298),   FML(0.925719),   FML(0.479244),   FML(0.396878),
   FML(0.620633),   FML(0.947443),   FML(0.047946),   FML(0.222422),   FML(0.739856),
   FML(0.222862),   FML(0.445760),   FML(0.521208),   FML(0.561449),   FML(0.586958),
   FML(0.160702),   FML(0.769435),   FML(0.661263),   FML(0.022825),   FML(0.833377),
   FML(0.069756),   FML(0.832292),   FML(0.756354),   FML(0.538540),   FML(0.428416),
  };
    u32 inputRows = 5;
    u32 inputCols = 5;

    fml correctOutput_k33[] = {     FML(1.56102),   FML(1.24948),   FML(1.53566),   FML(1.20933),   FML(0.50468),
   FML(1.78029),   FML(2.35626),   FML(2.83996),   FML(2.26340),   FML(1.11130),
   FML(1.35524),   FML(2.75086),   FML(2.56700),   FML(3.09070),   FML(1.89549),
   FML(1.54905),   FML(2.81535),   FML(1.97674),   FML(2.79138),   FML(1.43937),
   FML(1.27970),   FML(2.00366),   FML(1.93481),   FML(1.97671),   FML(1.12334),
  };

    fml correctOutput_k55[] = {     FML(3.2038),   FML(3.8252),   FML(4.5889),   FML(3.7384),   FML(3.1419),
   FML(4.2635),   FML(4.7312),   FML(5.9407),   FML(4.1117),   FML(3.5315),
   FML(5.3140),   FML(6.1820),   FML(7.5427),   FML(5.9327),   FML(4.4002),
   FML(4.0865),   FML(4.3556),   FML(5.5200),   FML(5.2992),   FML(3.8565),
   FML(3.4645),   FML(3.8450),   FML(4.6341),   FML(3.5627),   FML(2.4094),
  };

    fml correctOutput_k57[] = {     FML(3.3963),   FML(4.0654),   FML(3.9070),   FML(3.6626),   FML(3.5098),
   FML(3.6693),   FML(5.5860),   FML(5.1247),   FML(4.8566),   FML(4.5685),
   FML(4.7330),   FML(6.5910),   FML(5.7823),   FML(6.5695),   FML(5.5364),
   FML(3.6883),   FML(5.2329),   FML(4.9461),   FML(4.7171),   FML(4.3647),
   FML(4.2878),   FML(4.5086),   FML(4.9645),   FML(4.1240),   FML(3.6830),
  };

    TEST_ALL_KERNELS
}


void test26(const tTest& t)
{
    fml input[] = {  FML(0.024917),   FML(0.153115),   FML(0.978350),   FML(0.889005),   FML(0.944352),   FML(0.600810),  };
    u32 inputRows = 6;
    u32 inputCols = 1;

    fml correctOutput_k33[] = {    FML(0.30974),
   FML(0.33668),
   FML(0.45959),
   FML(0.83343),
   FML(0.79429),
   FML(0.78902),
   };

    fml correctOutput_k55[] = {     FML(1.3394),
   FML(1.8041),
   FML(2.4679),
   FML(2.7519),
   FML(2.7145),
   FML(2.1433),
  };

    fml correctOutput_k57[] = {     FML(1.4459),
   FML(1.6024),
   FML(2.3078),
   FML(2.4058),
   FML(2.5854),
   FML(2.2167),
  };

    TEST_ALL_KERNELS
}


void test27(const tTest& t)
{
    fml input[] = {     FML(0.420782),
   FML(0.688949),
   FML(0.050508),
   FML(0.914609),
   FML(0.616177),
   FML(0.868565),
  };
    u32 inputRows = 1;
    u32 inputCols = 6;

    fml correctOutput_k33[] = {  FML(0.86904),   FML(0.65843),   FML(1.43776),   FML(0.88276),   FML(1.58546),   FML(0.75395),  };

    fml correctOutput_k55[] = {  FML(1.5861),   FML(2.1190),   FML(2.1806),   FML(2.6787),   FML(1.9964),   FML(1.4257),  };

    fml correctOutput_k57[] = {  FML(2.3730),   FML(2.9579),   FML(3.2841),   FML(2.9195),   FML(3.1553),   FML(2.2479),  };

    TEST_ALL_KERNELS
}


void test28(const tTest& t)
{
    fml input[] = {     FML(0.94795),   FML(0.50513),   FML(0.61361),   FML(0.98511),   FML(0.12661),   FML(0.92467),
   FML(0.30161),   FML(0.69983),   FML(0.35548),   FML(0.20567),   FML(0.89743),   FML(0.35674),
  };
    u32 inputRows = 6;
    u32 inputCols = 2;

    fml correctOutput_k33[] = {   FML(1.15663),   FML(0.97638),
   FML(1.80648),   FML(1.78664),
   FML(1.50583),   FML(1.62999),
   FML(1.76252),   FML(1.69796),
   FML(1.88319),   FML(1.59575),
   FML(1.32658),   FML(1.43824),
    };

    fml correctOutput_k55[] = {      FML(2.6104),   FML(2.1592),
   FML(3.3750),   FML(2.9322),
   FML(4.1396),   FML(3.1210),
   FML(4.0674),   FML(3.1671),
   FML(3.3779),   FML(2.7844),
   FML(2.8730),   FML(1.9076),
 };

    fml correctOutput_k57[] = {    FML(2.2864),   FML(2.9986),
   FML(2.6844),   FML(3.5879),
   FML(2.7136),   FML(3.9136),
   FML(2.9868),   FML(3.8499),
   FML(2.5081),   FML(3.3602),
   FML(2.6032),   FML(2.8420),
   };

    TEST_ALL_KERNELS
}


void test29(const tTest& t)
{
    fml input[] = {    FML(0.174785),   FML(0.781971),
   FML(0.570553),   FML(0.373159),
   FML(0.062650),   FML(0.273662),
   FML(0.536907),   FML(0.926551),
   FML(0.216780),   FML(0.426709),
   FML(0.846934),   FML(0.345301),
   };
    u32 inputRows = 2;
    u32 inputCols = 6;

    fml correctOutput_k33[] = {     FML(1.05238),   FML(0.81043),   FML(1.83600),   FML(0.92098),   FML(1.68073),   FML(0.55743),
   FML(1.12622),   FML(1.50008),   FML(2.21011),   FML(1.33126),   FML(2.36763),   FML(1.19333),
  };

    fml correctOutput_k55[] = {     FML(1.8221),   FML(2.3813),   FML(2.6712),   FML(3.0840),   FML(2.5542),   FML(2.4796),
   FML(2.1622),   FML(2.9084),   FML(2.8416),   FML(3.1978),   FML(2.1667),   FML(1.7530),
  };

    fml correctOutput_k57[] = {     FML(2.2034),   FML(2.8420),   FML(3.1598),   FML(2.8481),   FML(3.2770),   FML(2.3944),
   FML(2.8023),   FML(3.7787),   FML(3.3641),   FML(3.8611),   FML(3.0809),   FML(2.3374),
  };

    TEST_ALL_KERNELS
}


void test30(const tTest& t)
{
    fml input[] = {      FML(0.676707),   FML(0.512286),   FML(0.025366),   FML(0.300592),   FML(0.065322),   FML(0.093554),
   FML(0.946335),   FML(0.151629),   FML(0.803106),   FML(0.065963),   FML(0.971981),   FML(0.129226),
   FML(0.598129),   FML(0.770452),   FML(0.257236),   FML(0.676373),   FML(0.981487),   FML(0.784496),
 };
    u32 inputRows = 6;
    u32 inputCols = 3;

    fml correctOutput_k33[] = {      FML(1.20736),   FML(1.91185),   FML(0.95856),
   FML(2.04464),   FML(2.92370),   FML(1.73069),
   FML(1.31898),   FML(2.22601),   FML(1.33293),
   FML(1.69459),   FML(2.35978),   FML(1.40116),
   FML(1.34307),   FML(2.57507),   FML(1.37642),
   FML(1.10974),   FML(2.16037),   FML(1.82559),
 };

    fml correctOutput_k55[] = {     FML(3.7596),   FML(3.1007),   FML(2.7750),
   FML(4.0362),   FML(3.7553),   FML(3.4393),
   FML(5.7098),   FML(5.0298),   FML(4.2248),
   FML(3.7852),   FML(4.2383),   FML(3.9824),
   FML(4.0604),   FML(3.6440),   FML(2.8618),
   FML(2.9160),   FML(3.1523),   FML(2.7776),
  };

    fml correctOutput_k57[] = {     FML(2.7627),   FML(3.5482),   FML(3.1611),
   FML(3.0248),   FML(3.0789),   FML(3.7758),
   FML(3.6395),   FML(3.7776),   FML(4.5037),
   FML(2.9117),   FML(2.7543),   FML(4.0151),
   FML(3.2489),   FML(3.2655),   FML(3.5714),
   FML(2.9252),   FML(2.2886),   FML(2.9948),
  };

    TEST_ALL_KERNELS
}


void test31(const tTest& t)
{
    fml input[] = {     FML(0.5690960),   FML(0.0352472),   FML(0.6312075),
   FML(0.1824721),   FML(0.1323598),   FML(0.0043037),
   FML(0.8612800),   FML(0.8509640),   FML(0.2502913),
   FML(0.9197450),   FML(0.2933078),   FML(0.8741376),
   FML(0.3220842),   FML(0.4562881),   FML(0.5927772),
   FML(0.9526094),   FML(0.1352174),   FML(0.5842774),
  };
    u32 inputRows = 3;
    u32 inputCols = 6;

    fml correctOutput_k33[] = {    FML(0.59392),   FML(1.99573),   FML(1.44053),   FML(1.60821),   FML(1.77130),   FML(0.63251),
   FML(0.80618),   FML(2.47476),   FML(2.55892),   FML(3.16162),   FML(2.85951),   FML(1.41580),
   FML(0.46632),   FML(1.56597),   FML(1.72009),   FML(2.24291),   FML(1.92258),   FML(1.21418),
   };

    fml correctOutput_k55[] = {     FML(2.7960),   FML(4.0518),   FML(4.9337),   FML(5.0166),   FML(4.4886),   FML(3.3733),
   FML(2.8681),   FML(3.5110),   FML(4.2077),   FML(3.9626),   FML(3.4698),   FML(3.1196),
   FML(2.5312),   FML(3.5324),   FML(4.8040),   FML(4.3818),   FML(4.0875),   FML(2.6007),
  };

    fml correctOutput_k57[] = {     FML(3.8431),   FML(4.2768),   FML(4.7930),   FML(5.5298),   FML(4.2352),   FML(4.4380),
   FML(3.0867),   FML(3.8786),   FML(3.6481),   FML(4.5172),   FML(3.4511),   FML(3.5228),
   FML(3.9055),   FML(4.0312),   FML(4.6602),   FML(5.1196),   FML(4.3113),   FML(3.9655),
  };

    TEST_ALL_KERNELS
}


void test32(const tTest& t)
{
    fml input[] = {     FML(0.19763),   FML(0.89820),   FML(0.40292),   FML(0.85752),   FML(0.31106),   FML(0.14859),
   FML(0.86247),   FML(0.49456),   FML(0.14361),   FML(0.15298),   FML(0.71271),   FML(0.20572),
   FML(0.67857),   FML(0.80394),   FML(0.54525),   FML(0.19391),   FML(0.90921),   FML(0.90637),
   FML(0.38320),   FML(0.41561),   FML(0.16163),   FML(0.29393),   FML(0.65851),   FML(0.35986),
  };
    u32 inputRows = 6;
    u32 inputCols = 4;

    fml correctOutput_k33[] = {     FML(1.37306),   FML(1.74443),   FML(1.56826),   FML(0.84919),
   FML(1.55283),   FML(3.03547),   FML(2.53507),   FML(1.72045),
   FML(1.32553),   FML(2.86834),   FML(1.93428),   FML(1.63636),
   FML(1.32822),   FML(2.56217),   FML(1.73725),   FML(1.15117),
   FML(1.53888),   FML(2.99131),   FML(2.06672),   FML(1.34418),
   FML(1.10838),   FML(2.35880),   FML(2.32990),   FML(2.06228),
  };

    fml correctOutput_k55[] = {      FML(3.6539),   FML(3.9633),   FML(3.9696),   FML(2.5910),
   FML(4.4813),   FML(4.6533),   FML(4.5585),   FML(2.7747),
   FML(5.8018),   FML(6.3756),   FML(6.0176),   FML(4.1734),
   FML(4.7507),   FML(5.1853),   FML(5.2353),   FML(4.0948),
   FML(3.6519),   FML(4.4181),   FML(4.0043),   FML(2.6291),
   FML(3.5089),   FML(3.6967),   FML(3.6246),   FML(2.3559),
 };

    fml correctOutput_k57[] = {    FML(3.1198),   FML(3.6182),   FML(3.8043),   FML(3.7285),
   FML(4.2644),   FML(4.6253),   FML(4.1526),   FML(4.8261),
   FML(4.5638),   FML(5.0613),   FML(4.6484),   FML(5.5360),
   FML(4.5135),   FML(4.1639),   FML(4.2614),   FML(4.9548),
   FML(3.8899),   FML(4.1021),   FML(4.0975),   FML(4.1040),
   FML(3.6775),   FML(3.5153),   FML(3.3979),   FML(3.7078),
   };

    TEST_ALL_KERNELS
}


void test33(const tTest& t)
{
    fml input[] = {     FML(0.0157600),   FML(0.9660186),   FML(0.5966798),   FML(0.4689802),
   FML(0.1790812),   FML(0.9533145),   FML(0.4924935),   FML(0.7984092),
   FML(0.0039856),   FML(0.6272127),   FML(0.6074684),   FML(0.0427728),
   FML(0.0594562),   FML(0.2754080),   FML(0.3711813),   FML(0.9366450),
   FML(0.5012522),   FML(0.3354853),   FML(0.7954316),   FML(0.5470300),
   FML(0.5806783),   FML(0.5903405),   FML(0.1243456),   FML(0.9145592),
  };
    u32 inputRows = 4;
    u32 inputCols = 6;

    fml correctOutput_k33[] = {      FML(1.18864),   FML(0.93497),   FML(0.78904),   FML(1.03186),   FML(1.31861),   FML(0.70300),
   FML(1.62925),   FML(2.09914),   FML(1.71461),   FML(2.03362),   FML(1.78570),   FML(1.39622),
   FML(2.45063),   FML(3.06065),   FML(3.14371),   FML(2.70897),   FML(2.33194),   FML(1.46647),
   FML(1.56958),   FML(1.90231),   FML(2.52492),   FML(2.12136),   FML(2.43436),   FML(1.52971),
 };

    fml correctOutput_k55[] = {      FML(2.8460),   FML(3.3217),   FML(5.0033),   FML(4.7474),   FML(3.7420),   FML(2.6599),
   FML(4.2401),   FML(4.7094),   FML(5.5657),   FML(6.0494),   FML(4.6909),   FML(4.0900),
   FML(3.9578),   FML(4.2440),   FML(4.8328),   FML(5.1393),   FML(3.9695),   FML(3.4309),
   FML(4.1805),   FML(4.6142),   FML(4.7769),   FML(4.7556),   FML(3.9236),   FML(2.5196),
 };

    fml correctOutput_k57[] = {      FML(2.0995),   FML(3.6431),   FML(4.2558),   FML(4.1385),   FML(3.9326),   FML(3.6551),
   FML(4.1918),   FML(5.5794),   FML(5.9879),   FML(5.8274),   FML(4.6217),   FML(4.6863),
   FML(3.6271),   FML(5.0054),   FML(5.5546),   FML(5.0286),   FML(4.4097),   FML(3.8041),
   FML(4.2947),   FML(5.0906),   FML(6.0628),   FML(5.1540),   FML(4.9153),   FML(3.6919),
 };

    TEST_ALL_KERNELS
}


void test34(const tTest& t)
{
    fml input[] = {     FML(0.940166),   FML(0.999423),   FML(0.383811),   FML(0.408223),   FML(0.258484),   FML(0.473027),
   FML(0.675888),   FML(0.836528),   FML(0.769963),   FML(0.338051),   FML(0.544250),   FML(0.014771),
   FML(0.405565),   FML(0.846690),   FML(0.336460),   FML(0.464707),   FML(0.122958),   FML(0.749222),
   FML(0.675371),   FML(0.255459),   FML(0.263633),   FML(0.865170),   FML(0.923362),   FML(0.157808),
   FML(0.607178),   FML(0.626431),   FML(0.218367),   FML(0.576806),   FML(0.263519),   FML(0.142368),
  };
    u32 inputRows = 6;
    u32 inputCols = 5;

    fml correctOutput_k33[] = {    FML(1.55253),   FML(2.02414),   FML(1.57225),   FML(1.65836),   FML(0.80336),
   FML(2.52181),   FML(3.43187),   FML(2.67172),   FML(2.65211),   FML(1.46757),
   FML(2.22647),   FML(3.18743),   FML(3.08196),   FML(2.56028),   FML(1.12096),
   FML(1.72537),   FML(2.02085),   FML(3.06158),   FML(1.91906),   FML(1.34355),
   FML(1.17964),   FML(2.09911),   FML(2.61878),   FML(2.09052),   FML(2.00461),
   FML(0.84726),   FML(1.75625),   FML(1.70345),   FML(1.61320),   FML(1.41579),
   };

    fml correctOutput_k55[] = {    FML(4.1784),   FML(4.5006),   FML(5.3760),   FML(4.2559),   FML(2.9393),
   FML(5.2165),   FML(5.9653),   FML(6.6610),   FML(5.0663),   FML(3.6693),
   FML(5.6039),   FML(6.1697),   FML(7.0837),   FML(5.8548),   FML(4.0864),
   FML(5.0942),   FML(6.1140),   FML(6.4319),   FML(5.3919),   FML(3.8271),
   FML(3.4340),   FML(4.4196),   FML(4.7404),   FML(3.8174),   FML(2.9622),
   FML(2.7618),   FML(3.3525),   FML(3.8174),   FML(3.1797),   FML(2.0952),
   };

    fml correctOutput_k57[] = {     FML(3.6179),   FML(5.3177),   FML(4.6162),   FML(4.9372),   FML(4.0819),
   FML(4.8933),   FML(6.4414),   FML(5.9038),   FML(5.5280),   FML(4.7040),
   FML(5.4276),   FML(6.0795),   FML(6.0630),   FML(5.8480),   FML(5.4499),
   FML(4.8097),   FML(5.6123),   FML(6.1329),   FML(5.6616),   FML(5.5040),
   FML(3.5166),   FML(4.7658),   FML(4.2040),   FML(4.2087),   FML(3.9257),
   FML(3.0627),   FML(4.0956),   FML(3.3257),   FML(3.7015),   FML(2.8260),
  };

    TEST_ALL_KERNELS
}


void test35(const tTest& t)
{
    fml input[] = {     FML(0.928109),   FML(0.072170),   FML(0.329125),   FML(0.123283),   FML(0.132344),
   FML(0.669254),   FML(0.741819),   FML(0.815710),   FML(0.948420),   FML(0.352942),
   FML(0.915476),   FML(0.266938),   FML(0.162948),   FML(0.775167),   FML(0.350660),
   FML(0.744949),   FML(0.666338),   FML(0.673814),   FML(0.063183),   FML(0.530559),
   FML(0.216340),   FML(0.260063),   FML(0.774028),   FML(0.367603),   FML(0.143126),
   FML(0.492394),   FML(0.383510),   FML(0.041004),   FML(0.389502),   FML(0.310170),
  };
    u32 inputRows = 5;
    u32 inputCols = 6;

    fml correctOutput_k33[] = {    FML(1.46718),   FML(1.85338),   FML(1.96425),   FML(1.33072),   FML(1.53500),   FML(0.51069),
   FML(2.40155),   FML(2.60299),   FML(3.48140),   FML(2.70762),   FML(2.29253),   FML(1.01864),
   FML(2.23471),   FML(1.91132),   FML(2.78814),   FML(2.16726),   FML(2.14146),   FML(1.25562),
   FML(2.02967),   FML(2.14195),   FML(2.76615),   FML(2.22204),   FML(2.00155),   FML(1.33788),
   FML(1.29525),   FML(1.77124),   FML(2.26295),   FML(1.69330),   FML(1.37959),   FML(0.94871),
   };

    fml correctOutput_k55[] = {    FML(3.9187),   FML(4.7073),   FML(5.0033),   FML(5.0394),   FML(3.1217),   FML(2.9686),
   FML(4.7681),   FML(5.6568),   FML(5.7619),   FML(6.3621),   FML(4.3162),   FML(3.0452),
   FML(4.5424),   FML(6.2295),   FML(6.9931),   FML(6.6003),   FML(5.0185),   FML(3.1705),
   FML(3.8641),   FML(4.2995),   FML(4.6583),   FML(4.4480),   FML(3.5015),   FML(2.8138),
   FML(3.3390),   FML(3.4154),   FML(3.7691),   FML(4.1461),   FML(2.4719),   FML(1.7475),
   };

    fml correctOutput_k57[] = {     FML(4.2299),   FML(5.1852),   FML(5.0141),   FML(5.0697),   FML(4.7330),   FML(3.3910),
   FML(3.8229),   FML(5.0849),   FML(6.0194),   FML(5.5777),   FML(5.7519),   FML(4.1489),
   FML(4.6336),   FML(5.8091),   FML(6.7553),   FML(6.4377),   FML(5.6624),   FML(4.7543),
   FML(3.3111),   FML(4.6996),   FML(5.3453),   FML(4.4925),   FML(4.0855),   FML(3.5430),
   FML(3.1721),   FML(3.6891),   FML(4.2107),   FML(4.1803),   FML(3.6188),   FML(2.5504),
  };

    TEST_ALL_KERNELS
}


void test36(const tTest& t)
{
    fml input[] = {     FML(0.440269),   FML(0.393502),   FML(0.575961),   FML(0.889772),   FML(0.993714),   FML(0.507609),
   FML(0.553617),   FML(0.838744),   FML(0.153294),   FML(0.509816),   FML(0.089419),   FML(0.589347),
   FML(0.414109),   FML(0.794328),   FML(0.216379),   FML(0.352587),   FML(0.804082),   FML(0.699308),
   FML(0.831077),   FML(0.310952),   FML(0.635974),   FML(0.670740),   FML(0.725298),   FML(0.345503),
   FML(0.604004),   FML(0.060013),   FML(0.914578),   FML(0.605310),   FML(0.608388),   FML(0.748774),
   FML(0.298367),   FML(0.075995),   FML(0.653536),   FML(0.440349),   FML(0.105972),   FML(0.700583),
  };
    u32 inputRows = 6;
    u32 inputCols = 6;

    fml correctOutput_k33[] = {     FML(1.41727),   FML(1.60362),   FML(1.66007),   FML(1.23004),   FML(1.18959),   FML(0.70932),
   FML(1.68673),   FML(2.41732),   FML(2.92226),   FML(2.79336),   FML(2.41944),   FML(1.17129),
   FML(1.62566),   FML(2.52308),   FML(2.86371),   FML(2.64580),   FML(2.06877),   FML(1.08522),
   FML(1.21425),   FML(2.66836),   FML(2.42126),   FML(2.73436),   FML(2.74944),   FML(1.95630),
   FML(1.67143),   FML(3.46979),   FML(2.42871),   FML(3.04670),   FML(2.68766),   FML(1.55932),
   FML(1.31869),   FML(2.74131),   FML(1.94271),   FML(2.85677),   FML(2.16007),   FML(1.45065),
  };

    fml correctOutput_k55[] = {     FML(2.9498),   FML(4.1670),   FML(5.5127),   FML(5.2979),   FML(3.9448),   FML(2.7660),
   FML(4.7087),   FML(5.5942),   FML(6.2758),   FML(5.1333),   FML(4.2223),   FML(3.8850),
   FML(5.2678),   FML(6.3342),   FML(8.5855),   FML(7.4295),   FML(5.9264),   FML(4.1798),
   FML(5.6852),   FML(6.5214),   FML(8.0878),   FML(7.3637),   FML(6.3212),   FML(4.3246),
   FML(4.0961),   FML(4.6369),   FML(6.1427),   FML(5.8133),   FML(4.9194),   FML(3.3956),
   FML(4.3151),   FML(4.2094),   FML(5.1224),   FML(4.7383),   FML(3.8936),   FML(2.5157),
  };

    fml correctOutput_k57[] = {     FML(3.4947),   FML(4.7088),   FML(5.1466),   FML(4.8606),   FML(4.3330),   FML(3.7209),
   FML(4.4545),   FML(5.8804),   FML(5.8461),   FML(5.4960),   FML(4.6078),   FML(4.3930),
   FML(5.1683),   FML(6.7263),   FML(6.8958),   FML(7.5447),   FML(6.5828),   FML(5.8745),
   FML(4.9504),   FML(7.2786),   FML(7.1828),   FML(7.4147),   FML(6.3628),   FML(5.7403),
   FML(4.3188),   FML(6.1123),   FML(5.8051),   FML(6.2339),   FML(4.6292),   FML(4.5331),
   FML(4.4430),   FML(5.3696),   FML(5.6788),   FML(5.3822),   FML(4.3898),   FML(4.1023),
  };

    TEST_ALL_KERNELS
}


void test37(const tTest& t)
{
    fml input[] = {  FML(0.62812),   FML(0.59223),   FML(0.73519),   FML(0.57685),   FML(0.32709),   FML(0.24847),   FML(0.75299),  };
    u32 inputRows = 7;
    u32 inputCols = 1;

    fml correctOutput_k33[] = {     FML(0.35971),
   FML(0.64743),
   FML(0.64099),
   FML(0.69289),
   FML(0.59989),
   FML(0.48194),
   FML(0.48009),
  };

    fml correctOutput_k55[] = {      FML(1.8443),
   FML(2.1363),
   FML(2.3741),
   FML(2.1441),
   FML(2.2316),
   FML(1.7415),
   FML(1.5295),
 };

    fml correctOutput_k57[] = {      FML(1.8436),
   FML(1.9457),
   FML(2.2806),
   FML(2.1147),
   FML(2.1920),
   FML(1.6588),
   FML(1.7905),
 };

    TEST_ALL_KERNELS
}


void test38(const tTest& t)
{
    fml input[] = {      FML(0.95988),
   FML(0.16390),
   FML(0.16823),
   FML(0.90585),
   FML(0.65578),
   FML(0.19132),
   FML(0.49080),
 };
    u32 inputRows = 1;
    u32 inputCols = 7;

    fml correctOutput_k33[] = {  FML(0.50763),   FML(1.03986),   FML(1.11693),   FML(0.98499),   FML(1.06271),   FML(1.10264),   FML(0.46288),  };

    fml correctOutput_k55[] = {  FML(1.6374),   FML(1.9314),   FML(2.2232),   FML(2.0733),   FML(1.9131),   FML(1.3630),   FML(1.0685),  };

    fml correctOutput_k57[] = {  FML(2.5045),   FML(3.1680),   FML(2.4717),   FML(3.0860),   FML(2.8584),   FML(2.1473),   FML(1.8764),  };

    TEST_ALL_KERNELS
}


void test39(const tTest& t)
{
    fml input[] = {     FML(0.260469),   FML(0.461238),   FML(0.766705),   FML(0.330352),   FML(0.369108),   FML(0.765896),   FML(0.245089),
   FML(0.382538),   FML(0.981662),   FML(0.997929),   FML(0.819146),   FML(0.636069),   FML(0.034778),   FML(0.466110),
  };
    u32 inputRows = 7;
    u32 inputCols = 2;

    fml correctOutput_k33[] = {     FML(1.38265),   FML(0.55625),
   FML(2.25560),   FML(1.18605),
   FML(2.65413),   FML(1.79168),
   FML(2.49346),   FML(1.81066),
   FML(1.57009),   FML(1.36372),
   FML(1.36121),   FML(1.45689),
   FML(1.06102),   FML(1.23939),
  };

    fml correctOutput_k55[] = {     FML(2.8026),   FML(2.6834),
   FML(3.5123),   FML(3.2791),
   FML(4.2731),   FML(3.6602),
   FML(4.4533),   FML(3.7722),
   FML(4.3252),   FML(3.2802),
   FML(2.9120),   FML(2.2273),
   FML(2.5031),   FML(1.8414),
  };

    fml correctOutput_k57[] = {     FML(1.9441),   FML(2.7948),
   FML(2.4209),   FML(3.5863),
   FML(2.9263),   FML(4.0811),
   FML(2.9208),   FML(4.0208),
   FML(2.6743),   FML(3.8936),
   FML(2.0826),   FML(2.8731),
   FML(2.0123),   FML(2.3979),
  };

    TEST_ALL_KERNELS
}


void test40(const tTest& t)
{
    fml input[] = {    FML(0.852374),   FML(0.247480),
   FML(0.158633),   FML(0.708827),
   FML(0.473556),   FML(0.081686),
   FML(0.691971),   FML(0.443316),
   FML(0.243482),   FML(0.803716),
   FML(0.696000),   FML(0.730407),
   FML(0.544213),   FML(0.373210),
   };
    u32 inputRows = 2;
    u32 inputCols = 7;

    fml correctOutput_k33[] = {     FML(1.04454),   FML(1.30400),   FML(1.39733),   FML(1.47276),   FML(1.90709),   FML(1.31435),   FML(0.86380),
   FML(1.37249),   FML(1.78662),   FML(1.93394),   FML(1.94612),   FML(2.45197),   FML(2.07117),   FML(1.69926),
  };

    fml correctOutput_k55[] = {     FML(2.1045),   FML(2.5171),   FML(2.6473),   FML(3.2173),   FML(2.9310),   FML(2.8614),   FML(2.3831),
   FML(2.3356),   FML(2.7882),   FML(3.0331),   FML(3.5563),   FML(3.5120),   FML(2.6036),   FML(1.7725),
  };

    fml correctOutput_k57[] = {     FML(2.6624),   FML(3.3171),   FML(3.3503),   FML(3.7098),   FML(3.3801),   FML(3.0359),   FML(3.0956),
   FML(2.6753),   FML(3.6382),   FML(3.9032),   FML(4.1068),   FML(4.0925),   FML(3.3199),   FML(2.8939),
  };

    TEST_ALL_KERNELS
}


void test41(const tTest& t)
{
    fml input[] = {     FML(0.377257),   FML(0.900014),   FML(0.765738),   FML(0.271460),   FML(0.080015),   FML(0.247458),   FML(0.262078),
   FML(0.036116),   FML(0.866383),   FML(0.222909),   FML(0.550992),   FML(0.716977),   FML(0.751364),   FML(0.718367),
   FML(0.070603),   FML(0.187834),   FML(0.142627),   FML(0.361179),   FML(0.408666),   FML(0.597108),   FML(0.504388),
  };
    u32 inputRows = 7;
    u32 inputCols = 3;

    fml correctOutput_k33[] = {     FML(1.03938),   FML(0.84995),   FML(0.43487),
   FML(1.41877),   FML(1.69642),   FML(0.94826),
   FML(1.97415),   FML(2.60421),   FML(1.43215),
   FML(1.81173),   FML(2.05382),   FML(1.03694),
   FML(1.94903),   FML(1.98077),   FML(1.56119),
   FML(1.98863),   FML(2.08794),   FML(1.77262),
   FML(1.50837),   FML(1.90311),   FML(1.77919),
  };

    fml correctOutput_k55[] = {     FML(2.4151),   FML(2.3333),   FML(2.8433),
   FML(3.8223),   FML(3.0497),   FML(2.7306),
   FML(4.0203),   FML(3.4280),   FML(3.1745),
   FML(5.1453),   FML(4.2986),   FML(3.6456),
   FML(4.5085),   FML(4.0582),   FML(3.8055),
   FML(3.6926),   FML(3.6446),   FML(3.0179),
   FML(3.5221),   FML(3.2659),   FML(2.2963),
  };

    fml correctOutput_k57[] = {     FML(1.8668),   FML(2.6268),   FML(2.5797),
   FML(2.8890),   FML(3.7877),   FML(3.0880),
   FML(2.7545),   FML(3.4706),   FML(3.4087),
   FML(3.2652),   FML(3.8171),   FML(4.3429),
   FML(3.0920),   FML(3.3510),   FML(4.3121),
   FML(2.9175),   FML(3.1946),   FML(3.6991),
   FML(2.8908),   FML(2.9866),   FML(3.1453),
  };

    TEST_ALL_KERNELS
}


void test42(const tTest& t)
{
    fml input[] = {    FML(0.981712),   FML(0.743769),   FML(0.969797),
   FML(0.832011),   FML(0.712507),   FML(0.936537),
   FML(0.547834),   FML(0.306557),   FML(0.377040),
   FML(0.025243),   FML(0.399548),   FML(0.681540),
   FML(0.238755),   FML(0.449917),   FML(0.394011),
   FML(0.871334),   FML(0.495486),   FML(0.629243),
   FML(0.321833),   FML(0.667006),   FML(0.540223),
   };
    u32 inputRows = 3;
    u32 inputCols = 7;

    fml correctOutput_k33[] = {     FML(1.57822),   FML(1.72381),   FML(1.27346),   FML(1.21489),   FML(1.44208),   FML(1.33923),   FML(0.92947),
   FML(2.66531),   FML(3.15825),   FML(2.77480),   FML(1.92149),   FML(2.26679),   FML(2.45133),   FML(1.71752),
   FML(1.93342),   FML(2.51355),   FML(2.53024),   FML(1.67939),   FML(2.16821),   FML(2.12766),   FML(1.51568),
  };

    fml correctOutput_k55[] = {    FML(4.7563),   FML(4.6905),   FML(5.1592),   FML(5.1948),   FML(4.6312),   FML(4.0951),   FML(2.9905),
   FML(4.0181),   FML(4.1471),   FML(4.5679),   FML(4.5282),   FML(4.0215),   FML(3.5681),   FML(2.7727),
   FML(4.8384),   FML(4.7579),   FML(4.1697),   FML(3.9911),   FML(4.5066),   FML(3.4442),   FML(2.3401),
   };

    fml correctOutput_k57[] = {     FML(4.0347),   FML(5.2431),   FML(5.7825),   FML(6.2948),   FML(4.7258),   FML(4.2091),   FML(4.0959),
   FML(3.5515),   FML(4.6991),   FML(5.0785),   FML(5.4119),   FML(4.4389),   FML(3.4611),   FML(3.4596),
   FML(4.5560),   FML(5.4966),   FML(5.8033),   FML(5.8551),   FML(5.1452),   FML(3.8067),   FML(3.5006),
  };

    TEST_ALL_KERNELS
}


void test43(const tTest& t)
{
    fml input[] = {    FML(0.5631212),   FML(0.6854290),   FML(0.1699934),   FML(0.9622344),   FML(0.8711099),   FML(0.0058312),   FML(0.0557464),
   FML(0.8263030),   FML(0.3322975),   FML(0.8941075),   FML(0.6458813),   FML(0.7563019),   FML(0.6816131),   FML(0.7481291),
   FML(0.4753393),   FML(0.5369742),   FML(0.8872320),   FML(0.2475427),   FML(0.8469658),   FML(0.3101510),   FML(0.4241769),
   FML(0.4361485),   FML(0.9073743),   FML(0.7913143),   FML(0.7517798),   FML(0.2260671),   FML(0.1231102),   FML(0.7694550),
   };
    u32 inputRows = 7;
    u32 inputCols = 4;

    fml correctOutput_k33[] = {    FML(1.24700),   FML(1.57878),   FML(1.93049),   FML(0.70090),
   FML(2.13362),   FML(3.12127),   FML(3.27506),   FML(1.46862),
   FML(2.05633),   FML(2.64257),   FML(3.37696),   FML(1.87772),
   FML(2.15958),   FML(3.08117),   FML(3.37109),   FML(1.82588),
   FML(2.36898),   FML(3.18153),   FML(2.43185),   FML(1.46450),
   FML(2.32947),   FML(2.69968),   FML(2.79379),   FML(1.47168),
   FML(1.35593),   FML(1.25782),   FML(2.26822),   FML(0.98008),
   };

    fml correctOutput_k55[] = {     FML(4.2565),   FML(4.8763),   FML(4.4660),   FML(3.7817),
   FML(4.5593),   FML(6.0461),   FML(5.9237),   FML(4.8359),
   FML(6.8785),   FML(7.8708),   FML(7.4647),   FML(5.2143),
   FML(5.4290),   FML(6.2498),   FML(6.6426),   FML(5.1297),
   FML(5.8463),   FML(7.0247),   FML(5.9996),   FML(4.8415),
   FML(4.3133),   FML(4.2549),   FML(4.4050),   FML(3.5577),
   FML(3.5336),   FML(4.3052),   FML(3.4891),   FML(2.3296),
  };

    fml correctOutput_k57[] = {     FML(3.8511),   FML(4.0863),   FML(3.9352),   FML(4.3723),
   FML(4.7052),   FML(5.0133),   FML(5.1424),   FML(5.9326),
   FML(5.5739),   FML(6.7223),   FML(6.6277),   FML(6.7112),
   FML(5.0809),   FML(6.0292),   FML(5.2168),   FML(5.9819),
   FML(5.2225),   FML(6.0867),   FML(5.5056),   FML(5.8110),
   FML(3.9450),   FML(3.8253),   FML(4.3267),   FML(3.9370),
   FML(3.6700),   FML(3.6709),   FML(4.0393),   FML(3.3951),
  };

    TEST_ALL_KERNELS
}


void test44(const tTest& t)
{
    fml input[] = {     FML(0.641772),   FML(0.220415),   FML(0.774901),   FML(0.262930),
   FML(0.625038),   FML(0.574776),   FML(0.574666),   FML(0.607177),
   FML(0.790824),   FML(0.593115),   FML(0.392120),   FML(0.412882),
   FML(0.593198),   FML(0.805159),   FML(0.292678),   FML(0.714963),
   FML(0.529652),   FML(0.723107),   FML(0.715465),   FML(0.755291),
   FML(0.428796),   FML(0.042279),   FML(0.630844),   FML(0.126480),
   FML(0.832725),   FML(0.534621),   FML(0.510347),   FML(0.648105),
  };
    u32 inputRows = 4;
    u32 inputCols = 7;

    fml correctOutput_k33[] = {     FML(1.28276),   FML(1.84573),   FML(1.90100),   FML(1.87892),   FML(1.17241),   FML(1.80236),   FML(0.64414),
   FML(1.93913),   FML(2.77988),   FML(2.98653),   FML(3.27966),   FML(2.52005),   FML(2.91931),   FML(1.24227),
   FML(1.77535),   FML(2.36460),   FML(2.91339),   FML(3.18804),   FML(2.34113),   FML(2.85123),   FML(1.04050),
   FML(1.54503),   FML(2.10492),   FML(2.19086),   FML(2.19592),   FML(1.94407),   FML(2.60166),   FML(1.27121),
  };

    fml correctOutput_k55[] = {     FML(3.9393),   FML(4.3731),   FML(5.4160),   FML(5.3760),   FML(5.4924),   FML(4.3206),   FML(3.5090),
   FML(4.5026),   FML(5.8960),   FML(7.3420),   FML(6.2900),   FML(6.2659),   FML(4.8895),   FML(4.1516),
   FML(4.1522),   FML(4.8796),   FML(5.8420),   FML(5.8866),   FML(5.9351),   FML(5.0527),   FML(3.5359),
   FML(3.3849),   FML(4.1743),   FML(5.0544),   FML(4.7793),   FML(4.1792),   FML(3.4279),   FML(2.5975),
  };

    fml correctOutput_k57[] = {     FML(3.8065),   FML(5.1098),   FML(5.5505),   FML(6.4698),   FML(5.4242),   FML(4.7635),   FML(4.0374),
   FML(4.3427),   FML(6.0873),   FML(6.1122),   FML(7.1409),   FML(6.7277),   FML(5.2964),   FML(4.7865),
   FML(4.3141),   FML(5.6704),   FML(6.0171),   FML(6.3962),   FML(6.4452),   FML(5.3010),   FML(4.2349),
   FML(3.6648),   FML(4.9176),   FML(5.3501),   FML(5.5740),   FML(5.3794),   FML(3.9935),   FML(3.5429),
  };

    TEST_ALL_KERNELS
}


void test45(const tTest& t)
{
    fml input[] = {     FML(8.0628e-01),   FML(3.0888e-01),   FML(4.1754e-01),   FML(7.8709e-02),   FML(1.3436e-01),   FML(2.6268e-01),   FML(7.2130e-01),
   FML(6.5611e-01),   FML(7.8068e-01),   FML(1.7351e-01),   FML(5.7752e-01),   FML(4.0148e-01),   FML(3.0513e-01),   FML(6.6679e-02),
   FML(8.0438e-01),   FML(3.7096e-01),   FML(8.8364e-01),   FML(5.7586e-01),   FML(7.8524e-01),   FML(8.0289e-01),   FML(6.4054e-01),
   FML(3.8244e-01),   FML(5.3523e-01),   FML(1.1786e-04),   FML(4.6880e-01),   FML(9.0107e-01),   FML(9.0064e-01),   FML(2.0411e-01),
   FML(6.1963e-01),   FML(2.5341e-01),   FML(5.5505e-01),   FML(8.4981e-01),   FML(7.4713e-02),   FML(1.3433e-02),   FML(2.6517e-01),
  };
    u32 inputRows = 7;
    u32 inputCols = 5;

    fml correctOutput_k33[] = {     FML(1.47916),   FML(1.79968),   FML(1.57273),   FML(1.54982),   FML(0.65350),
   FML(1.88906),   FML(3.19623),   FML(2.51121),   FML(2.66862),   FML(1.30961),
   FML(1.59698),   FML(2.62253),   FML(2.19602),   FML(2.77668),   FML(1.03588),
   FML(1.37926),   FML(2.55088),   FML(2.38760),   FML(2.72640),   FML(1.02130),
   FML(1.29544),   FML(2.41263),   FML(3.18056),   FML(2.37312),   FML(1.80870),
   FML(0.95518),   FML(2.53848),   FML(2.78092),   FML(2.37559),   FML(1.77993),
   FML(0.74455),   FML(2.18947),   FML(1.83270),   FML(2.10982),   FML(1.31695),
  };

    fml correctOutput_k55[] = {     FML(4.0508),   FML(3.9917),   FML(4.6917),   FML(3.8164),   FML(2.9790),
   FML(4.4782),   FML(5.0571),   FML(6.1006),   FML(4.7825),   FML(4.0079),
   FML(5.1519),   FML(6.2085),   FML(6.6223),   FML(5.8524),   FML(4.6827),
   FML(4.9245),   FML(5.8594),   FML(6.8641),   FML(5.8788),   FML(4.6499),
   FML(4.3097),   FML(6.0447),   FML(6.5846),   FML(5.1421),   FML(4.3813),
   FML(3.7042),   FML(5.0791),   FML(5.1811),   FML(4.4139),   FML(3.1652),
   FML(3.1221),   FML(3.8409),   FML(4.1118),   FML(3.0131),   FML(2.0954),
  };

    fml correctOutput_k57[] = {     FML(3.4620),   FML(4.6323),   FML(4.5408),   FML(4.2872),   FML(3.8338),
   FML(3.9003),   FML(5.2210),   FML(5.0509),   FML(4.9090),   FML(4.6394),
   FML(4.6754),   FML(5.3681),   FML(5.6303),   FML(6.0222),   FML(5.2644),
   FML(4.8421),   FML(4.8389),   FML(5.8010),   FML(6.0367),   FML(5.7674),
   FML(4.6058),   FML(4.8624),   FML(6.0639),   FML(5.6499),   FML(5.2365),
   FML(3.9766),   FML(5.0745),   FML(4.3049),   FML(4.8958),   FML(4.1575),
   FML(3.7241),   FML(4.1838),   FML(3.2210),   FML(3.9949),   FML(2.9565),
  };

    TEST_ALL_KERNELS
}


void test46(const tTest& t)
{
    fml input[] = {     FML(0.099997),   FML(0.634130),   FML(0.808520),   FML(0.252147),   FML(0.534630),
   FML(0.054143),   FML(0.470317),   FML(0.718463),   FML(0.601712),   FML(0.766503),
   FML(0.839265),   FML(0.156486),   FML(0.343562),   FML(0.519968),   FML(0.641296),
   FML(0.898705),   FML(0.397929),   FML(0.357643),   FML(0.815803),   FML(0.801529),
   FML(0.322900),   FML(0.708781),   FML(0.529772),   FML(0.241708),   FML(0.967218),
   FML(0.931585),   FML(0.314035),   FML(0.244513),   FML(0.406186),   FML(0.949873),
   FML(0.909840),   FML(0.959574),   FML(0.209308),   FML(0.584359),   FML(0.722951),
  };
    u32 inputRows = 5;
    u32 inputCols = 7;

    fml correctOutput_k33[] = {     FML(0.72380),   FML(1.21408),   FML(1.45854),   FML(1.71000),   FML(1.89406),   FML(2.10208),   FML(0.99363),
   FML(1.36140),   FML(1.91232),   FML(2.32965),   FML(2.86468),   FML(2.72494),   FML(3.08962),   FML(1.91089),
   FML(2.00331),   FML(2.48719),   FML(2.54617),   FML(2.02326),   FML(2.08593),   FML(2.77219),   FML(1.26540),
   FML(2.24986),   FML(2.80691),   FML(3.14367),   FML(2.55751),   FML(2.72451),   FML(2.37255),   FML(1.04523),
   FML(1.46621),   FML(2.06249),   FML(2.81657),   FML(2.54435),   FML(2.77238),   FML(2.34995),   FML(1.60422),
  };

    fml correctOutput_k55[] = {     FML(3.3237),   FML(4.3117),   FML(5.3654),   FML(5.0756),   FML(4.7671),   FML(4.0725),   FML(3.4240),
   FML(3.9149),   FML(5.0585),   FML(5.9953),   FML(6.3862),   FML(6.8658),   FML(5.3971),   FML(3.8850),
   FML(4.7592),   FML(6.2915),   FML(8.0882),   FML(8.0233),   FML(8.5353),   FML(7.4622),   FML(5.0600),
   FML(4.1507),   FML(4.8838),   FML(5.5986),   FML(5.7421),   FML(5.2845),   FML(5.2283),   FML(4.1002),
   FML(4.2199),   FML(4.6897),   FML(5.0234),   FML(5.4106),   FML(4.9742),   FML(3.5872),   FML(2.3457),
  };

    fml correctOutput_k57[] = {     FML(3.2239),   FML(4.4665),   FML(5.1587),   FML(6.4628),   FML(5.3074),   FML(4.5932),   FML(4.3712),
   FML(4.0193),   FML(5.1999),   FML(5.1812),   FML(7.1275),   FML(6.7253),   FML(5.6272),   FML(4.9651),
   FML(4.9893),   FML(6.6082),   FML(7.7407),   FML(8.4739),   FML(8.0852),   FML(6.7037),   FML(6.4130),
   FML(3.9983),   FML(5.3420),   FML(6.0603),   FML(6.3066),   FML(5.9200),   FML(4.5537),   FML(4.7564),
   FML(4.3298),   FML(5.6955),   FML(5.9560),   FML(6.5166),   FML(6.1203),   FML(4.6751),   FML(3.9364),
  };

    TEST_ALL_KERNELS
}


void test47(const tTest& t)
{
    fml input[] = {     FML(0.869328),   FML(0.490232),   FML(0.460184),   FML(0.698951),   FML(0.517681),   FML(0.743976),   FML(0.673771),
   FML(0.937985),   FML(0.195023),   FML(0.527406),   FML(0.627390),   FML(0.973239),   FML(0.139031),   FML(0.919978),
   FML(0.539641),   FML(0.339154),   FML(0.712480),   FML(0.853650),   FML(0.782541),   FML(0.073868),   FML(0.270816),
   FML(0.741749),   FML(0.482605),   FML(0.866943),   FML(0.143933),   FML(0.187946),   FML(0.993527),   FML(0.176620),
   FML(0.535939),   FML(0.147685),   FML(0.250232),   FML(0.988947),   FML(0.655057),   FML(0.684019),   FML(0.403854),
   FML(0.048192),   FML(0.529587),   FML(0.267345),   FML(0.355980),   FML(0.113033),   FML(0.386825),   FML(0.840679),
  };
    u32 inputRows = 7;
    u32 inputCols = 6;

    fml correctOutput_k33[] = {     FML(1.24935),   FML(1.64935),   FML(1.89388),   FML(1.26487),   FML(1.30773),   FML(0.66092),
   FML(1.94918),   FML(3.12124),   FML(3.20949),   FML(2.17050),   FML(2.31623),   FML(1.00198),
   FML(1.59707),   FML(2.71169),   FML(2.21480),   FML(2.51483),   FML(2.25424),   FML(0.98196),
   FML(2.17255),   FML(3.28211),   FML(2.55692),   FML(3.45013),   FML(1.98343),   FML(1.38067),
   FML(1.95968),   FML(2.98583),   FML(2.98522),   FML(3.40730),   FML(1.81452),   FML(1.90554),
   FML(2.09276),   FML(2.60289),   FML(2.82146),   FML(2.58044),   FML(2.50004),   FML(1.48575),
   FML(1.50460),   FML(1.82469),   FML(1.87753),   FML(1.79488),   FML(2.61962),   FML(1.45071),
  };

    fml correctOutput_k55[] = {     FML(4.2159),   FML(4.9568),   FML(5.1494),   FML(4.2696),   FML(3.3385),   FML(2.5980),
   FML(4.7860),   FML(5.4748),   FML(6.7571),   FML(6.3443),   FML(4.9815),   FML(3.4400),
   FML(6.4808),   FML(7.0186),   FML(8.2088),   FML(7.4013),   FML(5.1994),   FML(3.1864),
   FML(5.0884),   FML(6.2813),   FML(8.2016),   FML(6.9866),   FML(5.8217),   FML(4.1434),
   FML(6.3607),   FML(6.3790),   FML(8.1455),   FML(6.9825),   FML(5.3104),   FML(4.3223),
   FML(4.1957),   FML(5.2684),   FML(6.0936),   FML(6.1145),   FML(4.3891),   FML(2.7044),
   FML(3.9920),   FML(4.3435),   FML(3.8407),   FML(4.4326),   FML(3.5607),   FML(2.4333),
  };

    fml correctOutput_k57[] = {     FML(4.3962),   FML(5.1076),   FML(4.9072),   FML(4.9408),   FML(4.5425),   FML(3.5766),
   FML(3.8174),   FML(5.5844),   FML(5.9899),   FML(6.0093),   FML(5.5917),   FML(4.3819),
   FML(5.1762),   FML(7.3164),   FML(7.1575),   FML(7.8169),   FML(6.2798),   FML(5.0225),
   FML(5.3496),   FML(6.9625),   FML(7.0464),   FML(7.3697),   FML(5.4430),   FML(5.9597),
   FML(5.4605),   FML(7.0186),   FML(8.1193),   FML(7.1770),   FML(6.0399),   FML(5.2783),
   FML(4.2194),   FML(5.5722),   FML(6.5108),   FML(5.5835),   FML(5.3071),   FML(3.8862),
   FML(3.6996),   FML(5.2556),   FML(5.2938),   FML(4.4422),   FML(4.2104),   FML(3.5236),
  };

    TEST_ALL_KERNELS
}


void test48(const tTest& t)
{
    fml input[] = {     FML(0.442192),   FML(0.884189),   FML(0.650267),   FML(0.263203),   FML(0.473883),   FML(0.053294),
   FML(0.323169),   FML(0.434060),   FML(0.853767),   FML(0.350068),   FML(0.127065),   FML(0.931383),
   FML(0.092343),   FML(0.197101),   FML(0.680829),   FML(0.922208),   FML(0.362439),   FML(0.125061),
   FML(0.446166),   FML(0.818741),   FML(0.528226),   FML(0.692816),   FML(0.104809),   FML(0.912774),
   FML(0.206464),   FML(0.640395),   FML(0.377323),   FML(0.761235),   FML(0.937604),   FML(0.110223),
   FML(0.582906),   FML(0.839552),   FML(0.275248),   FML(0.087632),   FML(0.521961),   FML(0.175300),
   FML(0.975041),   FML(0.246553),   FML(0.220100),   FML(0.508040),   FML(0.189594),   FML(0.995401),
  };
    u32 inputRows = 6;
    u32 inputCols = 7;

    fml correctOutput_k33[] = {    FML(0.93087),   FML(0.93153),   FML(1.53942),   FML(1.07990),   FML(1.79215),   FML(1.49798),   FML(0.83943),
   FML(1.79871),   FML(2.27597),   FML(2.38503),   FML(1.79221),   FML(2.70120),   FML(2.30477),   FML(1.88045),
   FML(1.99099),   FML(3.21913),   FML(2.93837),   FML(2.75951),   FML(2.67529),   FML(2.37578),   FML(1.41992),
   FML(1.58264),   FML(3.01940),   FML(2.71352),   FML(3.43073),   FML(2.14721),   FML(2.06698),   FML(0.82490),
   FML(1.52273),   FML(2.04001),   FML(2.53988),   FML(3.08346),   FML(2.15689),   FML(2.96691),   FML(0.98724),
   FML(1.33223),   FML(1.26663),   FML(1.95136),   FML(1.57583),   FML(1.90192),   FML(2.41665),   FML(1.07682),
   };

    fml correctOutput_k55[] = {     FML(3.5298),   FML(4.0864),   FML(4.8213),   FML(4.7800),   FML(4.8333),   FML(4.3940),   FML(3.1179),
   FML(4.0968),   FML(5.3398),   FML(6.4651),   FML(6.7434),   FML(6.6303),   FML(4.8246),   FML(3.4283),
   FML(4.8966),   FML(5.3284),   FML(6.8849),   FML(6.6568),   FML(6.7151),   FML(5.8878),   FML(4.5123),
   FML(5.2949),   FML(6.2516),   FML(7.7204),   FML(6.8755),   FML(6.8886),   FML(5.6150),   FML(3.9017),
   FML(3.7954),   FML(4.3481),   FML(5.5067),   FML(5.8836),   FML(4.9984),   FML(3.9483),   FML(2.4978),
   FML(2.6950),   FML(3.9864),   FML(4.2831),   FML(4.0787),   FML(4.0681),   FML(3.4523),   FML(2.6851),
  };

    fml correctOutput_k57[] = {     FML(3.1215),   FML(4.1910),   FML(4.5021),   FML(5.4749),   FML(5.3457),   FML(4.3583),   FML(4.3447),
   FML(4.0247),   FML(5.7189),   FML(5.7089),   FML(6.9964),   FML(6.9067),   FML(5.3516),   FML(4.9138),
   FML(4.5543),   FML(6.6965),   FML(6.7193),   FML(7.5385),   FML(6.1463),   FML(5.7268),   FML(5.1081),
   FML(5.1710),   FML(6.6146),   FML(6.7421),   FML(8.7567),   FML(6.8336),   FML(6.0956),   FML(5.0619),
   FML(3.8609),   FML(4.6917),   FML(5.5883),   FML(5.6322),   FML(5.7289),   FML(4.4808),   FML(3.3604),
   FML(3.5815),   FML(4.0013),   FML(5.0956),   FML(4.7381),   FML(5.1174),   FML(3.6227),   FML(3.7786),
  };

    TEST_ALL_KERNELS
}


void test49(const tTest& t)
{
    fml input[] = {     FML(0.321239),   FML(0.176561),   FML(0.169673),   FML(0.999892),   FML(0.850074),   FML(0.164117),   FML(0.176474),
   FML(0.688657),   FML(0.911181),   FML(0.967740),   FML(0.909265),   FML(0.521753),   FML(0.295321),   FML(0.477024),
   FML(0.573646),   FML(0.999096),   FML(0.605754),   FML(0.411062),   FML(0.622976),   FML(0.476222),   FML(0.332373),
   FML(0.158697),   FML(0.385260),   FML(0.924634),   FML(0.735964),   FML(0.108748),   FML(0.041740),   FML(0.339003),
   FML(0.938489),   FML(0.592278),   FML(0.776579),   FML(0.992382),   FML(0.722160),   FML(0.977603),   FML(0.663297),
   FML(0.563794),   FML(0.407637),   FML(0.470082),   FML(0.028553),   FML(0.441672),   FML(0.665066),   FML(0.887846),
   FML(0.254693),   FML(0.763865),   FML(0.316972),   FML(0.357320),   FML(0.792243),   FML(0.285545),   FML(0.582162),
  };
    u32 inputRows = 7;
    u32 inputCols = 7;

    fml correctOutput_k33[] = {     FML(1.56629),   FML(1.79736),   FML(1.30603),   FML(1.96804),   FML(1.27394),   FML(1.78502),   FML(0.72547),
   FML(2.38979),   FML(2.76480),   FML(3.09998),   FML(3.34529),   FML(2.34781),   FML(2.95527),   FML(1.32751),
   FML(2.47885),   FML(2.66329),   FML(3.92816),   FML(3.70231),   FML(2.33004),   FML(2.73049),   FML(1.36689),
   FML(2.23387),   FML(2.91424),   FML(3.44664),   FML(3.54528),   FML(2.77848),   FML(3.06576),   FML(1.00175),
   FML(2.08755),   FML(3.39139),   FML(2.38918),   FML(3.47487),   FML(2.47251),   FML(2.93461),   FML(0.91042),
   FML(1.66582),   FML(2.55401),   FML(1.73513),   FML(3.04564),   FML(2.38826),   FML(3.14592),   FML(1.62976),
   FML(0.96521),   FML(1.32692),   FML(1.41630),   FML(2.19040),   FML(2.19584),   FML(2.66505),   FML(1.66380),
  };

    fml correctOutput_k55[] = {     FML(3.7880),   FML(4.5656),   FML(5.8533),   FML(6.5307),   FML(5.7255),   FML(4.1181),   FML(3.1565),
   FML(5.3263),   FML(6.2562),   FML(7.9954),   FML(7.3555),   FML(6.5515),   FML(5.4806),   FML(4.1156),
   FML(6.3615),   FML(7.4711),   FML(8.9771),   FML(8.7453),   FML(8.2766),   FML(5.6947),   FML(4.7916),
   FML(5.8238),   FML(6.9854),   FML(8.5216),   FML(8.0978),   FML(7.5066),   FML(5.4352),   FML(4.9483),
   FML(5.5325),   FML(5.8991),   FML(7.5560),   FML(7.9064),   FML(8.0580),   FML(6.2159),   FML(5.1238),
   FML(4.2910),   FML(3.9283),   FML(5.6362),   FML(6.1644),   FML(5.5066),   FML(4.4498),   FML(3.7740),
   FML(3.0758),   FML(3.1608),   FML(3.9289),   FML(4.6432),   FML(4.8608),   FML(4.0527),   FML(2.8857),
  };

    fml correctOutput_k57[] = {     FML(3.3519),   FML(4.7218),   FML(5.6364),   FML(6.1794),   FML(5.6934),   FML(5.3766),   FML(4.1017),
   FML(4.7423),   FML(6.1797),   FML(7.6408),   FML(7.9553),   FML(7.0880),   FML(5.9770),   FML(5.1468),
   FML(5.2826),   FML(7.5794),   FML(9.0683),   FML(9.1157),   FML(8.6742),   FML(6.4695),   FML(5.1811),
   FML(5.0101),   FML(8.2285),   FML(8.1036),   FML(8.2189),   FML(7.6703),   FML(6.4609),   FML(5.1634),
   FML(4.5656),   FML(7.3005),   FML(7.7763),   FML(8.1485),   FML(7.1793),   FML(7.0314),   FML(5.8540),
   FML(3.6940),   FML(5.0145),   FML(6.2236),   FML(6.1746),   FML(5.7385),   FML(5.0032),   FML(4.3200),
   FML(3.1798),   FML(3.6937),   FML(5.4258),   FML(5.3445),   FML(5.0998),   FML(4.4748),   FML(4.1012),
  };

    TEST_ALL_KERNELS
}


void test50(const tTest& t)
{
    fml input[] = {  FML(0.167708),   FML(0.039835),   FML(0.140912),   FML(0.630703),   FML(0.338575),   FML(0.182834),   FML(0.703168),   FML(0.337239),  };
    u32 inputRows = 8;
    u32 inputCols = 1;

    fml correctOutput_k33[] = {     FML(0.32013),
   FML(0.38814),
   FML(0.34031),
   FML(0.42320),
   FML(0.62517),
   FML(0.48178),
   FML(0.44817),
   FML(0.65727),
  };

    fml correctOutput_k55[] = {     FML(0.88025),
   FML(1.22568),
   FML(1.40051),
   FML(1.48728),
   FML(1.84205),
   FML(1.91207),
   FML(1.63412),
   FML(1.39096),
  };

    fml correctOutput_k57[] = {     FML(1.1425),
   FML(1.3403),
   FML(1.3946),
   FML(1.6190),
   FML(1.8330),
   FML(1.7994),
   FML(1.7772),
   FML(1.5217),
  };

    TEST_ALL_KERNELS
}


void test51(const tTest& t)
{
    fml input[] = {     FML(0.176728),
   FML(0.011971),
   FML(0.083009),
   FML(0.426775),
   FML(0.151098),
   FML(0.091378),
   FML(0.636817),
   FML(0.843213),
  };
    u32 inputRows = 1;
    u32 inputCols = 8;

    fml correctOutput_k33[] = {  FML(0.32977),   FML(0.48032),   FML(0.64859),   FML(0.50731),   FML(0.65156),   FML(0.89631),   FML(1.06071),   FML(0.76468),  };

    fml correctOutput_k55[] = {  FML(0.86937),   FML(1.16376),   FML(1.22621),   FML(1.18902),   FML(1.51446),   FML(2.09063),   FML(1.84669),   FML(1.37639),  };

    fml correctOutput_k57[] = {  FML(1.4234),   FML(1.5770),   FML(1.4830),   FML(2.0033),   FML(2.5367),   FML(2.2374),   FML(2.2704),   FML(2.2598),  };

    TEST_ALL_KERNELS
}


void test52(const tTest& t)
{
    fml input[] = {    FML(0.063015),   FML(0.267807),   FML(0.827856),   FML(0.686066),   FML(0.644949),   FML(0.735767),   FML(0.553552),   FML(0.639133),
   FML(0.999616),   FML(0.707905),   FML(0.306720),   FML(0.702924),   FML(0.699043),   FML(0.885282),   FML(0.489594),   FML(0.623118),
   };
    u32 inputRows = 8;
    u32 inputCols = 2;

    fml correctOutput_k33[] = {     FML(1.62863),   FML(0.45889),
   FML(1.82838),   FML(1.14447),
   FML(1.76346),   FML(1.50492),
   FML(2.03679),   FML(1.79215),
   FML(2.38132),   FML(1.82579),
   FML(2.20295),   FML(1.83184),
   FML(2.15793),   FML(1.87178),
   FML(1.42628),   FML(1.50174),
  };

    fml correctOutput_k55[] = {      FML(2.5418),   FML(2.4807),
   FML(3.3288),   FML(3.0964),
   FML(4.3642),   FML(3.5748),
   FML(4.7493),   FML(3.8576),
   FML(4.4239),   FML(3.6455),
   FML(4.9144),   FML(3.8392),
   FML(3.7510),   FML(3.0477),
   FML(3.4635),   FML(2.3901),
 };

    fml correctOutput_k57[] = {      FML(2.1572),   FML(2.6184),
   FML(2.1952),   FML(3.3784),
   FML(2.5870),   FML(4.1484),
   FML(3.0239),   FML(4.4607),
   FML(3.1290),   FML(4.1963),
   FML(3.3138),   FML(4.5368),
   FML(2.5720),   FML(3.5840),
   FML(2.6435),   FML(3.1862),
 };

    TEST_ALL_KERNELS
}


void test53(const tTest& t)
{
    fml input[] = {     FML(0.759124),   FML(0.431274),
   FML(0.438353),   FML(0.150368),
   FML(0.243045),   FML(0.338389),
   FML(0.697792),   FML(0.233137),
   FML(0.013596),   FML(0.827095),
   FML(0.827519),   FML(0.373287),
   FML(0.352845),   FML(0.505877),
   FML(0.967431),   FML(0.432089),
  };
    u32 inputRows = 2;
    u32 inputCols = 8;

    fml correctOutput_k33[] = {     FML(0.82191),   FML(1.30696),   FML(1.33145),   FML(1.20067),   FML(1.69397),   FML(1.13884),   FML(1.96718),   FML(0.66024),
   FML(1.10642),   FML(1.93563),   FML(1.61268),   FML(1.73136),   FML(2.03966),   FML(1.87021),   FML(2.52190),   FML(1.43396),
  };

    fml correctOutput_k55[] = {     FML(2.0908),   FML(2.3282),   FML(2.3175),   FML(2.6039),   FML(2.7826),   FML(3.3975),   FML(3.0803),   FML(2.2096),
   FML(2.1578),   FML(2.4592),   FML(2.7948),   FML(3.0937),   FML(3.0607),   FML(3.3549),   FML(2.4588),   FML(2.0282),
  };

    fml correctOutput_k57[] = {     FML(2.5786),   FML(3.2127),   FML(2.9380),   FML(3.4202),   FML(3.7329),   FML(3.4167),   FML(3.4635),   FML(2.7429),
   FML(2.2954),   FML(3.4265),   FML(3.0919),   FML(4.1276),   FML(3.7485),   FML(4.1274),   FML(2.8377),   FML(2.8797),
  };

    TEST_ALL_KERNELS
}


void test54(const tTest& t)
{
    fml input[] = {     FML(0.916870),   FML(0.199643),   FML(0.719588),   FML(0.165928),   FML(0.823608),   FML(0.935213),   FML(0.664849),   FML(0.938296),
   FML(0.026034),   FML(0.989207),   FML(0.536126),   FML(0.333578),   FML(0.869171),   FML(0.134870),   FML(0.539500),   FML(0.989631),
   FML(0.764197),   FML(0.558166),   FML(0.739754),   FML(0.412645),   FML(0.913601),   FML(0.724047),   FML(0.943993),   FML(0.375823),
  };
    u32 inputRows = 8;
    u32 inputCols = 3;

    fml correctOutput_k33[] = {     FML(1.16401),   FML(1.92127),   FML(0.49928),
   FML(1.94208),   FML(3.00216),   FML(1.40271),
   FML(1.80386),   FML(2.72892),   FML(1.93476),
   FML(1.95235),   FML(2.99740),   FML(1.50137),
   FML(1.45451),   FML(2.84574),   FML(1.44132),
   FML(1.88013),   FML(4.06975),   FML(1.76310),
   FML(2.06773),   FML(3.33692),   FML(1.29052),
   FML(1.81679),   FML(2.77938),   FML(1.89435),
  };

    fml correctOutput_k55[] = {     FML(3.9059),   FML(3.4439),   FML(3.4899),
   FML(4.4336),   FML(3.9498),   FML(3.7667),
   FML(6.0001),   FML(5.5483),   FML(5.1813),
   FML(5.6581),   FML(5.4873),   FML(5.3451),
   FML(6.5806),   FML(5.8207),   FML(5.4070),
   FML(6.1880),   FML(5.5150),   FML(5.6072),
   FML(5.6003),   FML(5.3016),   FML(4.6942),
   FML(4.4443),   FML(3.7809),   FML(2.9610),
  };

    fml correctOutput_k57[] = {    FML(2.8463),   FML(3.3278),   FML(3.5573),
   FML(3.3360),   FML(3.6322),   FML(3.9888),
   FML(4.5065),   FML(4.4809),   FML(5.4133),
   FML(3.6891),   FML(4.2465),   FML(5.0170),
   FML(4.7425),   FML(5.2598),   FML(5.2903),
   FML(4.3043),   FML(5.0122),   FML(5.0887),
   FML(4.5010),   FML(4.8590),   FML(4.8394),
   FML(4.4191),   FML(3.8667),   FML(3.8367),
   };

    TEST_ALL_KERNELS
}


void test55(const tTest& t)
{
    fml input[] = {     FML(0.561970),   FML(0.852803),   FML(0.979483),
   FML(0.486507),   FML(0.255674),   FML(0.816477),
   FML(0.216876),   FML(0.592863),   FML(0.611712),
   FML(0.744141),   FML(0.079334),   FML(0.493236),
   FML(0.186499),   FML(0.026388),   FML(0.750921),
   FML(0.820644),   FML(0.530973),   FML(0.165772),
   FML(0.941147),   FML(0.404819),   FML(0.651738),
   FML(0.662034),   FML(0.406593),   FML(0.104598),
  };
    u32 inputRows = 3;
    u32 inputCols = 8;

    fml correctOutput_k33[] = {      FML(0.92769),   FML(1.41476),   FML(1.28956),   FML(0.73043),   FML(1.82968),   FML(1.52821),   FML(1.77183),   FML(0.98691),
   FML(1.79972),   FML(2.80557),   FML(2.12330),   FML(2.02916),   FML(2.31226),   FML(2.46716),   FML(2.75094),   FML(1.86439),
   FML(1.57904),   FML(2.77995),   FML(1.80742),   FML(1.91816),   FML(1.24812),   FML(1.83030),   FML(1.51101),   FML(1.28915),

 };

    fml correctOutput_k55[] = {     FML(3.9918),   FML(4.5292),   FML(5.5638),   FML(4.8278),   FML(5.1539),   FML(4.7153),   FML(3.7866),   FML(2.7225),
   FML(3.4138),   FML(3.6767),   FML(3.9226),   FML(3.9835),   FML(4.2336),   FML(4.3327),   FML(3.6648),   FML(2.5801),
   FML(4.3929),   FML(3.8573),   FML(4.3462),   FML(3.5978),   FML(3.9627),   FML(4.1265),   FML(3.3419),   FML(2.3249),
  };

    fml correctOutput_k57[] = {    FML(3.5417),   FML(4.9365),   FML(5.0988),   FML(6.6873),   FML(6.0214),   FML(4.7025),   FML(5.0545),   FML(4.0065),
   FML(2.9868),   FML(4.4250),   FML(3.7667),   FML(5.0430),   FML(4.3192),   FML(4.2056),   FML(3.4886),   FML(3.3867),
   FML(4.4857),   FML(5.2167),   FML(4.9051),   FML(6.0170),   FML(5.0334),   FML(4.5792),   FML(3.5453),   FML(3.5179),
   };

    TEST_ALL_KERNELS
}


void test56(const tTest& t)
{
    fml input[] = {     FML(0.816713),   FML(0.354610),   FML(0.975125),   FML(0.979920),   FML(0.965844),   FML(0.108280),   FML(0.074736),   FML(0.421862),
   FML(0.095001),   FML(0.446746),   FML(0.524352),   FML(0.824851),   FML(0.887797),   FML(0.700142),   FML(0.914434),   FML(0.011422),
   FML(0.183717),   FML(0.458862),   FML(0.661868),   FML(0.372408),   FML(0.558474),   FML(0.863554),   FML(0.295969),   FML(0.511900),
   FML(0.811456),   FML(0.530750),   FML(0.833128),   FML(0.843267),   FML(0.201142),   FML(0.325841),   FML(0.850899),   FML(0.471149),
  };
    u32 inputRows = 8;
    u32 inputCols = 4;

    fml correctOutput_k33[] = {     FML(0.79106),   FML(1.35631),   FML(1.46743),   FML(0.53921),
   FML(1.53101),   FML(2.48644),   FML(2.46499),   FML(1.26097),
   FML(1.89936),   FML(2.72016),   FML(3.07065),   FML(1.50910),
   FML(2.51846),   FML(3.43250),   FML(3.12788),   FML(1.68142),
   FML(2.62325),   FML(3.65282),   FML(2.92427),   FML(1.51064),
   FML(2.61414),   FML(3.04677),   FML(3.06041),   FML(1.52791),
   FML(1.55676),   FML(2.11327),   FML(3.20368),   FML(1.58959),
   FML(1.00972),   FML(1.65640),   FML(2.30625),   FML(1.33283),
  };

    fml correctOutput_k55[] = {     FML(3.3927),   FML(4.4042),   FML(4.5083),   FML(3.4295),
   FML(4.5514),   FML(5.9169),   FML(5.9866),   FML(4.2671),
   FML(6.1403),   FML(6.5961),   FML(7.3492),   FML(5.0713),
   FML(6.3422),   FML(7.2981),   FML(6.8880),   FML(5.2282),
   FML(6.2056),   FML(7.0413),   FML(6.6609),   FML(5.2720),
   FML(5.7705),   FML(6.2787),   FML(6.1854),   FML(4.5435),
   FML(4.4399),   FML(4.9870),   FML(4.4912),   FML(3.2023),
   FML(2.8606),   FML(4.1099),   FML(3.3987),   FML(2.6085),
  };

    fml correctOutput_k57[] = {     FML(3.7888),   FML(3.8358),   FML(3.6106),   FML(4.5689),
   FML(4.3059),   FML(5.2608),   FML(4.6188),   FML(5.4747),
   FML(5.6333),   FML(6.5527),   FML(5.9590),   FML(6.7242),
   FML(5.4274),   FML(6.6445),   FML(5.8584),   FML(6.1701),
   FML(5.7491),   FML(6.3955),   FML(5.7480),   FML(6.3514),
   FML(5.2082),   FML(5.0932),   FML(5.7976),   FML(5.5606),
   FML(4.3528),   FML(4.3927),   FML(4.4301),   FML(4.2164),
   FML(3.0142),   FML(3.7232),   FML(3.1016),   FML(3.4844),
  };

    TEST_ALL_KERNELS
}


void test57(const tTest& t)
{
    fml input[] = {     FML(0.928363),   FML(0.585579),   FML(0.410171),   FML(0.080055),
   FML(0.654562),   FML(0.363439),   FML(0.720108),   FML(0.540296),
   FML(0.576590),   FML(0.846849),   FML(0.437083),   FML(0.945944),
   FML(0.828541),   FML(0.447882),   FML(0.085496),   FML(0.036883),
   FML(0.657816),   FML(0.734745),   FML(0.068040),   FML(0.630831),
   FML(0.383381),   FML(0.857243),   FML(0.707280),   FML(0.482543),
   FML(0.804502),   FML(0.115124),   FML(0.935517),   FML(0.356978),
   FML(0.581479),   FML(0.752936),   FML(0.674133),   FML(0.982295),
  };
    u32 inputRows = 4;
    u32 inputCols = 8;

    fml correctOutput_k33[] = {     FML(1.16693),   FML(2.09629),   FML(1.78474),   FML(1.89952),   FML(1.88135),   FML(1.53861),   FML(1.73306),   FML(0.86564),
   FML(2.06756),   FML(3.31686),   FML(2.54916),   FML(2.91281),   FML(3.21039),   FML(3.00444),   FML(3.16469),   FML(1.58202),
   FML(1.82926),   FML(3.00068),   FML(1.99000),   FML(2.74724),   FML(2.64230),   FML(2.64771),   FML(3.53534),   FML(1.43946),
   FML(1.41091),   FML(2.14730),   FML(1.68741),   FML(1.87911),   FML(1.34711),   FML(2.03927),   FML(2.95345),   FML(1.80483),
  };

    fml correctOutput_k55[] = {     FML(4.0853),   FML(4.2807),   FML(4.9422),   FML(5.0490),   FML(5.6659),   FML(5.4342),   FML(4.5425),   FML(3.9002),
   FML(5.0131),   FML(5.4522),   FML(6.5197),   FML(6.8793),   FML(6.3255),   FML(6.2246),   FML(5.3675),   FML(4.6105),
   FML(4.4940),   FML(4.4391),   FML(5.2945),   FML(5.7609),   FML(6.0018),   FML(6.3328),   FML(5.5340),   FML(3.7898),
   FML(3.6190),   FML(3.8754),   FML(4.1220),   FML(4.1391),   FML(4.7133),   FML(4.5816),   FML(4.2309),   FML(3.2880),
  };

    fml correctOutput_k57[] = {     FML(3.6947),   FML(5.0856),   FML(5.4754),   FML(6.7467),   FML(6.3668),   FML(5.0555),   FML(4.5423),   FML(4.3438),
   FML(3.8251),   FML(5.7906),   FML(6.8769),   FML(7.1902),   FML(7.8669),   FML(6.5817),   FML(5.1693),   FML(5.2297),
   FML(4.2532),   FML(4.8045),   FML(5.9642),   FML(6.9305),   FML(6.3908),   FML(6.5252),   FML(5.4524),   FML(4.7584),
   FML(3.4842),   FML(4.0028),   FML(5.2784),   FML(5.5818),   FML(5.6193),   FML(5.6712),   FML(3.9683),   FML(4.2925),
  };

    TEST_ALL_KERNELS
}


void test58(const tTest& t)
{
    fml input[] = {     FML(0.609275),   FML(0.503698),   FML(0.471096),   FML(0.555857),   FML(0.614753),   FML(0.489529),   FML(0.336084),   FML(0.045855),
   FML(0.431360),   FML(0.090122),   FML(0.818061),   FML(0.896938),   FML(0.699207),   FML(0.165110),   FML(0.510241),   FML(0.727755),
   FML(0.312827),   FML(0.924340),   FML(0.492701),   FML(0.210578),   FML(0.803147),   FML(0.593715),   FML(0.507746),   FML(0.875472),
   FML(0.814748),   FML(0.937455),   FML(0.117433),   FML(0.371780),   FML(0.071891),   FML(0.708319),   FML(0.370159),   FML(0.710926),
   FML(0.656361),   FML(0.490559),   FML(0.962150),   FML(0.498117),   FML(0.701198),   FML(0.386538),   FML(0.395574),   FML(0.583655),
  };
    u32 inputRows = 8;
    u32 inputCols = 5;

    fml correctOutput_k33[] = {     FML(0.75877),   FML(1.72842),   FML(1.96368),   FML(1.55834),   FML(0.97012),
   FML(1.62713),   FML(2.76772),   FML(2.45580),   FML(3.25001),   FML(2.02047),
   FML(1.96291),   FML(2.42628),   FML(2.48494),   FML(3.41873),   FML(1.61880),
   FML(2.36167),   FML(2.73982),   FML(2.38861),   FML(2.66519),   FML(1.14280),
   FML(1.89394),   FML(2.96003),   FML(2.63134),   FML(2.43035),   FML(1.07299),
   FML(1.63102),   FML(2.96971),   FML(2.43217),   FML(2.66938),   FML(1.20974),
   FML(1.62660),   FML(2.57946),   FML(2.49982),   FML(2.66495),   FML(1.50341),
   FML(1.37429),   FML(1.96785),   FML(2.34316),   FML(2.27509),   FML(1.32489),
  };

    fml correctOutput_k55[] = {     FML(3.5053),   FML(3.9056),   FML(5.5306),   FML(4.6942),   FML(3.8072),
   FML(4.3335),   FML(5.6672),   FML(6.9165),   FML(5.5545),   FML(3.6683),
   FML(5.9346),   FML(6.0464),   FML(8.1139),   FML(6.9448),   FML(4.7846),
   FML(4.8736),   FML(6.1765),   FML(8.0542),   FML(5.9251),   FML(4.8426),
   FML(5.7753),   FML(5.8565),   FML(6.5364),   FML(5.7711),   FML(4.4095),
   FML(5.7579),   FML(6.2489),   FML(7.3180),   FML(6.5283),   FML(4.4933),
   FML(3.9050),   FML(4.8273),   FML(4.9019),   FML(4.8578),   FML(3.8775),
   FML(3.3345),   FML(4.2650),   FML(4.8832),   FML(3.5848),   FML(2.3798),
  };

    fml correctOutput_k57[] = {     FML(3.2761),   FML(5.0525),   FML(4.1996),   FML(4.9082),   FML(4.8546),
   FML(4.0045),   FML(6.1138),   FML(5.6876),   FML(5.9688),   FML(4.9337),
   FML(4.8857),   FML(7.0955),   FML(6.9306),   FML(6.4037),   FML(6.1734),
   FML(5.0231),   FML(6.0932),   FML(6.7515),   FML(5.9774),   FML(5.3353),
   FML(4.4560),   FML(6.4417),   FML(6.2258),   FML(5.9045),   FML(5.4187),
   FML(4.8169),   FML(5.9205),   FML(6.0680),   FML(6.3096),   FML(5.9696),
   FML(3.9345),   FML(4.7256),   FML(5.0239),   FML(4.5382),   FML(4.4108),
   FML(3.9223),   FML(4.4777),   FML(4.9273),   FML(4.3120),   FML(3.7263),
  };

    TEST_ALL_KERNELS
}


void test59(const tTest& t)
{
    fml input[] = {    FML(0.6064830),   FML(0.7188080),   FML(0.6328022),   FML(0.9726999),   FML(0.5749075),
   FML(0.0314088),   FML(0.5660373),   FML(0.9581217),   FML(0.2801106),   FML(0.7915846),
   FML(0.1620689),   FML(0.2912427),   FML(0.5734016),   FML(0.0304394),   FML(0.3380080),
   FML(0.6400421),   FML(0.9299222),   FML(0.7252336),   FML(0.1899619),   FML(0.9578943),
   FML(0.4429365),   FML(0.4227350),   FML(0.3257807),   FML(0.6660347),   FML(0.7690261),
   FML(0.5513454),   FML(0.3164264),   FML(0.0071053),   FML(0.5387141),   FML(0.9374935),
   FML(0.8890711),   FML(0.7010000),   FML(0.8542243),   FML(0.8173565),   FML(0.6967316),
   FML(0.7466912),   FML(0.5055567),   FML(0.5576688),   FML(0.7001146),   FML(0.7096755),
   };
    u32 inputRows = 5;
    u32 inputCols = 8;

    fml correctOutput_k33[] = {     FML(0.82037),   FML(1.11820),   FML(1.61584),   FML(1.16300),   FML(1.51278),   FML(1.89785),   FML(1.72090),   FML(0.99598),
   FML(1.84350),   FML(2.24220),   FML(2.60974),   FML(1.95864),   FML(2.43767),   FML(3.11318),   FML(2.82887),   FML(2.07007),
   FML(2.03467),   FML(2.49348),   FML(2.98846),   FML(2.48195),   FML(2.52387),   FML(2.90459),   FML(2.38537),   FML(1.87495),
   FML(2.16186),   FML(2.72041),   FML(3.13434),   FML(2.59504),   FML(2.57439),   FML(2.91855),   FML(2.68330),   FML(2.01956),
   FML(1.60037),   FML(2.05850),   FML(1.96762),   FML(1.75370),   FML(2.53420),   FML(2.83166),   FML(2.85341),   FML(1.89221),
  };

    fml correctOutput_k55[] = {     FML(3.3963),   FML(4.3162),   FML(5.2445),   FML(5.0564),   FML(5.2633),   FML(5.8977),   FML(4.2131),   FML(3.0071),
   FML(3.7700),   FML(4.9136),   FML(6.2028),   FML(5.8702),   FML(6.2027),   FML(6.7095),   FML(5.5479),   FML(4.1280),
   FML(5.4737),   FML(6.7626),   FML(8.1300),   FML(7.0986),   FML(7.4545),   FML(8.8004),   FML(7.6432),   FML(5.5067),
   FML(4.2194),   FML(4.4023),   FML(5.8152),   FML(5.8543),   FML(5.8763),   FML(6.8398),   FML(5.4736),   FML(4.0341),
   FML(4.0756),   FML(4.5132),   FML(5.0459),   FML(5.0363),   FML(4.6414),   FML(5.4129),   FML(4.3579),   FML(2.7340),
  };

    fml correctOutput_k57[] = {    FML(3.3637),   FML(4.3560),   FML(4.3014),   FML(6.4813),   FML(6.8557),   FML(5.0195),   FML(4.8836),   FML(4.2551),
   FML(4.0354),   FML(5.9012),   FML(5.9676),   FML(7.1436),   FML(7.2152),   FML(5.3926),   FML(5.1942),   FML(5.6355),
   FML(5.5113),   FML(7.7061),   FML(7.4062),   FML(8.6665),   FML(9.0019),   FML(7.5713),   FML(7.1920),   FML(7.0756),
   FML(4.0911),   FML(5.7326),   FML(5.4046),   FML(6.4354),   FML(7.3723),   FML(6.1379),   FML(5.4650),   FML(4.8851),
   FML(4.1794),   FML(5.5713),   FML(6.0214),   FML(6.3859),   FML(6.9576),   FML(5.6092),   FML(5.0728),   FML(4.1581),
  };

    TEST_ALL_KERNELS
}


void test60(const tTest& t)
{
    fml input[] = {    FML(0.657854),   FML(0.112469),   FML(0.947455),   FML(0.327469),   FML(0.568907),   FML(0.632081),   FML(0.933093),   FML(0.603901),
   FML(0.063987),   FML(0.378818),   FML(0.840801),   FML(0.923418),   FML(0.886384),   FML(0.388796),   FML(0.152764),   FML(0.210747),
   FML(0.661289),   FML(0.713039),   FML(0.890332),   FML(0.986696),   FML(0.273078),   FML(0.652027),   FML(0.886643),   FML(0.330779),
   FML(0.054089),   FML(0.924845),   FML(0.721878),   FML(0.850918),   FML(0.084790),   FML(0.181370),   FML(0.488647),   FML(0.145751),
   FML(0.918662),   FML(0.592487),   FML(0.040974),   FML(0.747436),   FML(0.247156),   FML(0.284667),   FML(0.913511),   FML(0.505031),
   FML(0.255612),   FML(0.094743),   FML(0.068141),   FML(0.111717),   FML(0.167651),   FML(0.075211),   FML(0.648262),   FML(0.469228),
   };
    u32 inputRows = 8;
    u32 inputCols = 6;

    fml correctOutput_k33[] = {     FML(0.70074),   FML(1.79133),   FML(1.20202),   FML(1.96961),   FML(0.79111),   FML(0.96076),
   FML(1.60975),   FML(2.86462),   FML(2.36909),   FML(2.69587),   FML(1.78098),   FML(1.67672),
   FML(2.05403),   FML(3.21640),   FML(3.54170),   FML(3.15027),   FML(2.21415),   FML(1.03142),
   FML(2.74373),   FML(3.52072),   FML(3.48083),   FML(2.98652),   FML(1.86874),   FML(0.87551),
   FML(2.12084),   FML(2.92863),   FML(3.04537),   FML(2.81137),   FML(1.81927),   FML(1.26833),
   FML(1.64893),   FML(3.15775),   FML(2.16538),   FML(2.22020),   FML(1.37090),   FML(0.90933),
   FML(1.22172),   FML(3.12868),   FML(1.77958),   FML(2.92147),   FML(1.91517),   FML(1.28643),
   FML(1.05074),   FML(2.51593),   FML(1.46352),   FML(2.60563),   FML(2.12730),   FML(1.82307),
  };

    fml correctOutput_k55[] = {     FML(4.0317),   FML(4.2680),   FML(5.3375),   FML(4.7792),   FML(4.1300),   FML(2.7916),
   FML(5.0585),   FML(6.4118),   FML(8.2250),   FML(6.9449),   FML(5.1590),   FML(3.2940),
   FML(5.7954),   FML(6.8254),   FML(7.1442),   FML(6.9061),   FML(5.1879),   FML(3.1179),
   FML(6.2418),   FML(7.7064),   FML(8.5515),   FML(6.8303),   FML(4.3529),   FML(2.3518),
   FML(6.6561),   FML(7.2081),   FML(8.7092),   FML(6.6857),   FML(5.5951),   FML(3.4551),
   FML(5.5110),   FML(5.8673),   FML(7.2459),   FML(5.9817),   FML(5.0029),   FML(3.3646),
   FML(4.8177),   FML(4.3883),   FML(5.1360),   FML(4.5066),   FML(3.5914),   FML(2.2965),
   FML(3.5079),   FML(3.4988),   FML(4.1042),   FML(3.8551),   FML(3.2800),   FML(2.1741),
  };

    fml correctOutput_k57[] = {     FML(3.5448),   FML(4.2067),   FML(4.5489),   FML(5.6687),   FML(4.6408),   FML(4.3862),
   FML(4.5378),   FML(5.6830),   FML(6.5823),   FML(6.7175),   FML(6.3295),   FML(4.9131),
   FML(5.5458),   FML(6.8132),   FML(7.2319),   FML(6.8963),   FML(6.1562),   FML(4.3563),
   FML(5.6617),   FML(7.3921),   FML(7.8505),   FML(7.4434),   FML(6.2074),   FML(4.6650),
   FML(5.7401),   FML(7.0645),   FML(7.5766),   FML(6.9956),   FML(5.7334),   FML(5.3288),
   FML(4.5179),   FML(6.4850),   FML(6.4138),   FML(6.4968),   FML(4.8379),   FML(4.6393),
   FML(4.1837),   FML(5.5711),   FML(5.2969),   FML(5.6196),   FML(4.0552),   FML(3.8802),
   FML(3.6698),   FML(4.0658),   FML(4.5940),   FML(4.4156),   FML(3.4251),   FML(3.3776),
  };

    TEST_ALL_KERNELS
}


void test61(const tTest& t)
{
    fml input[] = {     FML(0.301021),   FML(0.379879),   FML(0.200237),   FML(0.110808),   FML(0.177972),   FML(0.850345),
   FML(0.863766),   FML(0.583159),   FML(0.884982),   FML(0.315493),   FML(0.879943),   FML(0.264983),
   FML(0.270299),   FML(0.576190),   FML(0.361958),   FML(0.392123),   FML(0.263632),   FML(0.744610),
   FML(0.343374),   FML(0.225057),   FML(0.883672),   FML(0.844403),   FML(0.728948),   FML(0.394359),
   FML(0.493977),   FML(0.129510),   FML(0.508707),   FML(0.242284),   FML(0.901641),   FML(0.316977),
   FML(0.184493),   FML(0.864628),   FML(0.991109),   FML(0.534496),   FML(0.340234),   FML(0.085574),
   FML(0.457550),   FML(0.861555),   FML(0.501300),   FML(0.850102),   FML(0.204054),   FML(0.748924),
   FML(0.337175),   FML(0.685456),   FML(0.768317),   FML(0.752745),   FML(0.573688),   FML(0.320891),
  };
    u32 inputRows = 6;
    u32 inputCols = 8;

    fml correctOutput_k33[] = {     FML(1.44716),   FML(1.25977),   FML(1.36799),   FML(1.04707),   FML(1.39322),   FML(1.66319),   FML(1.34943),   FML(0.71789),
   FML(2.20055),   FML(2.20752),   FML(2.85266),   FML(1.97323),   FML(2.67132),   FML(2.43721),   FML(2.76121),   FML(1.54429),
   FML(1.82185),   FML(2.12055),   FML(3.22533),   FML(1.96835),   FML(3.03005),   FML(2.88316),   FML(3.88531),   FML(1.91400),
   FML(1.93790),   FML(1.77760),   FML(3.46145),   FML(2.63270),   FML(3.36389),   FML(2.69892),   FML(3.46572),   FML(1.74762),
   FML(1.47458),   FML(1.88173),   FML(2.82925),   FML(2.48031),   FML(2.48286),   FML(2.72062),   FML(2.65257),   FML(1.72148),
   FML(1.26196),   FML(2.17907),   FML(2.29018),   FML(2.24472),   FML(1.98309),   FML(2.23943),   FML(1.47578),   FML(1.25173),
  };

    fml correctOutput_k55[] = {    FML(3.3570),   FML(4.1727),   FML(4.6901),   FML(5.3491),   FML(4.8991),   FML(5.2210),   FML(4.2328),   FML(4.0356),
   FML(3.6268),   FML(4.7987),   FML(4.9358),   FML(6.0279),   FML(6.5299),   FML(7.6929),   FML(5.5038),   FML(4.6035),
   FML(4.9327),   FML(6.0655),   FML(6.7261),   FML(8.3341),   FML(7.1057),   FML(8.1670),   FML(6.4169),   FML(4.6988),
   FML(4.5741),   FML(6.5036),   FML(6.9701),   FML(7.0945),   FML(7.8733),   FML(8.5569),   FML(6.7488),   FML(4.5845),
   FML(3.8092),   FML(4.8294),   FML(6.0464),   FML(5.8649),   FML(6.0212),   FML(5.6028),   FML(4.6537),   FML(3.6099),
   FML(3.2173),   FML(3.7657),   FML(4.5438),   FML(3.8105),   FML(3.9478),   FML(4.5820),   FML(3.7383),   FML(2.3562),
   };

    fml correctOutput_k57[] = {     FML(3.2398),   FML(4.1826),   FML(5.1702),   FML(5.3566),   FML(5.8146),   FML(4.6095),   FML(4.6081),   FML(4.4984),
   FML(3.6624),   FML(4.1574),   FML(5.7284),   FML(6.4144),   FML(7.3345),   FML(6.1569),   FML(6.2525),   FML(5.0375),
   FML(4.7043),   FML(6.1496),   FML(7.0830),   FML(8.2140),   FML(9.7298),   FML(7.1444),   FML(7.0584),   FML(6.0152),
   FML(4.3569),   FML(5.5536),   FML(6.9913),   FML(8.7546),   FML(8.9313),   FML(7.8587),   FML(6.6819),   FML(5.8541),
   FML(3.7499),   FML(5.8310),   FML(5.6566),   FML(6.7176),   FML(7.1336),   FML(6.1872),   FML(4.5107),   FML(4.6502),
   FML(3.5265),   FML(4.7095),   FML(4.4577),   FML(5.4511),   FML(5.0377),   FML(4.9815),   FML(3.9361),   FML(3.6438),
  };

    TEST_ALL_KERNELS
}


void test62(const tTest& t)
{
    fml input[] = {      FML(0.521390),   FML(0.786576),   FML(0.482176),   FML(0.961636),   FML(0.457765),   FML(0.292554),   FML(0.492953),   FML(0.262133),
   FML(0.366874),   FML(0.918921),   FML(0.055818),   FML(0.039095),   FML(0.084701),   FML(0.032013),   FML(0.901283),   FML(0.727088),
   FML(0.843432),   FML(0.360780),   FML(0.683644),   FML(0.368012),   FML(0.035571),   FML(0.942200),   FML(0.808276),   FML(0.518538),
   FML(0.450525),   FML(0.403884),   FML(0.548406),   FML(0.734034),   FML(0.144521),   FML(0.522359),   FML(0.780146),   FML(0.601679),
   FML(0.274701),   FML(0.190164),   FML(0.971954),   FML(0.797198),   FML(0.509378),   FML(0.244559),   FML(0.714098),   FML(0.845024),
   FML(0.485820),   FML(0.058568),   FML(0.945476),   FML(0.597153),   FML(0.366144),   FML(0.872558),   FML(0.421560),   FML(0.411688),
   FML(0.087595),   FML(0.437030),   FML(0.614594),   FML(0.125067),   FML(0.879345),   FML(0.665906),   FML(0.971597),   FML(0.807636),
 };
    u32 inputRows = 8;
    u32 inputCols = 7;

    fml correctOutput_k33[] = {     FML(1.34465),   FML(1.67940),   FML(1.36427),   FML(1.26322),   FML(1.07165),   FML(0.94097),   FML(0.62233),
   FML(1.61247),   FML(2.96755),   FML(2.69274),   FML(2.73855),   FML(2.30367),   FML(1.90219),   FML(0.99214),
   FML(1.41636),   FML(2.94791),   FML(2.70820),   FML(2.84044),   FML(2.50990),   FML(2.15070),   FML(1.26326),
   FML(0.74089),   FML(2.21860),   FML(1.78946),   FML(3.17191),   FML(3.20101),   FML(3.46315),   FML(1.91555),
   FML(0.90439),   FML(2.57026),   FML(1.59631),   FML(2.26685),   FML(2.92827),   FML(2.99314),   FML(1.33132),
   FML(1.32299),   FML(2.39967),   FML(1.70879),   FML(2.21464),   FML(2.36113),   FML(3.12897),   FML(1.70467),
   FML(1.75822),   FML(2.67638),   FML(2.89871),   FML(3.43595),   FML(2.76756),   FML(3.34101),   FML(1.83245),
   FML(1.73101),   FML(2.36232),   FML(3.02288),   FML(2.94116),   FML(2.42058),   FML(3.02013),   FML(1.47262),
  };

    fml correctOutput_k55[] = {     FML(3.6739),   FML(4.3852),   FML(5.4848),   FML(5.3421),   FML(4.8549),   FML(3.7114),   FML(2.9285),
   FML(4.5892),   FML(5.1213),   FML(6.1482),   FML(5.1865),   FML(5.5638),   FML(4.6903),   FML(3.9813),
   FML(4.3751),   FML(5.3686),   FML(7.2410),   FML(6.2345),   FML(7.2575),   FML(5.6964),   FML(4.0510),
   FML(5.1135),   FML(5.4350),   FML(6.8305),   FML(7.0791),   FML(7.2338),   FML(5.7194),   FML(4.4305),
   FML(4.2476),   FML(5.3506),   FML(7.3619),   FML(7.7905),   FML(9.4508),   FML(7.4571),   FML(5.2546),
   FML(4.4336),   FML(5.3990),   FML(7.6067),   FML(8.1198),   FML(8.6562),   FML(6.8803),   FML(5.3229),
   FML(3.9938),   FML(4.5129),   FML(5.1848),   FML(5.7761),   FML(6.1653),   FML(5.4510),   FML(4.4045),
   FML(3.3659),   FML(4.7815),   FML(5.4724),   FML(4.7361),   FML(5.7923),   FML(4.4649),   FML(2.9455),
  };

    fml correctOutput_k57[] = {    FML(3.5301),   FML(4.7166),   FML(5.3252),   FML(5.4948),   FML(4.3275),   FML(4.3039),   FML(3.8871),
   FML(4.8685),   FML(5.7766),   FML(5.6088),   FML(6.1583),   FML(4.8330),   FML(4.9648),   FML(4.6047),
   FML(4.3285),   FML(6.3400),   FML(6.6500),   FML(8.0412),   FML(6.4954),   FML(6.0088),   FML(5.2274),
   FML(4.9350),   FML(6.0869),   FML(6.6981),   FML(8.0717),   FML(6.5864),   FML(6.2713),   FML(5.4291),
   FML(4.1849),   FML(5.6899),   FML(6.6134),   FML(8.8806),   FML(7.6777),   FML(7.2356),   FML(6.8076),
   FML(4.4330),   FML(5.7778),   FML(7.0747),   FML(8.8281),   FML(7.6536),   FML(6.6935),   FML(6.4524),
   FML(4.4478),   FML(5.2289),   FML(5.8410),   FML(7.0697),   FML(6.4884),   FML(5.3023),   FML(4.8114),
   FML(4.0570),   FML(4.8792),   FML(5.9792),   FML(6.3430),   FML(6.2958),   FML(5.0818),   FML(4.1080),
   };

    TEST_ALL_KERNELS
}


void test63(const tTest& t)
{
    fml input[] = {     FML(0.190384),   FML(0.400250),   FML(0.062168),   FML(0.216091),   FML(0.603179),   FML(0.958365),   FML(0.217429),
   FML(0.844085),   FML(0.953977),   FML(0.848255),   FML(0.311871),   FML(0.245676),   FML(0.669278),   FML(0.428444),
   FML(0.893952),   FML(0.864584),   FML(0.208111),   FML(0.688564),   FML(0.586169),   FML(0.245370),   FML(0.283883),
   FML(0.121163),   FML(0.883649),   FML(0.145455),   FML(0.538285),   FML(0.571484),   FML(0.280109),   FML(0.482320),
   FML(0.018528),   FML(0.951907),   FML(0.404899),   FML(0.989789),   FML(0.956207),   FML(0.172788),   FML(0.189112),
   FML(0.108973),   FML(0.520656),   FML(0.666373),   FML(0.481236),   FML(0.661184),   FML(0.407792),   FML(0.036093),
   FML(0.161340),   FML(0.474444),   FML(0.439547),   FML(0.647631),   FML(0.564297),   FML(0.398197),   FML(0.044196),
   FML(0.427250),   FML(0.944328),   FML(0.862650),   FML(0.544189),   FML(0.405268),   FML(0.829584),   FML(0.757045),
  };
    u32 inputRows = 7;
    u32 inputCols = 8;

    fml correctOutput_k33[] = {    FML(1.71007),   FML(1.89699),   FML(1.78699),   FML(1.72117),   FML(0.97625),   FML(0.92960),   FML(1.50717),   FML(0.49997),
   FML(2.39331),   FML(2.64876),   FML(3.15040),   FML(2.89700),   FML(2.05807),   FML(1.86693),   FML(2.60676),   FML(1.07917),
   FML(2.04431),   FML(2.54355),   FML(3.32591),   FML(3.48768),   FML(3.01713),   FML(3.03434),   FML(3.25590),   FML(1.60953),
   FML(1.36721),   FML(2.10976),   FML(2.44312),   FML(2.88580),   FML(2.44977),   FML(2.99526),   FML(2.89279),   FML(1.63146),
   FML(1.37952),   FML(2.27294),   FML(2.22094),   FML(3.19279),   FML(2.88838),   FML(3.31918),   FML(2.88946),   FML(1.60445),
   FML(1.67484),   FML(2.47411),   FML(2.27273),   FML(2.27133),   FML(2.32862),   FML(2.40914),   FML(2.98336),   FML(1.35092),
   FML(1.55356),   FML(2.08155),   FML(1.90579),   FML(1.14575),   FML(1.27215),   FML(1.08547),   FML(2.05473),   FML(1.15469),
   };

    fml correctOutput_k55[] = {      FML(3.7172),   FML(3.9445),   FML(4.1813),   FML(4.7769),   FML(4.1509),   FML(4.7905),   FML(4.2365),   FML(3.4504),
   FML(4.7679),   FML(5.7819),   FML(6.7627),   FML(6.7995),   FML(5.9670),   FML(6.0668),   FML(5.1853),   FML(4.0610),
   FML(5.1086),   FML(6.5120),   FML(7.5408),   FML(7.5790),   FML(8.0792),   FML(8.0152),   FML(6.8898),   FML(4.6950),
   FML(5.2138),   FML(6.1522),   FML(8.0461),   FML(7.9362),   FML(7.5992),   FML(7.9556),   FML(7.1413),   FML(5.0176),
   FML(4.7496),   FML(5.4770),   FML(6.8532),   FML(6.8342),   FML(6.4379),   FML(6.6449),   FML(5.5428),   FML(4.1434),
   FML(3.6431),   FML(4.1519),   FML(4.6428),   FML(5.2673),   FML(5.2107),   FML(5.5483),   FML(4.5576),   FML(3.0930),
   FML(3.0418),   FML(3.6151),   FML(3.7649),   FML(3.5948),   FML(3.2809),   FML(3.9290),   FML(3.3880),   FML(2.5794),
 };

    fml correctOutput_k57[] = {     FML(2.9653),   FML(3.9618),   FML(4.8229),   FML(4.7178),   FML(5.2238),   FML(4.0052),   FML(3.4196),   FML(3.7114),
   FML(4.6433),   FML(5.8216),   FML(7.0347),   FML(7.0866),   FML(7.8568),   FML(6.6474),   FML(5.4577),   FML(5.1732),
   FML(4.4900),   FML(6.0440),   FML(7.4818),   FML(7.6058),   FML(8.5080),   FML(8.2968),   FML(6.4602),   FML(6.1685),
   FML(4.4817),   FML(6.6403),   FML(7.8609),   FML(9.2935),   FML(9.0764),   FML(8.1815),   FML(6.1363),   FML(6.2922),
   FML(4.1233),   FML(6.0553),   FML(6.6214),   FML(7.4737),   FML(8.3383),   FML(7.0318),   FML(5.4402),   FML(5.1789),
   FML(4.0266),   FML(4.9958),   FML(5.3925),   FML(5.4703),   FML(6.4330),   FML(5.4088),   FML(4.7479),   FML(4.1295),
   FML(3.3044),   FML(4.0302),   FML(4.4147),   FML(4.3774),   FML(4.6871),   FML(4.2020),   FML(3.3723),   FML(3.2558),
  };

    TEST_ALL_KERNELS
}


void test64(const tTest& t)
{
    fml input[] = {     FML(0.4975843),   FML(0.7649783),   FML(0.9745824),   FML(0.1817283),   FML(0.7244010),   FML(0.3544084),   FML(0.8898476),   FML(0.9130203),
   FML(0.6418944),   FML(0.8815800),   FML(0.0557325),   FML(0.7716145),   FML(0.8292254),   FML(0.5422688),   FML(0.1324824),   FML(0.2391738),
   FML(0.4489478),   FML(0.2390138),   FML(0.8027390),   FML(0.5675383),   FML(0.3256793),   FML(0.5341854),   FML(0.5735624),   FML(0.1186487),
   FML(0.5879353),   FML(0.6426611),   FML(0.4510997),   FML(0.8991498),   FML(0.5320306),   FML(0.1743075),   FML(0.8502557),   FML(0.6880272),
   FML(0.3043464),   FML(0.2280800),   FML(0.1875134),   FML(0.8178375),   FML(0.8089752),   FML(0.0378812),   FML(0.1333288),   FML(0.5958778),
   FML(0.4037062),   FML(0.8725369),   FML(0.3607848),   FML(0.5923391),   FML(0.1022382),   FML(0.2615207),   FML(0.0086351),   FML(0.8925673),
   FML(0.9417999),   FML(0.5490311),   FML(0.9208133),   FML(0.5822324),   FML(0.7308032),   FML(0.0334275),   FML(0.0897206),   FML(0.9896298),
   FML(0.5720244),   FML(0.0226360),   FML(0.4729929),   FML(0.2238387),   FML(0.7430914),   FML(0.8890859),   FML(0.0859784),   FML(0.4774529),
  };
    u32 inputRows = 8;
    u32 inputCols = 8;

    fml correctOutput_k33[] = {     FML(1.52513),   FML(1.28625),   FML(1.78784),   FML(1.07078),   FML(1.75199),   FML(1.70550),   FML(1.18934),   FML(0.99425),
   FML(1.76281),   FML(2.83984),   FML(2.94353),   FML(1.82621),   FML(2.69927),   FML(2.79402),   FML(2.52778),   FML(1.91556),
   FML(1.98003),   FML(3.29117),   FML(2.92733),   FML(2.36334),   FML(2.75754),   FML(2.70111),   FML(2.30829),   FML(1.51196),
   FML(2.04695),   FML(2.75895),   FML(2.75757),   FML(3.12160),   FML(2.28790),   FML(2.93793),   FML(2.57191),   FML(1.86611),
   FML(2.03456),   FML(2.44246),   FML(3.06455),   FML(2.77753),   FML(2.63756),   FML(2.85727),   FML(2.70081),   FML(1.48204),
   FML(1.76174),   FML(2.81897),   FML(2.79666),   FML(1.96255),   FML(1.67248),   FML(1.77922),   FML(2.16526),   FML(1.44603),
   FML(1.20395),   FML(2.45709),   FML(2.52838),   FML(1.92228),   FML(1.98236),   FML(1.47679),   FML(1.73659),   FML(0.92680),
   FML(1.06286),   FML(2.28033),   FML(1.96540),   FML(1.92096),   FML(2.33982),   FML(1.69625),   FML(1.40899),   FML(1.07844),
  };

    fml correctOutput_k55[] = {     FML(3.8866),   FML(4.6255),   FML(5.0764),   FML(4.3812),   FML(5.1945),   FML(5.3721),   FML(4.0374),   FML(3.3367),
   FML(4.8197),   FML(5.8244),   FML(6.5935),   FML(6.6963),   FML(7.1436),   FML(6.2358),   FML(4.6756),   FML(3.5824),
   FML(5.7249),   FML(6.5511),   FML(7.6565),   FML(7.5480),   FML(7.4541),   FML(8.4480),   FML(6.9408),   FML(4.1923),
   FML(6.0297),   FML(6.5445),   FML(7.5249),   FML(7.0635),   FML(6.8541),   FML(6.7040),   FML(5.0316),   FML(3.7572),
   FML(5.1600),   FML(6.7538),   FML(8.0298),   FML(6.1386),   FML(6.3963),   FML(6.3299),   FML(4.5326),   FML(3.2766),
   FML(5.1109),   FML(6.1025),   FML(7.6400),   FML(6.8334),   FML(7.1057),   FML(7.9705),   FML(5.6374),   FML(4.2121),
   FML(4.4879),   FML(4.6127),   FML(5.0774),   FML(4.5442),   FML(3.3089),   FML(4.3909),   FML(4.3374),   FML(3.3841),
   FML(3.2995),   FML(3.8972),   FML(3.8032),   FML(3.8055),   FML(4.1139),   FML(3.4736),   FML(3.0321),   FML(1.9580),
  };

    fml correctOutput_k57[] = {     FML(3.7523),   FML(4.3994),   FML(5.3208),   FML(6.1870),   FML(5.7258),   FML(4.7041),   FML(4.7536),   FML(4.1360),
   FML(4.5260),   FML(6.1541),   FML(6.0278),   FML(7.5903),   FML(7.5133),   FML(5.9134),   FML(5.7600),   FML(4.8085),
   FML(5.1087),   FML(7.0149),   FML(6.7485),   FML(9.5903),   FML(8.7754),   FML(6.8796),   FML(6.8153),   FML(6.0463),
   FML(5.1957),   FML(6.6758),   FML(7.5892),   FML(8.4580),   FML(8.7918),   FML(6.7028),   FML(4.8855),   FML(4.8717),
   FML(5.6983),   FML(6.4544),   FML(7.1966),   FML(8.0799),   FML(7.9867),   FML(6.7876),   FML(5.4587),   FML(4.4468),
   FML(4.7713),   FML(6.5240),   FML(7.0378),   FML(7.7880),   FML(7.6275),   FML(6.6042),   FML(6.1113),   FML(5.3427),
   FML(4.4043),   FML(5.5620),   FML(4.6306),   FML(5.5969),   FML(5.0800),   FML(4.1636),   FML(3.7588),   FML(3.5411),
   FML(3.7480),   FML(4.6955),   FML(4.2434),   FML(5.1855),   FML(5.3728),   FML(4.6933),   FML(3.7966),   FML(3.2706),
  };

    TEST_ALL_KERNELS
}


int main()
{
    tCrashReporter::init();

    srand(time(0));

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

    return 0;
}
