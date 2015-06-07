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


int main()
{
    tCrashReporter::init();

    srand(time(0));

    tTest("convolve 2d test 1", test1);

    return 0;
}
