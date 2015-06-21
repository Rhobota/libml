/*
 * If defined, this file will take a long time to compile, but
 * will output very fast code. You should turn this on for
 * production code, but turn it off to do quick test iterations.
 */
#define COMPILE_A_BUNCH_OF_TEMPLATES_TO_MAKE_FAST_CODE 1

/*
 * If defined, this code will refuse to run the fallback
 * implementation, which is a very slow implementation.
 * This is nice to turn on if you want to be sure your
 * code uses the fast templated versions of the function
 * instead of the fallback.
 */
#define THROW_IF_FALLBACK_IMPL_NEEDED 0


#include "../cuda_stuff.ipp"

#define GPU_PART1_USE_TEMPLATE 0
#include "gpu_conv_impl.ipp"
#undef  GPU_PART1_USE_TEMPLATE

#define GPU_PART1_USE_TEMPLATE 1
#include "gpu_conv_impl.ipp"
#undef  GPU_PART1_USE_TEMPLATE


#if THROW_IF_FALLBACK_IMPL_NEEDED
#define RUN_FALLBACK_IMPL \
    throw eRuntimeError("The fallback (aka, slow) implementation is needed to convolve this input. But we've turned off the fallback implementation, so you'll need to turn it on or (preferably) modify your code to use one of the fast implementation paths.");
#else
#define RUN_FALLBACK_IMPL \
    if (canUseFastImpl) \
    { \
        gpu_conv2d_multi_input<<<gridSize, blockSize, sharedMemNeeded>>>( \
            inputComponents, kernelRows, kernelCols, kernelStepY, kernelStepX, numKernels, \
            inputPtr,  inputRows,   inputCols, \
            kernelPtr, \
            kernelBiases, scaleFactor, \
            outputPtr, outputRows, outputCols); \
    } \
    else \
    { \
        gpu_conv2d_multi_input_for_large_input<<<gridSize, blockSize, sharedMemNeeded>>>( \
            inputComponents, kernelRows, kernelCols, kernelStepY, kernelStepX, numKernels, \
            inputPtr,  inputRows,   inputCols, \
            kernelPtr, \
            kernelBiases, scaleFactor, \
            outputPtr, outputRows, outputCols); \
    }
#endif


#define SWITCH_KERNEL_DIMS(inputComponents, kernelStepY, kernelStepX, numKernels) \
    switch ((kernelRows * 0x10) + kernelCols) \
    { \
        case 0x33: \
        { \
            if (canUseFastImpl) \
            { \
                gpu_conv2d_multi_input_templated<inputComponents, 3, 3, kernelStepY, kernelStepX, numKernels><<<gridSize, blockSize, sharedMemNeeded>>>( \
                    inputPtr,  inputRows,   inputCols, \
                    kernelPtr, \
                    kernelBiases, scaleFactor, \
                    outputPtr, outputRows, outputCols); \
            } \
            else \
            { \
                gpu_conv2d_multi_input_for_large_input_templated<inputComponents, 3, 3, kernelStepY, kernelStepX, numKernels><<<gridSize, blockSize, sharedMemNeeded>>>( \
                    inputPtr,  inputRows,   inputCols, \
                    kernelPtr, \
                    kernelBiases, scaleFactor, \
                    outputPtr, outputRows, outputCols); \
            } \
            break; \
        } \
        case 0x55: \
        { \
            if (canUseFastImpl) \
            { \
                gpu_conv2d_multi_input_templated<inputComponents, 5, 5, kernelStepY, kernelStepX, numKernels><<<gridSize, blockSize, sharedMemNeeded>>>( \
                    inputPtr,  inputRows,   inputCols, \
                    kernelPtr, \
                    kernelBiases, scaleFactor, \
                    outputPtr, outputRows, outputCols); \
            } \
            else \
            { \
                gpu_conv2d_multi_input_for_large_input_templated<inputComponents, 5, 5, kernelStepY, kernelStepX, numKernels><<<gridSize, blockSize, sharedMemNeeded>>>( \
                    inputPtr,  inputRows,   inputCols, \
                    kernelPtr, \
                    kernelBiases, scaleFactor, \
                    outputPtr, outputRows, outputCols); \
            } \
            break; \
        } \
        default: \
            RUN_FALLBACK_IMPL \
    } \


#define SWITCH_KERNEL_STEP(inputComponents, numKernels) \
    switch ((kernelStepY * 0x10) + kernelStepX) \
    { \
        case 0x11: SWITCH_KERNEL_DIMS(inputComponents, 1, 1, numKernels)  break; \
        case 0x22: SWITCH_KERNEL_DIMS(inputComponents, 2, 2, numKernels)  break; \
        default: \
            RUN_FALLBACK_IMPL \
    } \


#define SWITCH_NUM_KERNELS(inputComponents) \
    switch (numKernels) \
    { \
        case 4: SWITCH_KERNEL_STEP(inputComponents, 4)  break; \
        case 8: SWITCH_KERNEL_STEP(inputComponents, 8)  break; \
        case 16: SWITCH_KERNEL_STEP(inputComponents, 16)  break; \
        case 32: SWITCH_KERNEL_STEP(inputComponents, 32)  break; \
        case 64: SWITCH_KERNEL_STEP(inputComponents, 64)  break; \
        default: \
            RUN_FALLBACK_IMPL \
    } \


#if COMPILE_A_BUNCH_OF_TEMPLATES_TO_MAKE_FAST_CODE
#define RUN_CONV2D_GPU_FUNTION \
    switch (inputComponents) \
    { \
        case 1: SWITCH_NUM_KERNELS(1)  break; \
        case 2: SWITCH_NUM_KERNELS(2)  break; \
        case 3: SWITCH_NUM_KERNELS(3)  break; \
        case 4: SWITCH_NUM_KERNELS(4)  break; \
        case 5: SWITCH_NUM_KERNELS(5)  break; \
        case 6: SWITCH_NUM_KERNELS(6)  break; \
        case 7: SWITCH_NUM_KERNELS(7)  break; \
        case 8: SWITCH_NUM_KERNELS(8)  break; \
        case 16: SWITCH_NUM_KERNELS(16)  break; \
        case 32: SWITCH_NUM_KERNELS(32)  break; \
        case 64: SWITCH_NUM_KERNELS(64)  break; \
        default: \
            RUN_FALLBACK_IMPL \
    }
#else
#define RUN_CONV2D_GPU_FUNTION \
    RUN_FALLBACK_IMPL
#endif

