#include "conv2d_gpu_common.ipp"

#include "../cuda_stuff.ipp"

#define GPU_ACCUM_USE_TEMPLATE 0
#include "conv2d_gpu_accum_impl.ipp"
#undef  GPU_ACCUM_USE_TEMPLATE

#define GPU_ACCUM_USE_TEMPLATE 1
#include "conv2d_gpu_accum_impl.ipp"
#undef  GPU_ACCUM_USE_TEMPLATE


#if THROW_IF_FALLBACK_IMPL_NEEDED
#define RUN_FALLBACK_IMPL \
    throw eRuntimeError("The fallback (aka, slow) implementation is needed to accum this input. But we've turned off the fallback implementation, so you'll need to turn it on or (preferably) modify your code to use one of the fast implementation paths.");
#else
#define RUN_FALLBACK_IMPL \
    if (canUseFastImpl) \
    { \
        accum_in_one_pass<<<gridSize, blockSize, sharedMemNeeded>>>( \
            inputComponents, kernelRows, kernelCols, numKernels, \
            inputPtr,  inputRows,   inputCols, \
            dk_ptr, \
            db_ptr, scaleFactor, \
            dA_ptr); \
    } \
    else \
    { \
        accum_in_multiple_passes<<<gridSize, blockSize, sharedMemNeeded>>>( \
            inputComponents, kernelRows, kernelCols, numKernels, \
            inputPtr,  inputRows,   inputCols, \
            dk_ptr, \
            db_ptr, scaleFactor, \
            dA_ptr); \
    }
#endif


#define SWITCH_KERNEL_DIMS(inputComponents, numKernels) \
    switch ((kernelRows * 0x10) + kernelCols) \
    { \
        case 0x33: \
        { \
            if (canUseFastImpl) \
            { \
                accum_in_one_pass_templated<inputComponents, 3, 3, numKernels><<<gridSize, blockSize, sharedMemNeeded>>>( \
                    inputPtr,  inputRows,   inputCols, \
                    dk_ptr, \
                    db_ptr, scaleFactor, \
                    dA_ptr); \
            } \
            else \
            { \
                accum_in_multiple_passes_templated<inputComponents, 3, 3, numKernels><<<gridSize, blockSize, sharedMemNeeded>>>( \
                    inputPtr,  inputRows,   inputCols, \
                    dk_ptr, \
                    db_ptr, scaleFactor, \
                    dA_ptr); \
            } \
            break; \
        } \
        case 0x55: \
        { \
            if (canUseFastImpl) \
            { \
                accum_in_one_pass_templated<inputComponents, 5, 5, numKernels><<<gridSize, blockSize, sharedMemNeeded>>>( \
                    inputPtr,  inputRows,   inputCols, \
                    dk_ptr, \
                    db_ptr, scaleFactor, \
                    dA_ptr); \
            } \
            else \
            { \
                accum_in_multiple_passes_templated<inputComponents, 5, 5, numKernels><<<gridSize, blockSize, sharedMemNeeded>>>( \
                    inputPtr,  inputRows,   inputCols, \
                    dk_ptr, \
                    db_ptr, scaleFactor, \
                    dA_ptr); \
            } \
            break; \
        } \
        default: \
            RUN_FALLBACK_IMPL \
    } \


#define SWITCH_NUM_KERNELS(inputComponents) \
    switch (numKernels) \
    { \
        case 4: SWITCH_KERNEL_DIMS(inputComponents, 4)  break; \
        case 8: SWITCH_KERNEL_DIMS(inputComponents, 8)  break; \
        case 16: SWITCH_KERNEL_DIMS(inputComponents, 16)  break; \
        case 32: SWITCH_KERNEL_DIMS(inputComponents, 32)  break; \
        case 64: SWITCH_KERNEL_DIMS(inputComponents, 64)  break; \
        default: \
            RUN_FALLBACK_IMPL \
    } \


#if COMPILE_A_BUNCH_OF_TEMPLATES_TO_MAKE_FAST_CODE
#define RUN_CONV2D_ACCUM_GPU_FUNTION \
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
#define RUN_CONV2D_ACCUM_GPU_FUNTION \
    RUN_FALLBACK_IMPL
#endif

