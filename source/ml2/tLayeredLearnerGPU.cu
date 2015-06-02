#include <ml2/tLayeredLearnerGPU.h>

#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include <cassert>
#include <iostream>


namespace ml2
{


#define ENABLE_DEVICE_FUNCTIONS
#include "common.ipp"


tLayeredLearnerGPU::tLayeredLearnerGPU(u32 numInputDims, u32 numOutputDims)
    : tLayeredLearnerBase(numInputDims, numOutputDims),
      m_gpu_buf_1(NULL),
      m_gpu_buf_1_size(0),
      m_gpu_buf_2(NULL),
      m_gpu_buf_2_size(0),
      m_local_buf_1(NULL),
      m_local_buf_1_size(0)
{
}

tLayeredLearnerGPU::tLayeredLearnerGPU(iReadable* in)
    : tLayeredLearnerBase(in),
      m_gpu_buf_1(NULL),
      m_gpu_buf_1_size(0),
      m_gpu_buf_2(NULL),
      m_gpu_buf_2_size(0),
      m_local_buf_1(NULL),
      m_local_buf_1_size(0)
{
}

tLayeredLearnerGPU::~tLayeredLearnerGPU()
{
    // The super d'tors are called automatically.

    if (m_gpu_buf_1)
    {
        cuda_assert( cudaFree(m_gpu_buf_1) );
        m_gpu_buf_1 = NULL;
        m_gpu_buf_1_size = 0;
    }

    if (m_gpu_buf_2)
    {
        cuda_assert( cudaFree(m_gpu_buf_2) );
        m_gpu_buf_2 = NULL;
        m_gpu_buf_2_size = 0;
    }

    delete [] m_local_buf_1;
    m_local_buf_1 = NULL;
    m_local_buf_1_size = 0;
}

void tLayeredLearnerGPU::update()
{
    m_copyTo_gpu_buf_1(m_inputMatrix, m_inputMatrixUsed);
    m_copyTo_gpu_buf_2(m_targetMatrix, m_targetMatrixUsed);

    m_update(m_gpu_buf_1, m_inputMatrixUsed, m_numInputDims,
             m_gpu_buf_2, m_targetMatrixUsed, m_numOutputDims);

    m_clearMatrices();
}

void tLayeredLearnerGPU::evaluate(const tIO& input, tIO& output)
{
    m_clearMatrices();
    m_copyToInputMatrix(input);

    m_copyTo_gpu_buf_1(m_inputMatrix, m_inputMatrixUsed);

    const fml* outputPtr = NULL;
    u32 expectedOutputDims = m_numOutputDims;
    u32 expectedOutputCount = 1;

    m_evaluate(m_gpu_buf_1, m_inputMatrixUsed, m_numInputDims,
               outputPtr, expectedOutputDims, expectedOutputCount);

    m_copyTo_local_buf_1(outputPtr, expectedOutputDims*expectedOutputCount);

    m_putOutput(output, m_local_buf_1, expectedOutputDims, expectedOutputCount);

    m_clearMatrices();
}

void tLayeredLearnerGPU::evaluateBatch(const std::vector<tIO>& inputs,
                                             std::vector<tIO>& outputs)
{
    outputs.resize(inputs.size());
    evaluateBatch(inputs.begin(), inputs.end(), outputs.begin());
}

void tLayeredLearnerGPU::evaluateBatch(std::vector<tIO>::const_iterator inputStart,
                                       std::vector<tIO>::const_iterator inputEnd,
                                       std::vector<tIO>::iterator outputStart)
{
    m_clearMatrices();
    std::vector<tIO>::const_iterator sitr;
    for (sitr = inputStart; sitr != inputEnd; sitr++)
        m_copyToInputMatrix(*sitr);

    m_copyTo_gpu_buf_1(m_inputMatrix, m_inputMatrixUsed);

    const fml* outputPtr = NULL;
    u32 expectedOutputDims = m_numOutputDims;
    u32 expectedOutputCount = (u32) (inputEnd - inputStart);

    m_evaluate(m_gpu_buf_1, m_inputMatrixUsed, m_numInputDims,
               outputPtr, expectedOutputDims, expectedOutputCount);

    m_copyTo_local_buf_1(outputPtr, expectedOutputDims*expectedOutputCount);

    m_putOutput(outputStart, m_local_buf_1, expectedOutputDims, expectedOutputCount);

    m_clearMatrices();
}

static
iLearner* s_newLearnerFunc(iReadable* in)
{
    return new tLayeredLearnerGPU(in);
}

static u32 learnerId = 87986;
static bool didRegister = iLearner::registerLearnerFuncWithHeaderId(s_newLearnerFunc, learnerId);

u32 tLayeredLearnerGPU::headerId() const
{
    if (!didRegister)
        throw eRuntimeError("Registering my learner id didn't work!");
    return learnerId;
}

class tCalculateOutput_da
{
    public:

        __device__
        fml operator()(const fml& ai, const fml& yi)
        {
            return ai - yi;
        }
};

void tLayeredLearnerGPU::m_calculate_output_da(const fml* output, fml* target, u32 dims, u32 count)
{
    thrust::device_ptr<fml>       y = thrust::device_pointer_cast(target);
    thrust::device_ptr<const fml> a = thrust::device_pointer_cast(output);

    // We want: y = a - y
    thrust::transform(a, a+dims*count, y, y, tCalculateOutput_da());
}

void tLayeredLearnerGPU::m_copyTo_gpu_buf_1(const fml* local_buf, u32 size)
{
    if (m_gpu_buf_1 == NULL || m_gpu_buf_1_size < size)
    {
        if (m_gpu_buf_1)
        {
            cuda_assert( cudaFree(m_gpu_buf_1) );
            m_gpu_buf_1 = NULL;
            m_gpu_buf_1_size = 0;
        }
        cuda_assert( cudaMalloc((void**)(&m_gpu_buf_1), size*sizeof(fml)) );
        m_gpu_buf_1_size = size;
    }
    cuda_assert( cudaMemcpy(m_gpu_buf_1, local_buf, size*sizeof(fml), cudaMemcpyHostToDevice) );
}

void tLayeredLearnerGPU::m_copyTo_gpu_buf_2(const fml* local_buf, u32 size)
{
    if (m_gpu_buf_2 == NULL || m_gpu_buf_2_size < size)
    {
        if (m_gpu_buf_2)
        {
            cuda_assert( cudaFree(m_gpu_buf_2) );
            m_gpu_buf_2 = NULL;
            m_gpu_buf_2_size = 0;
        }
        cuda_assert( cudaMalloc((void**)(&m_gpu_buf_2), size*sizeof(fml)) );
        m_gpu_buf_2_size = size;
    }
    cuda_assert( cudaMemcpy(m_gpu_buf_2, local_buf, size*sizeof(fml), cudaMemcpyHostToDevice) );
}

void tLayeredLearnerGPU::m_copyTo_local_buf_1(const fml* gpu_buf, u32 size)
{
    if (m_local_buf_1 == NULL || m_local_buf_1_size < size)
    {
        delete [] m_local_buf_1;
        m_local_buf_1 = new fml[size];
        m_local_buf_1_size = size;
    }
    cuda_assert( cudaMemcpy(m_local_buf_1, gpu_buf, size*sizeof(fml), cudaMemcpyDeviceToHost) );
}


}   // namespace ml2
