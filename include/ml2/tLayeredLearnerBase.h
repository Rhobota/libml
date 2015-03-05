#ifndef __ml2_tLayeredLearnerBase_h__
#define __ml2_tLayeredLearnerBase_h__


#include <ml2/iLearner.h>
#include <ml2/iLayer.h>


namespace ml2
{


class tLayeredLearnerBase : public iLearner, public bNonCopyable
{
    public:

        tLayeredLearnerBase(u32 numInputDims, u32 numOutputDims);

        tLayeredLearnerBase(iReadable* in);

        ~tLayeredLearnerBase();

        void addLayer(iLayer* layer);  // <-- takes ownership of the layer


        ///////////////////////////////////////////////////////////////////////
        // iLearner interface:   (partial interface only)
        ///////////////////////////////////////////////////////////////////////

        void addExample(const tIO& input, const tIO& target);

        fml calculateError(const tIO& output, const tIO& target);

        fml calculateError(const std::vector<tIO>& outputs,
                           const std::vector<tIO>& targets);

        void reset();

        void printLearnerInfo(std::ostream& out) const;

        std::string learnerInfoString() const;


        ///////////////////////////////////////////////////////////////////////
        // The iPackable interface:
        ///////////////////////////////////////////////////////////////////////

        void pack(iWritable* out) const;
        void unpack(iReadable* in);


    protected:

        void m_growInputMatrix(u32 newSize);
        void m_growTargetMatrix(u32 newSize);

        void m_copyToInputMatrix(const tIO& input);
        void m_copyToTargetMatrix(const tIO& target);

        void m_clearMatrices();

        void m_pushInputForward(const fml* input, u32 numInputDims, u32 inputCount,
                                const fml*& output, u32 expectedOutputDims, u32& expectedOutputCount);

        void m_calculate_output_da(const fml* output, fml* target, u32 dims, u32 count);

        void m_backpropagate(const fml* output_da, u32 numOutputDims, u32 outputCount,
                             const fml* input, u32 numInputDims, u32 inputCount);

        void m_putOutput(tIO& output, const fml* outputPtr, u32 numOutputDims, u32& outputCount);

        void m_putOutput(std::vector<tIO>::iterator outputStart, const fml* outputPtr, u32 numOutputDims, u32& outputCount);

        void m_update(fml* inputMatrix, u32 inputMatrixUsed, u32 inputMatrixNumDims,
                      fml* targetMatrix, u32 targetMatrixUsed, u32 targetMatrixNumDims);

        void m_evaluate(fml* inputMatrix, u32 inputMatrixUsed, u32 inputMatrixNumDims,
                        const fml*& output, u32 expectedOutputDims, u32& expectedOutputCount);


    protected:

        std::vector<iLayer*> m_layers;

        u32 m_numInputDims;
        u32 m_numOutputDims;

        fml* m_inputMatrix;
        u32  m_inputMatrixSize;
        u32  m_inputMatrixUsed;

        fml* m_targetMatrix;
        u32  m_targetMatrixSize;
        u32  m_targetMatrixUsed;
};


}   // namespace ml2


#endif   // __ml2_tLayeredLearnerBase_h__