#ifndef __ml2_iLayer_h__
#define __ml2_iLayer_h__


#include <ml2/common.h>


namespace ml2
{


class iLayer
{
    public:

        //////////////////////////////////////////////////////////////////////
        //
        // Methods for evaluating example(s). These are also needed when
        // training.
        //
        //////////////////////////////////////////////////////////////////////

        /**
         * Passes 'input' into this layer. Input is in the form of a matrix, stored
         * in column-major order, where each column is a single example. Therefore,
         * 'numInputDims' denotes the number of rows in the matrix, and 'count'
         * denotes the number of columns in the matrix.
         */
        virtual void takeInput(fml* input, u32 numInputDims, u32 count) = 0;

        /**
         * Obtains the output of this layer. The output will reflect the output
         * from the most recent call to takeInput(). The returned value is a
         * matrix, stored in column-major order, where each column is the output
         * from a single example. Therefore, 'numOutputDims' upon return will
         * contain the number of rows in the matrix, and 'count' upon return
         * will contain the number of columns in the matrix.
         *
         * Do not free or delete the matrix yourself. It is owned by the layer.
         * Therefore, also do not hold a reference to the matrix after this
         * layer is destroyed.
         */
        virtual fml* getOutput(u32& numOutputDims, u32& count) const = 0;


        //////////////////////////////////////////////////////////////////////
        //
        // Methods for backpropagating error. These are only needed when
        // training.
        //
        //////////////////////////////////////////////////////////////////////

        /**
         * Passes the error gradients of the most recent output to this layer
         * so that it can learn from them. The 'outputErrorGradients' matrix
         * should be the same size as the most recent output (as returned
         * by getOutput()). 'numOutputDims' and 'outputCount' are passed here to
         * prove that the caller knows what they are doing.
         *
         * Also, 'input', 'numInputDims', and 'inputCount' are passed here to
         * remind this layer what the input was, which it will need in order
         * to do its calculations, and this way the layer doesn't have to
         * remember the most recent input on its own.
         */
        virtual void takeOutputErrorGradients(
                          fml* outputErrorGradients, u32 numOutputDims, u32 outputCount,
                          fml* input, u32 numInputDims, u32 inputCount) = 0;

        /**
         * When takeOutputErrorGradients() is called, the error gradients for this
         * layer's input will be calculated. This method is used to access those
         * input error gradients so that they can be passed down to the layer
         * below this one. The returned matrix is stored column-wise, where each
         * column denotes the error gradients for a single input example. Therefore,
         * 'numInputDims' upon return will contain the number of rows in the matrix,
         * and 'count' upon return will contain the number of columns in the matrix.
         *
         * Do not free or delete the matrix yourself. It is owned by the layer.
         * Therefore, also do not hold a reference to the matrix after this
         * layer is destroyed.
         */
        virtual fml* getInputErrorGradients(u32& numInputDims, u32& count) const = 0;


        //////////////////////////////////////////////////////////////////////
        //
        // Virtual destructor.
        //
        //////////////////////////////////////////////////////////////////////

        virtual ~iLayer() { }
};


}   // namespace ml2


#endif  // __ml2_iLayer_h__
