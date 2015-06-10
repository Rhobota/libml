#ifndef __ml2_iLayer_h__
#define __ml2_iLayer_h__


#include <ml2/common.h>


namespace ml2
{


class iLayer : public iPackable
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
        virtual void takeInput(const fml* input, u32 numInputDims, u32 count) = 0;

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
        virtual const fml* getOutput(u32& numOutputDims, u32& count) const = 0;


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
         *
         * If you plan to subsequently call getInputErrorGradients(), you
         * need to set 'calculateInputErrorGradients' to true. Otherwise,
         * if you don't need to call getInputErrorGradients(), set it to
         * false to save calculation time.
         */
        virtual void takeOutputErrorGradients(
                          const fml* outputErrorGradients, u32 numOutputDims, u32 outputCount,
                          const fml* input, u32 numInputDims, u32 inputCount,
                          bool calculateInputErrorGradients) = 0;

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
        virtual const fml* getInputErrorGradients(u32& numInputDims, u32& count) const = 0;


        //////////////////////////////////////////////////////////////////////
        //
        // Methods for misc things.
        //
        //////////////////////////////////////////////////////////////////////

        /**
         * Asks the layer to calculate the error between the given output
         * and the given target. For example, the layer may calculate
         * the standard squared error or the cross-entropy loss, if one of
         * those is appropriate. Or the layer may do something else.
         */
        virtual fml calculateError(const tIO& output, const tIO& target) = 0;

        /**
         * Asks the layer to calculate the error between all the given
         * output/target pairs. For example, the layer may calculate
         * the average standard squared error or the average cross-entropy
         * loss, if one of those is appropriate. Or the layer may do
         * something else.
         */
        virtual fml calculateError(const std::vector<tIO>& outputs,
                                   const std::vector<tIO>& targets) = 0;

        /**
         * Resets the layer to its initial state.
         */
        virtual void reset() = 0;

        /**
         * Prints the layer's configuration in a readable format.
         */
        virtual void printLayerInfo(std::ostream& out) const = 0;

        /**
         * Returns a single-line version of printLayerInfo().
         */
        virtual std::string layerInfoString() const = 0;

        /**
         * Returns the header id of this layer type.
         * Each subclass should return a unique id here!
         */
        virtual u32 headerId() const = 0;


        //////////////////////////////////////////////////////////////////////
        //
        // Virtual destructor.
        //
        //////////////////////////////////////////////////////////////////////

        virtual ~iLayer() { }


        //////////////////////////////////////////////////////////////////////
        //
        // Static methods.
        //
        //////////////////////////////////////////////////////////////////////

        /**
         * Functions of this signature know how to read a specific layer
         * subclass from a stream.
         */
        typedef iLayer* (*newLayerFunc)(iReadable* in);

        /**
         * Use this method in each layer subclass to register itself so
         * that it can be read from a stream by newLayerFromStream().
         */
        static bool registerLayerFuncWithHeaderId(newLayerFunc func, u32 headerId);

        /**
         * Call this to read a layer from a stream. The specific layer subclass
         * that is built will not be known by the caller, but that's the beauty
         * of it! For this to work, each layer subclass must register itself
         * by calling registerLayerFuncWithHeaderId() before you call newLayerFromStream().
         */
        static iLayer* newLayerFromStream(iReadable* in);

        /**
         * Writes the given layer to the stream.
         */
        static void writeLayerToStream(iLayer* layer, iWritable* out);
};


}   // namespace ml2


#endif  // __ml2_iLayer_h__
