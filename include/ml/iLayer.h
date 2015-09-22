#ifndef __ml_iLayer_h__
#define __ml_iLayer_h__


#include <ml/common.h>


namespace ml
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
         *
         * The layer knows if this input is for training based on the 'isTrainMode'
         * parameter. (Some layers do different things in train vs test mode.)
         *
         * A pointer to the previous layer is passed just for fun, in case the
         * layer wants to probe it for some reason. It will be NULL if this is
         * the first layer.
         */
        virtual void takeInput(const fml* input, u32 numInputDims, u32 count,
                               bool isTrainMode, iLayer* prevLayer) = 0;

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

        /**
         * This method should return true if this layer actually learns
         * things. It should return false if this layer is static and
         * never learns. E.g. A fully connected neural network layer
         * would return true here. But a max-pooling layer would return
         * false.
         *
         * This method is used to optimize the learning process so that
         * non-learning layers can be skipped during learning (where possible).
         */
        virtual bool doesLearn() const = 0;


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


}   // namespace ml


#endif  // __ml_iLayer_h__
