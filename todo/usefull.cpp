
void tANN::getImage(u32 layerIndex, u32 neuronIndex,
              bool color, u32 width, bool absolute,
              img::tImage* dest) const
{
    // Get the weights.
    tIO weights;
    getWeights(layerIndex, neuronIndex, weights);
    assert(weights.size() > 0);

    // Use the image creating method in ml::common to do the work.
    un_examplify(weights, color, width, absolute, dest);
    u32 height = dest->height();

    // Add an output indicator.
    nLayerType type = m_layers[layerIndex].layerType;
    fml output = (getOutput(layerIndex, neuronIndex) - s_squash_min(type))
                    / (s_squash_max(type) - s_squash_min(type));   // output now in [0, 1]
    u8 outputByte = (u8) (output * FML(255.0));
    u8 red = 0;
    u8 green = (u8) (255 - outputByte);
    u8 blue = outputByte;
    u32 ySpan = height / 5;
    u32 yStart = 0;
    u32 xSpan = width / 5;
    u32 xStart = (u32) (output * ((fml)(width-xSpan)));
    for (u32 r = yStart; r < yStart+ySpan; r++)
    {
        for (u32 c = xStart; c < xStart+xSpan; c++)
        {
            u8* buf = dest->buf() + r*dest->width()*3 + c*3;
            buf[0] = red;
            buf[1] = green;
            buf[2] = blue;
        }
    }
}


        /**
         * Generates an image representation of the specified feature map.
         * The image shows a visual representation of the weights of
         * the connections below the specified feature map.
         *
         * This method uses ml::un_examplify() to create the image, so see
         * that method for a description of the parameters.
         *
         * The generated image is stored in 'dest'.
         */
        void getFeatureMapImage(u32 layerIndex, u32 mapIndex,
                                bool color, bool absolute,
                                img::tImage* dest) const;

        /**
         * Generates an image representation of the output of every
         * replicated filter in the specified feature map. You can
         * think of this as a representation of the transformed
         * input to the specified layer, transformed by the specified
         * feature map.
         *
         * If 'pooled', an image of the pooled output will be created.
         * Else, an image of the full output of the layer will be created.
         *
         * The generated image is stored in 'dest'.
         */
        void getOutputImage(u32 layerIndex, u32 mapIndex,
                            bool pooled,
                            img::tImage* dest) const;

void tCNN::getFeatureMapImage(u32 layerIndex, u32 mapIndex,
                              bool color, bool absolute,
                              img::tImage* dest) const
{
    // Get the weights.
    tIO weights;
    getWeights(layerIndex, mapIndex, weights);
    assert(weights.size() > 0);
    u32 width = m_layers[layerIndex].getReceptiveFieldWidth();
    assert(width > 0);
    if (color)
    {
        if ((width % 3) > 0)
            throw eLogicError("Pixels do not align with width of the receptive field.");
        width /= 3;
    }
    assert(width > 0);

    // Use the image creating method in ml::common to do the work.
    un_examplify(weights, color, width, absolute, dest);
}

void tCNN::getOutputImage(u32 layerIndex, u32 mapIndex,
                          bool pooled,
                          img::tImage* dest) const
{
    if (mapIndex >= getNumFeatureMaps(layerIndex))
        throw eInvalidArgument("No layer/map with that index.");

    // Get the weights.
    tIO weights;
    u32 width;
    {
        u32 stride = getNumFeatureMaps(layerIndex);
        const Mat& alloutput = pooled ? m_layers[layerIndex].getOutput()
                                      : m_layers[layerIndex].getRealOutput();

        if (alloutput.cols() == 0)
            throw eInvalidArgument("There is no \"most recent\" output of this filter.");

        for (u32 i = mapIndex; i < (u32)alloutput.rows(); i += stride)
            weights.push_back(alloutput(i,alloutput.cols()-1));
        assert(weights.size() > 0);

        width = pooled ? (m_layers[layerIndex].getStepsX()+1) / m_layers[layerIndex].getPoolWidth()
                       : (m_layers[layerIndex].getStepsX()+1);
        assert(width > 0);
    }

    // Tell un_examplify() about the range of this data.
    // (Note, when creating images from weight values, the range is (-inf, inf), so it
    // is okay to let un_examplify() determine a good range itself, but here we know
    // the range and we want the resulting image to represent the values relative to that
    // range.
    fml minValue = s_squash_min(m_layers[layerIndex].getLayer().layerType);
    fml maxValue = s_squash_max(m_layers[layerIndex].getLayer().layerType);

    // Use the image creating method in ml::common to do the work.
    un_examplify(weights, false, width, false, dest, &minValue, &maxValue);
}

