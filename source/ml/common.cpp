#include <ml/common.h>
#include <ml/iLearner.h>

#include <rho/algo/stat_util.h>
#include <rho/algo/vector_util.h>
#include <rho/img/tCanvas.h>
#include <rho/sync/tTimer.h>

#include <cassert>
#include <limits>
#include <iomanip>
#include <sstream>


namespace ml
{


tIO examplify(u32 highDimension, u32 numDimensions)
{
    if (highDimension >= numDimensions)
        throw eInvalidArgument("highDimension must be < numDimensions");
    tIO target(numDimensions, 0.0);
    target[highDimension] = 1.0;
    return target;
}

u32 un_examplify(const tIO& output, fml* error)
{
    if (output.size() == 0)
        throw eInvalidArgument("The output vector must have at least one dimension!");
    u32 maxindex = 0;
    for (size_t i = 1; i < output.size(); i++)
        if (output[i] > output[maxindex])
            maxindex = (u32)i;
    if (error)
        *error = standardSquaredError(output, examplify(maxindex, (u32)output.size()));
    return maxindex;
}

tIO examplify(const img::tImage* image)
{
    if (image->bufUsed() == 0)
        throw eInvalidArgument("The example image must have at least one pixel in it!");
    const u8* buf = image->buf();
    tIO input(image->bufUsed(), 0.0);
    for (u32 i = 0; i < image->bufUsed(); i++)
        input[i] = buf[i] / FML(255.0);
    return input;
}

void un_examplify(const tIO& io, bool color, u32 width,
                  bool absolute, img::tImage* dest,
                  const fml* minValue, const fml* maxValue)
{
    if (io.size() == 0)
        throw eInvalidArgument("The example io must have at least one dimension!");
    if (width == 0)
        throw eInvalidArgument("Width may not be zero.");

    // Create a copy of io that can be modified.
    tIO weights = io;

    // Normalize the weights to [0.0, 255.0].
    fml maxval;
    fml minval;
    if (minValue && maxValue)
    {
        maxval = *maxValue;
        minval = *minValue;
        if (minval > maxval)
            throw eInvalidArgument("The minValue must be less than or equal to the maxValue.");
        for (u32 i = 0; i < weights.size(); i++)
            if (weights[i] < minval || weights[i] > maxval)
                throw eInvalidArgument("The minValue and maxValue cannot be true given this input vector.");
    }
    else
    {
        maxval = weights[0];
        minval = weights[0];
        for (u32 i = 1; i < weights.size(); i++)
        {
            maxval = std::max(maxval, weights[i]);
            minval = std::min(minval, weights[i]);
        }
        if (maxval == minval) maxval += FML(0.000001);
    }
    fml absmax = std::max(std::fabs(maxval), std::fabs(minval));
    if (color)
    {
        if (absolute)
        {
            for (u32 i = 0; i < weights.size(); i++)
                weights[i] = (std::fabs(weights[i]) / absmax) * FML(255.0);
        }
        else
        {
            for (u32 i = 0; i < weights.size(); i++)
            {
                fml val = ((weights[i] - minval) / (maxval - minval)) * FML(255.0);
                weights[i] = val;
            }
        }
    }

    // Calculate some stuff.
    u32 pixWidth = color ? 3 : 1;
    if ((weights.size() % pixWidth) > 0)
        throw eLogicError("Pixels do not align with the number of weights.");
    u32 numPix = (u32) weights.size() / pixWidth;
    if ((numPix % width) > 0)
        throw eLogicError("Cannot build image of that width. Last row not filled.");
    u32 height = numPix / width;

    // Create the image.
    dest->setFormat(img::kRGB24);
    dest->setBufSize(width*height*3);
    dest->setBufUsed(width*height*3);
    dest->setWidth(width);
    dest->setHeight(height);
    u8* buf = dest->buf();
    u32 bufIndex = 0;
    u32 wIndex = 0;
    for (u32 i = 0; i < height; i++)
    {
        for (u32 j = 0; j < width; j++)
        {
            if (color)
            {
                buf[bufIndex++] = (u8) weights[wIndex++];
                buf[bufIndex++] = (u8) weights[wIndex++];
                buf[bufIndex++] = (u8) weights[wIndex++];
            }
            else
            {
                u8 r = 0;     // <-- used if the weight is negative
                u8 g = 0;     // <-- used if the weight is positive
                u8 b = 0;     // <-- not used

                fml w = weights[wIndex++];

                if (w > FML(0.0))
                    g = (u8)(w / absmax * FML(255.0));

                if (w < FML(0.0))
                    r = (u8)(-w / absmax * FML(255.0));

                buf[bufIndex++] = r;
                buf[bufIndex++] = g;
                buf[bufIndex++] = b;
            }
        }
    }
}

void zscore(std::vector<tIO>& inputs, u32 dStart, u32 dEnd)
{
    // Make sure all the input looks okay.
    if (inputs.size() == 0)
        throw eInvalidArgument("There must be at least one training input!");
    for (size_t i = 1; i < inputs.size(); i++)
    {
        if (inputs[i].size() != inputs[0].size())
        {
            throw eInvalidArgument("Every training input must have the same dimensionality!");
        }
    }

    // For every dimension, we'll need to create a vector of all that dimensions examples.
    tIO dim(inputs.size(), 0.0);

    // For every dimension...
    for (size_t d = dStart; d < inputs[0].size() && d < dEnd; d++)
    {
        // ... Fill 'dim' with that dimension
        for (size_t i = 0; i < inputs.size(); i++)
            dim[i] = inputs[i][d];

        // ... Calculate the mean and stddev
        fml mean = algo::mean(dim);
        fml stddev = algo::stddev(dim);

        // ... Normalize that dimension
        if (stddev != FML(0.0))
        {
            for (size_t i = 0; i < inputs.size(); i++)
                inputs[i][d] = (inputs[i][d] - mean) / stddev;
        }
        else
        {
            for (size_t i = 0; i < inputs.size(); i++)
                inputs[i][d] = 0.0;
        }
    }
}

void zscore(std::vector<tIO>& trainingInputs, std::vector<tIO>& testInputs)
{
    // Make sure all the input looks okay.
    if (trainingInputs.size() == 0)
        throw eInvalidArgument("There must be at least one training input!");
    if (testInputs.size() == 0)
        throw eInvalidArgument("There must be at least one test input!");
    for (size_t i = 1; i < trainingInputs.size(); i++)
    {
        if (trainingInputs[i].size() != trainingInputs[0].size())
        {
            throw eInvalidArgument("Every training input must have the same dimensionality!");
        }
    }
    for (size_t i = 1; i < testInputs.size(); i++)
    {
        if (testInputs[i].size() != testInputs[0].size())
        {
            throw eInvalidArgument("Every test input must have the same dimensionality!");
        }
    }
    if (trainingInputs[0].size() != testInputs[0].size())
        throw eInvalidArgument("The training and test examples must all have the same dimensionality!");

    // For every dimension, we'll need to create a vector of all that dimensions examples.
    tIO dim(trainingInputs.size(), 0.0);

    // For every dimension...
    for (size_t d = 0; d < trainingInputs[0].size(); d++)
    {
        // ... Fill 'dim' with that dimension
        for (size_t i = 0; i < trainingInputs.size(); i++)
            dim[i] = trainingInputs[i][d];

        // ... Calculate the mean and stddev
        fml mean = algo::mean(dim);
        fml stddev = algo::stddev(dim);

        // ... Normalize that dimension
        if (stddev != FML(0.0))
        {
            for (size_t i = 0; i < trainingInputs.size(); i++)
                trainingInputs[i][d] = (trainingInputs[i][d] - mean) / stddev;
            for (size_t i = 0; i < testInputs.size(); i++)
                testInputs[i][d] = (testInputs[i][d] - mean) / stddev;
        }
        else
        {
            for (size_t i = 0; i < trainingInputs.size(); i++)
                trainingInputs[i][d] = 0.0;
            for (size_t i = 0; i < testInputs.size(); i++)
                testInputs[i][d] = 0.0;
        }
    }
}


fml standardSquaredError(const tIO& output, const tIO& target)
{
    if (output.size() != target.size())
        throw eInvalidArgument(
                "The output vector must have the same dimensionality as the target vector!");
    if (output.size() == 0)
        throw eInvalidArgument("The output and target vectors must have at least one dimension!");
    fml error = 0.0;
    for (size_t i = 0; i < output.size(); i++)
        error += (output[i]-target[i]) * (output[i]-target[i]);
    return FML(0.5) * error;
}

fml standardSquaredError(const std::vector<tIO>& outputs,
                         const std::vector<tIO>& targets)
{
    if (outputs.size() != targets.size())
    {
        throw eInvalidArgument("The number of examples in outputs and targets must "
                "be the same!");
    }

    if (outputs.size() == 0)
    {
        throw eInvalidArgument("There must be at least one output/target pair!");
    }

    for (size_t i = 0; i < outputs.size(); i++)
    {
        if (outputs[i].size() != targets[i].size() ||
            outputs[i].size() != outputs[0].size())
        {
            throw eInvalidArgument("Every output/target pair must have the same dimensionality!");
        }
    }

    if (outputs[0].size() == 0)
    {
        throw eInvalidArgument("The output/target pairs must have at least one dimension!");
    }

    fml error = 0.0;
    for (size_t i = 0; i < outputs.size(); i++)
        error += standardSquaredError(outputs[i], targets[i]);
    return error / ((fml)outputs.size());
}

fml crossEntropyCost(const tIO& output, const tIO& target)
{
    if (output.size() != target.size())
        throw eInvalidArgument(
                "The output vector must have the same dimensionality as the target vector!");
    if (output.size() == 0)
        throw eInvalidArgument("The output and target vectors must have at least one dimension!");
    fml osum = 0.0;
    fml tsum = 0.0;
    for (size_t i = 0; i < output.size(); i++)
    {
        if (output[i] > FML(1.0))
            throw eInvalidArgument("The output value cannot be >1.0 when it represents a probability.");
        if (output[i] < FML(0.0))
            throw eInvalidArgument("The output value cannot be <0.0 when it represents a probability.");
        if (target[i] > FML(1.0))
            throw eInvalidArgument("The target value cannot be >1.0 when it represents a probability.");
        if (target[i] < FML(0.0))
            throw eInvalidArgument("The target value cannot be <0.0 when it represents a probability.");
        osum += output[i];
        tsum += target[i];
    }
    if (osum > FML(1.0001) || osum < FML(0.9999))
        throw eInvalidArgument("The sum of the outputs must be 1.0.");
    if (tsum > FML(1.0001) || tsum < FML(0.9999))
        throw eInvalidArgument("The sum of the targets must be 1.0.");
    fml error = 0.0;
    for (size_t i = 0; i < output.size(); i++)
        if (target[i] > FML(0.0))
            error += target[i] * std::log(output[i]);
    return -error;
}

fml crossEntropyCost(const std::vector<tIO>& outputs,
                     const std::vector<tIO>& targets)
{
    if (outputs.size() != targets.size())
    {
        throw eInvalidArgument("The number of examples in outputs and targets must "
                "be the same!");
    }

    if (outputs.size() == 0)
    {
        throw eInvalidArgument("There must be at least one output/target pair!");
    }

    for (size_t i = 0; i < outputs.size(); i++)
    {
        if (outputs[i].size() != targets[i].size() ||
            outputs[i].size() != outputs[0].size())
        {
            throw eInvalidArgument("Every output/target pair must have the same dimensionality!");
        }
    }

    if (outputs[0].size() == 0)
    {
        throw eInvalidArgument("The output/target pairs must have at least one dimension!");
    }

    fml error = 0.0;
    for (size_t i = 0; i < outputs.size(); i++)
        error += crossEntropyCost(outputs[i], targets[i]);
    return error / ((fml)outputs.size());
}

fml rmsError(const std::vector<tIO>& outputs,
             const std::vector<tIO>& targets)
{
    fml sqrdError = standardSquaredError(outputs, targets);
    return std::sqrt(sqrdError * FML(2.0) / ((fml)outputs[0].size()));
}


void buildConfusionMatrix(const std::vector<tIO>& outputs,
                          const std::vector<tIO>& targets,
                                tConfusionMatrix& confusionMatrix)
{
    if (outputs.size() != targets.size())
    {
        throw eInvalidArgument("The number of examples in outputs and targets must "
                "be the same!");
    }

    if (outputs.size() == 0)
    {
        throw eInvalidArgument("There must be at least one output/target pair!");
    }

    for (size_t i = 0; i < outputs.size(); i++)
    {
        if (outputs[i].size() != targets[i].size() ||
            outputs[i].size() != outputs[0].size())
        {
            throw eInvalidArgument("Every output/target pair must have the same dimensionality!");
        }
    }

    if (outputs[0].size() == 0)
    {
        throw eInvalidArgument("The output/target pairs must have at least one dimension!");
    }

    confusionMatrix.resize(targets[0].size());
    for (size_t i = 0; i < confusionMatrix.size(); i++)
        confusionMatrix[i] = std::vector<u32>(outputs[0].size(), 0);

    for (size_t i = 0; i < outputs.size(); i++)
    {
        u32 target = un_examplify(targets[i]);
        u32 output = un_examplify(outputs[i]);
        confusionMatrix[target][output]++;
    }
}

static
void s_fill_cell(const std::vector<u32>& indices, const std::vector<tIO>& inputs,
                 bool color, u32 width, bool absolute,
                 u32 boxWidth, f64 ox, f64 oy,
                 algo::iLCG& lcg, img::tCanvas& canvas)
{
    const u32 kPadding = 5;

    img::tImage image;

    u8 bgColor[3] = { 255, 255, 255 };    // white
    img::tCanvas subcanvas(img::kRGB24, bgColor, 3);
    u32 subcanvX = 0;
    u32 subcanvY = 0;
    bool subcanvFull = false;

    for (size_t i = 0; i < indices.size(); i++)
    {
        un_examplify(inputs[indices[i]], color, width, absolute, &image);

        if (!subcanvFull)
        {
            subcanvas.drawImage(&image, subcanvX, subcanvY);
            subcanvX += image.width() + kPadding;
            if (subcanvX + image.width() + kPadding > boxWidth)
            {
                subcanvX = 0;
                subcanvY += image.height() + kPadding;
                if (subcanvY + image.height() + kPadding > boxWidth)
                {
                    subcanvFull = true;
                    subcanvas.genImage(&image);
                    f64 rx = ox + boxWidth / 2.0 - image.width() / 2.0;
                    f64 ry = oy + boxWidth / 2.0 - image.height() / 2.0;
                    canvas.drawImage(&image, (i32) round(rx), (i32) round(ry));
                }
            }
        }

        else
        {
            f64 rx = ((f64)lcg.next()) / ((f64)lcg.randMax()) * (boxWidth-width) + ox;
            f64 ry = ((f64)lcg.next()) / ((f64)lcg.randMax()) * (boxWidth-width) + oy;
            canvas.drawImage(&image, (i32) round(rx), (i32) round(ry));
        }
    }

    if (!subcanvFull)
    {
        subcanvas.genImage(&image);
        f64 rx = ox + boxWidth / 2.0 - image.width() / 2.0;
        f64 ry = oy + boxWidth / 2.0 - image.height() / 2.0;
        canvas.drawImage(&image, (i32) round(rx), (i32) round(ry));
    }
}

static
void s_drawGrid(img::tCanvas& canvas, u32 gridSize, u32 distBetweenLines)
{
    {
        img::tImage horiz;
        horiz.setFormat(img::kRGB24);
        horiz.setWidth(gridSize*distBetweenLines);
        horiz.setHeight(1);
        horiz.setBufSize(horiz.width() * horiz.height() * 3);
        horiz.setBufUsed(horiz.bufSize());
        for (u32 i = 0; i < horiz.bufUsed(); i++) horiz.buf()[i] = 0;  // <-- makes the lines black
        for (u32 i = 0; i <= gridSize; i++)
            canvas.drawImage(&horiz, 0, i*distBetweenLines);
    }

    {
        img::tImage vert;
        vert.setFormat(img::kRGB24);
        vert.setWidth(1);
        vert.setHeight(gridSize*distBetweenLines);
        vert.setBufSize(vert.width() * vert.height() * 3);
        vert.setBufUsed(vert.bufSize());
        for (u32 i = 0; i < vert.bufUsed(); i++) vert.buf()[i] = 0;  // <-- makes the lines black
        for (u32 i = 0; i <= gridSize; i++)
            canvas.drawImage(&vert, i*distBetweenLines, 0);
    }
}

void buildVisualConfusionMatrix(const std::vector<tIO>& inputs,
                                bool color, u32 width, bool absolute,
                                const std::vector<tIO>& outputs,
                                const std::vector<tIO>& targets,
                                      img::tImage* dest,
                                u32 cellWidthMultiplier)
{
    if (outputs.size() != targets.size())
    {
        throw eInvalidArgument("The number of examples in outputs and targets must "
                "be the same!");
    }

    if (outputs.size() != inputs.size())
    {
        throw eInvalidArgument("The number of examples in outputs and inputs must "
                "be the same!");
    }

    if (outputs.size() == 0)
    {
        throw eInvalidArgument("There must be at least one output/target pair!");
    }

    for (size_t i = 0; i < outputs.size(); i++)
    {
        if (outputs[i].size() != targets[i].size() ||
            outputs[i].size() != outputs[0].size())
        {
            throw eInvalidArgument("Every output/target pair must have the same dimensionality!");
        }
    }

    if (outputs[0].size() == 0)
    {
        throw eInvalidArgument("The output/target pairs must have at least one dimension!");
    }

    u32 numClasses = (u32) targets[0].size();      // same as outputs[0].size()

    std::vector< std::vector< std::vector<u32> > > holding(numClasses,
            std::vector< std::vector<u32> >(numClasses, std::vector<u32>()));

    for (size_t i = 0; i < outputs.size(); i++)    // same as targets.size()
    {
        u32 target = un_examplify(targets[i]);
        u32 output = un_examplify(outputs[i]);
        holding[target][output].push_back((u32)i);
    }

    algo::tKnuthLCG lcg;

    u32 boxWidth = cellWidthMultiplier * width;
    u8 bgColor[3] = { 255, 255, 255 };    // white
    img::tCanvas canvas(img::kRGB24, bgColor, 3);

    for (size_t i = 0; i < holding.size(); i++)
    {
        for (size_t j = 0; j < holding[i].size(); j++)
        {
            s_fill_cell(holding[i][j], inputs,
                        color, width, absolute,
                        boxWidth, (f64)(j*boxWidth), (f64)(i*boxWidth),
                        lcg, canvas);
        }
    }

    canvas.expandToIncludePoint(0, 0);
    canvas.expandToIncludePoint(numClasses*boxWidth, numClasses*boxWidth);
    s_drawGrid(canvas, numClasses, boxWidth);
    canvas.genImage(dest);
}

static
void checkConfusionMatrix(const tConfusionMatrix& confusionMatrix)
{
    if (confusionMatrix.size() == 0)
        throw eInvalidArgument("Invalid confusion matrix");

    for (size_t i = 0; i < confusionMatrix.size(); i++)
    {
        if (confusionMatrix[i].size() != confusionMatrix.size())
            throw eInvalidArgument("Invalid confusion matrix");
    }
}

static
void printDashes(const tConfusionMatrix& confusionMatrix, std::ostream& out, u32 s, u32 w)
{
    for (u32 i = 1; i < s; i++)
        out << " ";
    out << "+";
    for (size_t j = 0; j < confusionMatrix[0].size(); j++)
    {
        for (u32 i = 1; i < w; i++)
            out << "-";
        out << "+";
    }
    out << std::endl;
}

void print(const tConfusionMatrix& confusionMatrix, std::ostream& out)
{
    checkConfusionMatrix(confusionMatrix);

    u32 s = 14;
    u32 w = 10;

    out << "                   predicted" << std::endl;

    printDashes(confusionMatrix, out, s, w);

    for (size_t i = 0; i < confusionMatrix.size(); i++)
    {
        if (i == confusionMatrix.size()/2)
            out << "  correct    |";
        else
            out << "             |";
        for (size_t j = 0; j < confusionMatrix[i].size(); j++)
        {
            out << " " << std::right << std::setw(w-3) << confusionMatrix[i][j] << " |";
        }
        out << std::endl;
    }

    printDashes(confusionMatrix, out, s, w);

    out << std::endl;
}

f64  errorRate(const tConfusionMatrix& confusionMatrix)
{
    checkConfusionMatrix(confusionMatrix);

    u32 total = 0;
    for (size_t i = 0; i < confusionMatrix.size(); i++)
        for (size_t j = 0; j < confusionMatrix[i].size(); j++)
            total += confusionMatrix[i][j];
    u32 correct = 0;
    for (size_t i = 0; i < confusionMatrix.size(); i++)
        correct += confusionMatrix[i][i];
    return ((f64)(total - correct)) / total;
}

f64  accuracy(const tConfusionMatrix& confusionMatrix)
{
    checkConfusionMatrix(confusionMatrix);

    u32 total = 0;
    for (size_t i = 0; i < confusionMatrix.size(); i++)
        for (size_t j = 0; j < confusionMatrix[i].size(); j++)
            total += confusionMatrix[i][j];
    u32 correct = 0;
    for (size_t i = 0; i < confusionMatrix.size(); i++)
        correct += confusionMatrix[i][i];
    return ((f64)correct) / total;
}

f64  precision(const tConfusionMatrix& confusionMatrix)
{
    checkConfusionMatrix(confusionMatrix);

    if (confusionMatrix.size() != 2)
        throw eInvalidArgument("Precision is only defined for boolean classification.");

    f64 tp = (f64) confusionMatrix[1][1];
    f64 fp = (f64) confusionMatrix[0][1];
    //f64 tn = (f64) confusionMatrix[0][0];
    //f64 fn = (f64) confusionMatrix[1][0];

    return tp / (tp + fp);
}

f64  recall(const tConfusionMatrix& confusionMatrix)
{
    checkConfusionMatrix(confusionMatrix);

    if (confusionMatrix.size() != 2)
        throw eInvalidArgument("Recall is only defined for boolean classification.");

    f64 tp = (f64) confusionMatrix[1][1];
    //f64 fp = (f64) confusionMatrix[0][1];
    //f64 tn = (f64) confusionMatrix[0][0];
    f64 fn = (f64) confusionMatrix[1][0];

    return tp / (tp + fn);
}


bool train(iLearner* learner, iInputTargetGenerator* generator,
                              u32 batchSize,
                              iTrainObserver* trainObserver)
{
    if (!learner)
        throw eInvalidArgument("learner must not be NULL!");

    if (!generator)
        throw eInvalidArgument("generator must not be NULL!");

    if (batchSize == 0)
        throw eInvalidArgument("batchSize must be positive!");

    std::vector< std::pair<tIO,tIO> > batch(batchSize);

    while (generator->generate(batchSize, batch) && batch.size() > 0)
    {
        for (size_t i = 0; i < batch.size(); i++)
            learner->addExample(batch[i].first, batch[i].second);
        learner->update();
        if (trainObserver && !trainObserver->didUpdate(learner, batch))
            return false;
    }

    return true;
}

void evaluate(iLearner* learner, iInputTargetGenerator* generator,
                                 iOutputCollector* collector,
                                 u32 batchSize)
{
    if (!learner)
        throw eInvalidArgument("learner must not be NULL!");

    if (!generator)
        throw eInvalidArgument("generator must not be NULL!");

    if (!collector)
        throw eInvalidArgument("collector must not be NULL!");

    if (batchSize == 0)
        throw eInvalidArgument("batchSize must be positive!");

    std::vector<tIO> batch(batchSize);
    std::vector<tIO> outputs;

    while (generator->generate(batchSize, batch) && batch.size() > 0)
    {
        outputs.resize(batch.size());
        learner->evaluateBatch(batch.begin(),
                               batch.end(),
                               outputs.begin());
    }
}

/*
 * This function is used "deinterlace" ("deinterleave" is the correct
 * term, actually) a vector of repeating component.
 *
 * For example, say you have a vector with the contents: a1b2c3d4
 * And you want to convert that to a vector: abcd1234
 * To do that call this function with numComponents=2 and unitLength=1.
 *
 * Or, say you have a vector with the contents: ab12cd34ef56
 * And you want to convert that to a vector: abcdef123456
 * To do that call this function with numComponents=2 and unitLength=2.
 *
 * Or, say you have a vector with the contents: RGBRGBRGBRGB
 * And you want to convert that to a vector: RRRRGGGGBBBB
 * To do that call this function with numComponents=3 and unitLength=1.
 *
 * 'output' must be allocated by the caller, and of course delete by
 * the caller as well.
 */
template <class T>
void deinterlace(const T* input, T* output, u32 arrayLen, u32 numComponents, u32 unitLength=1)
{
    assert((arrayLen % unitLength) == 0);
    u32 numUnits = arrayLen / unitLength;

    assert((numUnits % numComponents) == 0);
    u32 groupSize = numUnits / numComponents;

    u32 stride = groupSize * unitLength;

    u32 s = 0;

    for (u32 g = 0; g < groupSize; g++)
    {
        u32 d = g * unitLength;

        for (u32 c = 0; c < numComponents; c++)
        {
            for (u32 u = 0; u < unitLength; u++)
                output[d+u] = input[s+u];

            s += unitLength;
            d += stride;
        }
    }
}

void visualize(iLearner* learner, const tIO& example,
               bool color, u32 width, bool absolute,
               img::tImage* dest)
{
    // TODO
    u8 bgColor[3] = { 0, 0, 205 };    // "Medium Blue" from http://www.tayloredmktg.com/rgb/
    img::tCanvas canvas(img::kRGB24, bgColor, 3);
    canvas.genImage(dest);
}


u32  ezTrain(iLearner* learner, iInputTargetGenerator* trainingSetGenerator,
                                iInputTargetGenerator* testSetGenerator,
                                u32 batchSize,
                                iEZTrainObserver* trainObserver)
{
    if (!learner)
        throw eInvalidArgument("learner must not be NULL!");

    if (!trainingSetGenerator)
        throw eInvalidArgument("trainingSetGenerator must not be NULL!");

    if (!testSetGenerator)
        throw eInvalidArgument("testSetGenerator must not be NULL!");

    if (batchSize == 0)
        throw eInvalidArgument("batchSize must be positive!");

    u64 trainStartTime = sync::tTimer::usecTime();

    for (u32 epochs = 0; true; epochs++)
    {
        // Note the start time of this epoch.
        u64 startTime = sync::tTimer::usecTime();

        // Train if this is not the zero'th epoch. This is so that the user will get a
        // callback before any training has happened, so that the user knows what the
        // initial state of the learner looks like.
        if (epochs > 0)
        {
            if (! train(learner, trainingSetGenerator,
                        batchSize, trainObserver))
            {
                if (trainObserver)
                {
                    f64 trainElapsedTime = (f64)(sync::tTimer::usecTime() - trainStartTime);
                    trainElapsedTime /= 1000000;  // usecs to secs
                    trainObserver->didFinishTraining(learner, epochs-1,
                                                     trainElapsedTime);
                }
                return epochs-1;
            }

            // Shuffle the training data for the next iteration.
            trainingSetGenerator->shuffle();
        }

        // Call the epoch observer.
        if (trainObserver)
        {
            // Get the evaluation method used by this learner.
            iOutputPerformanceEvaluator* evaluator = learner->getOutputPerformanceEvaluator();
            evaluator->reset();

            // Evaluate the learner using the training set.
            evaluate(learner, trainingSetGenerator, evaluator, batchSize);
            trainingSetGenerator->restart();
            f64 trainingSetPerformance = evaluator->calculatePerformance();
            evaluator->reset();

            // Evaluate the learner using the test set.
            evaluate(learner, testSetGenerator, evaluator, batchSize);
            testSetGenerator->restart();
            f64 testSetPerformance = evaluator->calculatePerformance();
            evaluator->reset();

            // Calculate the elapsed time.
            f64 elapsedTime = (f64)(sync::tTimer::usecTime() - startTime);
            elapsedTime /= 1000000;  // usecs to secs

            if (! trainObserver->didFinishEpoch(learner, epochs, elapsedTime,
                                                trainingSetPerformance, testSetPerformance,
                                                evaluator->isPositivePerformanceGood()))
            {
                f64 trainElapsedTime = (f64)(sync::tTimer::usecTime() - trainStartTime);
                trainElapsedTime /= 1000000;  // usecs to secs
                trainObserver->didFinishTraining(learner, epochs,
                                                 trainElapsedTime);
                return epochs;
            }
        }
    }
}


tSmartStoppingWrapper::tSmartStoppingWrapper(u32 minEpochs,
                                             u32 maxEpochs,
                                             f64 significantThreshold,
                                             f64 patienceIncrease,
                                             iEZTrainObserver* wrappedObserver)
    : m_minEpochs(minEpochs),
      m_maxEpochs(maxEpochs),
      m_significantThreshold(significantThreshold),
      m_patienceIncrease(patienceIncrease),
      m_obs(wrappedObserver)
{
    if (m_maxEpochs < m_minEpochs)
        throw eInvalidArgument("max epochs must be >= min epochs");
    if (m_significantThreshold < 0.0)
        throw eInvalidArgument("The significance threshold cannot be less than zero.");
    if (m_significantThreshold >= 1.0)
        throw eInvalidArgument("The significance threshold must be less than 1.0.");
    if (m_patienceIncrease <= 1.0)
        throw eInvalidArgument("The patience increase must be greater than 1.0.");
    m_reset();
}

bool tSmartStoppingWrapper::didUpdate(iLearner* learner, const std::vector< std::pair<tIO,tIO> >& mostRecentBatch)
{
    return (!m_obs || m_obs->didUpdate(learner, mostRecentBatch));
}

bool tSmartStoppingWrapper::didFinishEpoch(iLearner* learner,
                                           u32 epochsCompleted,
                                           f64 epochTrainTimeInSeconds,
                                           f64 trainingSetPerformance,
                                           f64 testSetPerformance,
                                           bool positivePerformanceIsGood)
{
    if (m_obs && !m_obs->didFinishEpoch(learner,
                                        epochsCompleted,
                                        epochTrainTimeInSeconds,
                                        trainingSetPerformance,
                                        testSetPerformance,
                                        positivePerformanceIsGood))
    {
        return false;
    }

    if (isnan(m_bestFoundTestSetPerformance))
    {
        m_bestFoundTestSetPerformance = testSetPerformance;
    }
    else
    {
        if (positivePerformanceIsGood)
        {
            if (testSetPerformance > m_bestFoundTestSetPerformance * (1.0 + m_significantThreshold))
            {
                m_bestFoundTestSetPerformance = testSetPerformance;
                m_allowedEpochs = (u32)std::ceil(std::max((f64)m_minEpochs, epochsCompleted * m_patienceIncrease));
            }
        }
        else
        {
            if (testSetPerformance < m_bestFoundTestSetPerformance * (1.0 - m_significantThreshold))
            {
                m_bestFoundTestSetPerformance = testSetPerformance;
                m_allowedEpochs = (u32)std::ceil(std::max((f64)m_minEpochs, epochsCompleted * m_patienceIncrease));
            }
        }
    }

    return (epochsCompleted < m_allowedEpochs && epochsCompleted < m_maxEpochs);
}

void tSmartStoppingWrapper::didFinishTraining(iLearner* learner,
                                              u32 epochsCompleted,
                                              f64 trainingTimeInSeconds)
{
    if (m_obs) m_obs->didFinishTraining(learner, epochsCompleted,
                                        trainingTimeInSeconds);
    m_reset();
}

void tSmartStoppingWrapper::m_reset()
{
    m_bestFoundTestSetPerformance = std::numeric_limits<double>::quiet_NaN();
    m_allowedEpochs = m_minEpochs;
}


tBestRememberingWrapper::tBestRememberingWrapper(iEZTrainObserver* wrappedObserver)
    : m_bestAfterEpochsCompleted(0),
      m_bestTestSetPerformance(0.0),
      m_serializedLearner(),
      m_obs(wrappedObserver)
{
    reset();
}

void tBestRememberingWrapper::reset()
{
    m_bestAfterEpochsCompleted = 0;
    m_bestTestSetPerformance = std::numeric_limits<double>::quiet_NaN();
}

u32 tBestRememberingWrapper::bestAfterEpochsCompleted() const
{
    return m_bestAfterEpochsCompleted;
}

f64 tBestRememberingWrapper::bestTestSetPerformance()   const
{
    return m_bestTestSetPerformance;
}

iLearner* tBestRememberingWrapper::newBestLearner() const
{
    tByteReadable readable(m_serializedLearner.getBuf());
    return iLearner::newLearnerFromStream(&readable);
}

bool tBestRememberingWrapper::didUpdate(iLearner* learner, const std::vector< std::pair<tIO,tIO> >& mostRecentBatch)
{
    return (!m_obs || m_obs->didUpdate(learner, mostRecentBatch));
}

bool tBestRememberingWrapper::didFinishEpoch(iLearner* learner,
                                             u32 epochsCompleted,
                                             f64 epochTrainTimeInSeconds,
                                             f64 trainingSetPerformance,
                                             f64 testSetPerformance,
                                             bool positivePerformanceIsGood)
{
    // Delegate to the wrapped object whether or not to quit training.
    bool retVal = (!m_obs || m_obs->didFinishEpoch(learner,
                                                   epochsCompleted,
                                                   epochTrainTimeInSeconds,
                                                   trainingSetPerformance,
                                                   testSetPerformance,
                                                   positivePerformanceIsGood));

    // If this is the zero'th epoch, reset myself in case I'm being re-used
    // and the user forgot to reset me.
    if (epochsCompleted == 0)
        reset();

    // Evaluate the performance on the test set and see if it's the best yet.
    if ((isnan(m_bestTestSetPerformance))                                                    ||
        (positivePerformanceIsGood  && testSetPerformance > m_bestTestSetPerformance)        ||
        (!positivePerformanceIsGood && testSetPerformance < m_bestTestSetPerformance)        )
    {
        m_bestAfterEpochsCompleted = epochsCompleted;
        m_bestTestSetPerformance = testSetPerformance;
        m_serializedLearner.reset();
        iLearner::writeLearnerToStream(learner, &m_serializedLearner);
    }

    return retVal;
}

void tBestRememberingWrapper::didFinishTraining(iLearner* learner,
                                                u32 epochsCompleted,
                                                f64 trainingTimeInSeconds)
{
    if (m_obs) m_obs->didFinishTraining(learner, epochsCompleted,
                                        trainingTimeInSeconds);
}


}   // namespace ml
