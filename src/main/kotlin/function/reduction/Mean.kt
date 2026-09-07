package io.github.youssefrashidy.function.reduction

import io.github.youssefrashidy.function.Function
import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.function.structural.Reshape
import io.github.youssefrashidy.tensor.Tensor

object Mean: Function() {
    override fun forward(input: FunctionInput): Tensor {
        require(input is FunctionInput.MeanInput) {
            "Expected input type to be ${FunctionInput.MeanInput::class.simpleName} but got ${input::class.simpleName}"
        }

        val tensor = input.tensor
        val axes = input.axes
        val keepDims = input.keepDims

        val reduced = BooleanArray(tensor.shape.size)
        var count = 1

        for (axis in axes) {
            reduced[axis] = true
            count *= tensor.shape[axis]
        }

        val outputDims = IntArray(if (keepDims) tensor.shape.size else tensor.shape.size - axes.size)

        val keepDimsShape = IntArray(tensor.shape.size)

        var j = 0
        for (i in tensor.shape.indices) {
            if (reduced[i]) {
                if (keepDims)
                    outputDims[i] = 1

                keepDimsShape[i] = 1
            }
            else {
                outputDims[if (keepDims) i else j] = tensor.shape[i]
                if (!keepDims)
                    j++

                keepDimsShape[i] = tensor.shape[i]
            }
        }

        val outputTensor = Tensor(DoubleArray(outputDims.fold(1) { acc, dim -> acc * dim }) { 0.0 }, outputDims, tensor.requiresGrad)

        val indices = IntArray(tensor.shape.size) { 0 }
        val outputIndices = IntArray(outputDims.size)
        val inputDims = tensor.shape

        fun recursiveMean(currentDim: Int, outputDim: Int): Unit {
            if (currentDim == inputDims.size - 1) {
                for (i in 0 until inputDims[currentDim]) {
                    indices[currentDim] = i
                    if (!reduced[currentDim])
                        outputIndices[outputDim] = i

                    outputTensor.set(*outputIndices, value = outputTensor.get(*outputIndices) + tensor.get(*indices) / count)
                }
            }
            else {
                for (i in 0 until inputDims[currentDim]) {
                    indices[currentDim] = i
                    if (!reduced[currentDim])
                        outputIndices[outputDim] = i

                    recursiveMean(currentDim + 1, if (!reduced[currentDim] || keepDims) outputDim + 1 else outputDim)
                }
            }
        }

        recursiveMean(0, 0)

        outputTensor.prevTensors = arrayOf(tensor)
        backward(tensor, outputTensor, keepDimsShape,count)

        return outputTensor
    }

    override fun backward(vararg tensors: Tensor) {
        // TODO("Will settle for an API later")
    }

    private fun backward(inputTensor: Tensor, outputTensor: Tensor, gradShape: IntArray , divisor: Int) {
        if (inputTensor.grad == null && inputTensor.requiresGrad)
            inputTensor.grad = Tensor(DoubleArray(inputTensor.size), inputTensor.shape, false, inputTensor.strides)

        outputTensor.gradFn = {
            val reshapedOutputGrad = Reshape.reshape(outputTensor.grad!!, gradShape)
            val (_, broadcastedOutputGrad, _) = Tensor.broadcast(inputTensor.grad!!,reshapedOutputGrad)
            accumulateAdditionDivided(inputTensor.grad!!, broadcastedOutputGrad,divisor)
        }
    }
}