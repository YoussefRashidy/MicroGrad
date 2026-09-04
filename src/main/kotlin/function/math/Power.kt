package io.github.youssefrashidy.function.math

import io.github.youssefrashidy.function.Function
import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.tensor.Tensor
import kotlin.math.pow

class Power : Function() {

    override fun forward(input: FunctionInput): Tensor {
        require(input is FunctionInput.PowerInput) {
            "Expected PowerInput, but got ${input::class.simpleName}"
        }

        val tensor = input.tensor
        val power = input.power
        val targetShape = tensor.shape
        val outputTensor = Tensor(DoubleArray(tensor.size), targetShape, tensor.requiresGrad, tensor.strides)

        val indices = IntArray(tensor.rank) { 0 }

        fun recursivePower(currentDim: Int, indices: IntArray) {
            if (currentDim == targetShape.size - 1) {
                for (i in 0 until targetShape[currentDim]) {
                    indices[currentDim] = i
                    outputTensor.set(*indices, value = tensor.get(*indices).pow(power))
                }
            }
            else {
                for (i in 0 until targetShape[currentDim]) {
                    indices[currentDim] = i
                    recursivePower(currentDim + 1, indices)
                }
            }
        }

        recursivePower(0, indices)

        outputTensor.prevTensors = arrayOf(tensor)

        backward(tensor, outputTensor, power)

        return outputTensor
    }

    override fun backward(vararg tensors: Tensor) {
        // Power needs the exponent as additional metadata,
        // so the operation-specific helper will handle it.
    }

    private fun backward(inputTensor: Tensor, outputTensor: Tensor, power: Double) {
        if (inputTensor.grad == null)
            inputTensor.grad = Tensor(DoubleArray(inputTensor.size) { 0.0 }, inputTensor.shape, false, inputTensor.strides)

        if (inputTensor.requiresGrad) {
            outputTensor.gradFn = {
                accumulatePower(inputTensor.grad!!,outputTensor.grad!!, inputTensor, power)
            }
        }
    }
}