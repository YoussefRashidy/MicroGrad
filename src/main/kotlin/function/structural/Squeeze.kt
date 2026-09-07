package io.github.youssefrashidy.function.structural

import io.github.youssefrashidy.function.Function
import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.tensor.Tensor

object Squeeze : Function() {

    override fun forward(input: FunctionInput): Tensor {
        require(input is FunctionInput.Tensors) {
            "Expected FunctionInput.Tensors, but got ${input::class.simpleName}"
        }

        val tensors = input.tensors

        require(tensors.size == 1) {
            "Expected exactly 1 tensor, but got ${tensors.size}"
        }

        val inputTensor = tensors[0]

        require(inputTensor.rank > 1) {
            "Cannot squeeze tensor: tensor must have more than one dimension"
        }

        require(inputTensor.shape.last() == 1) {
            "Cannot squeeze the last dimension: expected size 1, but got ${inputTensor.shape.last()}"
        }

        val shape = inputTensor.shape.copyOf(inputTensor.shape.size - 1)

        val outputTensor = Tensor(
            inputTensor.backedArray,
            shape,
            inputTensor.requiresGrad
        )

        outputTensor.prevTensors = arrayOf(inputTensor)

        backward(inputTensor, outputTensor)

        return outputTensor
    }

    override fun backward(vararg tensors: Tensor) {
        val inputTensor = tensors[0]
        val outputTensor = tensors[1]

        if (inputTensor.grad == null)
            inputTensor.grad =
                Tensor(
                    DoubleArray(inputTensor.size),
                    inputTensor.shape,
                    false,
                    inputTensor.strides
                )

        if (inputTensor.requiresGrad)
            outputTensor.gradFn = {
                val unsqueezedGrad = Tensor(
                    outputTensor.grad!!.backedArray,
                    inputTensor.shape,
                    false,
                    inputTensor.strides
                )

                accumulateAddition(
                    inputTensor.grad!!,
                    unsqueezedGrad
                )
            }
    }
    operator fun invoke(tensor : Tensor) : Tensor = forward(FunctionInput.Tensors(arrayOf(tensor)))
}