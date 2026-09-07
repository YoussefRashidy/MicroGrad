package io.github.youssefrashidy.function.math

import io.github.youssefrashidy.MicroGrad
import io.github.youssefrashidy.function.Function
import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.tensor.Tensor
import kotlin.math.ln

object Log : Function() {
    override fun forward(input: FunctionInput): Tensor {
        require(input is FunctionInput.Tensors) {
            "Expected Tensors input, but got ${input::class.simpleName}"
        }

        require(input.tensors.size == 1) {
            "Expected exactly 1 tensor, but got ${input.tensors.size}"
        }

        val tensor = input.tensors[0]
        val targetShape = tensor.shape

        val outputTensor = Tensor(
            DoubleArray(tensor.size),
            targetShape,
            tensor.requiresGrad,
            tensor.strides
        )

        val indices = IntArray(tensor.shape.size) { 0 }

        fun recursiveLog(currentDim: Int, indices: IntArray) {
            if (currentDim == targetShape.size - 1) {
                for (i in 0 until targetShape[currentDim]) {
                    indices[currentDim] = i

                    val value = tensor.get(*indices)

                    require(value > 0.0) {
                        "Log is only defined for positive values, but got $value"
                    }

                    outputTensor.set(
                        *indices,
                        value = ln(value)
                    )
                }
            }
            else {
                for (i in 0 until targetShape[currentDim]) {
                    indices[currentDim] = i
                    recursiveLog(currentDim + 1, indices)
                }
            }
        }

        recursiveLog(0, indices)

        if(MicroGrad.gradEnabled){
            outputTensor.prevTensors = arrayOf(tensor)
            backward(tensor, outputTensor)
        }

        return outputTensor
    }

    override fun backward(vararg tensors: Tensor) {
        val inputTensor = tensors[0]
        val outputTensor = tensors[1]

        if (inputTensor.grad == null && inputTensor.requiresGrad)
            inputTensor.grad = Tensor(
                DoubleArray(inputTensor.size) { 0.0 },
                inputTensor.shape,
                false,
                inputTensor.strides
            )

        if (inputTensor.requiresGrad) {
            outputTensor.gradFn = {
                accumulateLog(
                    inputTensor.grad!!,
                    outputTensor.grad!!,
                    inputTensor
                )
            }
        }
    }
    operator fun invoke(tensor: Tensor): Tensor = forward(FunctionInput.Tensors(arrayOf(tensor)))

}