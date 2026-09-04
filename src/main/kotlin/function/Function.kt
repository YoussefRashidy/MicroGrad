package io.github.youssefrashidy.function

import io.github.youssefrashidy.tensor.Tensor
import kotlin.math.pow

abstract class Function {
    abstract fun forward(input: FunctionInput) : Tensor
    abstract fun backward(vararg tensors: Tensor) : Unit

    protected fun accumulateAddition(targetGradView: Tensor, outputGrad: Tensor) {
        val targetShape = outputGrad.shape
        val indices = IntArray(targetShape.size)

        fun recursiveAccumulate(currentDim: Int) {
            if (currentDim == targetShape.size - 1) {
                for (i in 0 until targetShape[currentDim]) {
                    indices[currentDim] = i
                    val currentVal = targetGradView.get(*indices)
                    val gradVal = outputGrad.get(*indices)
                    targetGradView.set(*indices, value = currentVal + gradVal)
                }
            } else {
                for (i in 0 until targetShape[currentDim]) {
                    indices[currentDim] = i
                    recursiveAccumulate(currentDim + 1)
                }
            }
        }
        recursiveAccumulate(0)
    }

    protected fun accumulateSubstraction(targetGradView: Tensor , outputGrad: Tensor){
        val targetShape = outputGrad.shape
        val indices = IntArray(targetShape.size)

        fun recursiveAccumulate(currentDim: Int) {
            if (currentDim == targetShape.size - 1) {
                for(i in 0 until targetShape[currentDim]) {
                    indices[i] = i
                    targetGradView.set(*indices, value = targetGradView.get(*indices)+ targetGradView.get(*indices) - outputGrad.get(*indices))
                }
            }
            else
                for(i in 0 until targetShape[currentDim]) {
                    indices[i] = i
                    recursiveAccumulate(currentDim + 1)
                }
        }
        recursiveAccumulate(0)
    }

    protected fun accumulateMultiplication(targetGradView: Tensor ,otherTensor: Tensor ,outputGrad: Tensor){
        val targetShape = outputGrad.shape
        val indices = IntArray(targetShape.size)

        fun recursiveAccumulate(currentDim: Int) {
            if (currentDim == targetShape.size - 1) {
                for(i in 0 until targetShape[currentDim]) {
                    indices[i] = i
                    targetGradView.set(*indices, value = targetGradView.get(*indices)+otherTensor.get(*indices) * outputGrad.get(*indices))
                }
            }
            else
                for(i in 0 until targetShape[currentDim]) {
                    indices[i] = i
                    recursiveAccumulate(currentDim + 1)
                }
        }
        recursiveAccumulate(0)
    }

    protected fun accumulateDivisionNumerator(targetGradView: Tensor ,otherTensor: Tensor ,outputGrad: Tensor){
        val targetShape = outputGrad.shape
        val indices = IntArray(targetShape.size)

        fun recursiveAccumulate(currentDim: Int) {
            if (currentDim == targetShape.size - 1) {
                for(i in 0 until targetShape[currentDim]) {
                    indices[i] = i
                    targetGradView.set(*indices, value =  targetGradView.get(*indices)+outputGrad.get(*indices) / otherTensor.get(*indices) )
                }
            }
            else
                for(i in 0 until targetShape[currentDim]) {
                    indices[i] = i
                    recursiveAccumulate(currentDim + 1)
                }
        }
        recursiveAccumulate(0)
    }

    protected fun accumulateDivisionDenominator(targetGradView: Tensor, selfTensor: Tensor, otherTensor: Tensor, outputGrad: Tensor){
        val targetShape = outputGrad.shape
        val indices = IntArray(targetShape.size)

        fun recursiveAccumulate(currentDim: Int) {
            if (currentDim == targetShape.size - 1) {
                for(i in 0 until targetShape[currentDim]) {
                    indices[i] = i
                    val b = selfTensor.get(*indices)
                    val partialB = -otherTensor.get(*indices)/(b*b)
                    targetGradView.set(*indices, value =  targetGradView.get(*indices)+outputGrad.get(*indices)*partialB)
                }
            }
            else
                for(i in 0 until targetShape[currentDim]) {
                    indices[i] = i
                    recursiveAccumulate(currentDim + 1)
                }
        }
        recursiveAccumulate(0)
    }

    protected fun accumulateExponentiation(inputGrad: Tensor, outputGrad: Tensor,output: Tensor){
        val indices = IntArray(inputGrad.shape.size)
        val targetShape = outputGrad.shape
        fun recursiveAccumulate(currentDim: Int) {
            if (currentDim == targetShape.size - 1) {
                for(i in 0 until targetShape[currentDim]) {
                    indices[i] = i
                    inputGrad.set(*indices, value =  inputGrad.get(*indices)+outputGrad.get(*indices)*output.get(*indices))
                }
            }
            else
                for(i in 0 until targetShape[currentDim]) {
                    indices[i] = i
                    recursiveAccumulate(currentDim + 1)
                }
        }
        recursiveAccumulate(0)
    }

    protected fun accumulateLog(inputGrad: Tensor, outputGrad: Tensor, input: Tensor) {
        val indices = IntArray(inputGrad.shape.size)
        val targetShape = outputGrad.shape

        fun recursiveAccumulate(currentDim: Int) {
            if (currentDim == targetShape.size - 1) {
                for (i in 0 until targetShape[currentDim]) {
                    indices[currentDim] = i

                    inputGrad.set(*indices, value = inputGrad.get(*indices)+outputGrad.get(*indices) / input.get(*indices))
                }
            }
            else {
                for (i in 0 until targetShape[currentDim]) {
                    indices[currentDim] = i
                    recursiveAccumulate(currentDim + 1)
                }
            }
        }

        recursiveAccumulate(0)
    }

    protected fun accumulatePower(inputGrad: Tensor, outputGrad: Tensor, input: Tensor, power: Double) {
        val indices = IntArray(inputGrad.shape.size)
        val targetShape = outputGrad.shape

        fun recursiveAccumulate(currentDim: Int) {
            if (currentDim == targetShape.size - 1) {
                for (i in 0 until targetShape[currentDim]) {
                    indices[currentDim] = i
                    val x = input.get(*indices)
                    inputGrad.set(*indices, value = inputGrad.get(*indices)+outputGrad.get(*indices) * power * x.pow(power - 1.0))
                }
            }
            else {
                for (i in 0 until targetShape[currentDim]) {
                    indices[currentDim] = i
                    recursiveAccumulate(currentDim + 1)
                }
            }
        }

        recursiveAccumulate(0)
    }
}


