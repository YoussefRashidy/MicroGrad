package io.github.youssefrashidy.function.reduction

import io.github.youssefrashidy.function.Function
import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.function.structural.Reshape
import io.github.youssefrashidy.tensor.Tensor

object Max: Function() {
    override fun forward(input: FunctionInput): Tensor {
        require(input is FunctionInput.MaxInput){

        }
        val tensor = input.tensor
        val axes = input.axes
        val keepDims = input.keepDims
        val reduced = BooleanArray(tensor.shape.size)
        for (axis in axes) {
            reduced[axis] = true
        }

        val outputDims = IntArray(if(keepDims) tensor.shape.size else tensor.shape.size - axes.size)
        var j = 0 ;
        for (i in tensor.shape.indices) {
            if (reduced[i]) {
                if (keepDims)
                    outputDims[i] = 1
            }
            else {
                outputDims[if (keepDims) i else j] = tensor.shape[i]
                if (!keepDims)
                    j++
            }
        }

        val outputTensor = Tensor(DoubleArray(outputDims.fold(1) { acc, dim -> acc * dim }){ Double.NEGATIVE_INFINITY}, outputDims,tensor.requiresGrad)
        val indices = IntArray(tensor.shape.size){0}
        val outputIndices= IntArray(outputDims.size)
        val inputDims = tensor.shape
        val maxIndices = IntArray(outputTensor.size)
        fun recursiveMax(currentDim :Int , outputDim : Int): Unit{
            if(currentDim == inputDims.size - 1){
                for(i in 0 until inputDims[currentDim]){
                    indices[currentDim] = i
                    if(!reduced[currentDim])
                        outputIndices[outputDim] = i
                    val inputValue = tensor.get(*indices)
                    val outputValue = outputTensor.get(*outputIndices)
                    if(inputValue > outputValue){
                        outputTensor.set(*outputIndices,value = inputValue)
                        maxIndices[outputTensor.getFlatIndex(*outputIndices)] = tensor.getFlatIndex(*indices)
                    }
                }
            }
            else {
                for(i in 0 until inputDims[currentDim]){
                    indices[currentDim] = i
                    if(!reduced[currentDim])
                        outputIndices[outputDim] = i
                    recursiveMax(currentDim+1, if(!reduced[currentDim] || keepDims) outputDim+ 1 else outputDim)
                }
            }
        }
        recursiveMax(0,0)
        outputTensor.prevTensors = arrayOf(tensor)
        backward(tensor,outputTensor , maxIndices)
        return outputTensor
    }

    override fun backward(vararg tensors: Tensor) {
        TODO("Not yet implemented")
    }
    private fun backward(inputTensor: Tensor , outputTensor : Tensor , maxIndices: IntArray){
        if (inputTensor.grad == null && inputTensor.requiresGrad)
            inputTensor.grad = Tensor(DoubleArray(inputTensor.size), inputTensor.shape, false, inputTensor.strides)

        outputTensor.gradFn = {
            accumulateAtIndices(inputTensor.grad!!, outputTensor.grad!!,maxIndices)
        }
    }
    operator fun invoke(tensor: Tensor, axes: IntArray, keepDims: Boolean = false): Tensor = forward(FunctionInput.MaxInput(tensor, axes, keepDims))
}