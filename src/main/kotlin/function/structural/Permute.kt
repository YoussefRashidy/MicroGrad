package io.github.youssefrashidy.function.structural

import io.github.youssefrashidy.function.Function
import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.tensor.Tensor

object Permute : Function() {
    override fun forward(input: FunctionInput): Tensor {
        require(input is FunctionInput.PermuteInput){

        }
        val inputTensor = input.tensors
        val permutation = input.permute
        require(permutation.size == inputTensor.rank){
            "Permutation must have ${inputTensor.shape.size} dimensions, but got ${permutation.size}"
        }
        require(permutation.toSet().size == permutation.size && permutation.all { it in inputTensor.shape.indices }) {
            "Invalid permutation: ${permutation.contentToString()}"
        }
        val outputTensor = permute(inputTensor,permutation)
        outputTensor.prevTensors = arrayOf(inputTensor)
        backward(inputTensor, outputTensor,permutation = permutation)
        return outputTensor
    }

    // I will address this later
    override fun backward(vararg tensors: Tensor) {
        val inputTensor = tensors[0]
        val outputTensor = tensors[1]
        if(inputTensor.grad== null)
            inputTensor.grad = Tensor(DoubleArray(inputTensor.size),inputTensor.shape,false,inputTensor.strides)
        if(inputTensor.requiresGrad){
            outputTensor.gradFn = {

            }
        }
    }

    private fun backward(vararg tensors: Tensor , permutation: IntArray) {
        val inputTensor = tensors[0]
        val outputTensor = tensors[1]
        val inversePermute = inversePermutation(permutation)
        if(inputTensor.grad== null)
            inputTensor.grad = Tensor(DoubleArray(inputTensor.size),inputTensor.shape,false,inputTensor.strides)
        if(inputTensor.requiresGrad){
            outputTensor.gradFn = {
                accumulateAddition(inputTensor.grad!!,permute(outputTensor.grad!!,inversePermute))
            }
        }
    }

    private fun permute(tensor: Tensor , permutation : IntArray) : Tensor{
        val newShape = IntArray(tensor.rank)
        val newStride = IntArray(tensor.rank)
        for(i in permutation.indices){
            newShape[i] = tensor.shape[permutation[i]]
            newStride[i] = tensor.strides[permutation[i]]
        }
        val outputTensor = Tensor(tensor.backedArray,newShape,tensor.requiresGrad,newStride)
        return outputTensor
    }

    private fun inversePermutation(permutation: IntArray): IntArray {
        val inverse = IntArray(permutation.size)

        for (i in permutation.indices) {
            inverse[permutation[i]] = i
        }

        return inverse
    }
}