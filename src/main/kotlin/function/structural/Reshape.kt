package io.github.youssefrashidy.function.structural

import io.github.youssefrashidy.function.Function
import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.tensor.Tensor

object Reshape: Function() {
    override fun forward(input: FunctionInput): Tensor {
        require(input is FunctionInput.ReshapeInput){
            "Expected FunctionInput.ReshapeInput, but got ${input::class.simpleName}"
        }
        val tensors = input.tensors
        val shape = input.shape
        require(tensors.size == 1){
            "Tensor array sizes should be 1, but ${tensors.size}"
        }
        val output = reshape(tensors[0], shape)
        output.prevTensors = arrayOf(tensors[0])
        backward(tensors[0], output)
        return output
    }

    override fun backward(vararg tensors: Tensor ) {
        val inputTensor = tensors[0]
        val outputTensor = tensors[1]
        if(inputTensor.grad == null && inputTensor.requiresGrad)
            inputTensor.grad = Tensor(DoubleArray(inputTensor.size){0.0},inputTensor.shape,false,inputTensor.strides)
        if (inputTensor.requiresGrad){
            outputTensor.gradFn = {
                 accumulateAddition(inputTensor.grad!!,reshape(outputTensor.grad!!,inputTensor.shape))
            }
        }
    }

    internal fun reshape(tensor: Tensor , shape : IntArray): Tensor {
        require(shape.fold(1){ acc, i ->  acc*i} == tensor.size){
            ""
        }
        return Tensor(tensor.backedArray,shape,tensor.requiresGrad)
    }
    operator fun invoke(tensor: Tensor, shape: IntArray): Tensor = forward(FunctionInput.ReshapeInput(arrayOf(tensor), shape))
    operator fun invoke(tensor: Tensor, vararg shape: Int): Tensor = forward(FunctionInput.ReshapeInput(arrayOf(tensor), shape))
}