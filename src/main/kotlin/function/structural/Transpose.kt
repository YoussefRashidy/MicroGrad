package io.github.youssefrashidy.function.operators.unaryoperators

import io.github.youssefrashidy.function.BackwardFunction
import io.github.youssefrashidy.function.Function
import io.github.youssefrashidy.tensor.Tensor

class Transpose : Function() {
    override fun forward(vararg tensors: Tensor): Tensor {
        require(tensors.size == 1){
            "Transpose operation can be applied to a single Tensor."
        }
        val tensor = tensors[0]
        require(tensor.rank <= 2){
            "Transpose operation can't be applied to higher order tensors"
        }

        val outputTensor = transpose2D(tensor)
        outputTensor.prevTensors = arrayOf(tensor)
        backward(tensor, outputTensor)
        return outputTensor
    }

    override fun backward(vararg tensors: Tensor) {
        val (a,at) = tensors
        if(a.grad == null)
            a.grad = Tensor(DoubleArray(a.size){0.0},a.shape,false,a.strides)
        at.gradFn = {
            if (a.requiresGrad)
                accumulateAddition(a.grad!!,transpose2D(at.grad!!))
        } as BackwardFunction?
    }

    fun transpose2D(tensor: Tensor): Tensor{
        require(tensor.rank == 2) {
            "2D transpose requires a rank 2 tensor, but got rank ${tensor.rank}"
        }

        val transposedShape : IntArray = IntArray(2)
        val transposedStride: IntArray = IntArray(2)
        transposedShape[0] = tensor.shape[1]
        transposedShape[1] = tensor.shape[0]
        transposedStride[0] = tensor.strides[1]
        transposedShape[1] = tensor.strides[0]
        return Tensor(tensor.backedArray, transposedShape,tensor.requiresGrad,transposedStride)
    }
}