package io.github.youssefrashidy.function.operators

import io.github.youssefrashidy.function.BackwardFunction
import io.github.youssefrashidy.function.Function
import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.tensor.Tensor

class Multiplication : Function() {
    override fun forward(input: FunctionInput): Tensor {
        require(input is FunctionInput.Tensors){

        }
        val tensors = input.tensors
        require(tensors.size == 2){
            "Binary Multiplication requires two parameters of two tensor but got ${tensors.size}"
        }
        val (aBroadcasted,bBroadcasted , targetShape) = Tensor.broadcast(tensors[0], tensors[1])
        val outputTensor:Tensor = Tensor(DoubleArray(targetShape.reduce { acc, dim -> acc*dim }){0.0} , targetShape)
        val indices: IntArray = IntArray(targetShape.size){0}

        fun recursiveMultiplication (currentDim: Int, indices: IntArray) {
            if(currentDim == targetShape.size-1){
                for (i in 0 until targetShape[currentDim]){
                    indices[currentDim] = i
                    outputTensor.set(*indices,value = aBroadcasted.get(*indices) * bBroadcasted.get(*indices))
                }
            }
            else
                for(i in 0 until targetShape[currentDim]){
                    indices[currentDim] = i
                    recursiveMultiplication(currentDim+1, indices)
                }
        }
        recursiveMultiplication(0,indices)

        return outputTensor
    }

    override fun backward(vararg tensors: Tensor) {
        require(tensors.size == 5){
            "Binary Multiplication backward propagation requires three parameters of three tensor but got ${tensors.size}"
        }
        val (a,b,aBroadcasted,bBroadcasted,outputTensor) = tensors
        if(a.grad == null)
            a.grad = Tensor(DoubleArray(a.size){0.0},a.shape,false,a.strides)
        if(b.grad == null)
            b.grad = Tensor(DoubleArray(b.size){0.0},b.shape,false,b.strides)

        outputTensor.gradFn = {
            if(a.requiresGrad){
                val aGradBroadcasted = Tensor(a.grad!!.backedArray ,aBroadcasted.shape,false,aBroadcasted.strides)
                accumulateMultiplication(aGradBroadcasted, bBroadcasted,outputTensor.grad!!)
            }

            if(b.requiresGrad){
                val bGradBroadcasted = Tensor(b.grad!!.backedArray,bBroadcasted.shape,false,bBroadcasted.strides)
                accumulateMultiplication(bGradBroadcasted, aBroadcasted,outputTensor.grad!!)
            }
        } as BackwardFunction?
    }
}