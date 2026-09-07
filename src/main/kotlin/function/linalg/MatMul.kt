package io.github.youssefrashidy.function.linalg

import io.github.youssefrashidy.MicroGrad
import io.github.youssefrashidy.function.Function
import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.function.structural.Permute
import io.github.youssefrashidy.function.structural.Reshape
import io.github.youssefrashidy.tensor.Tensor

object MatMul : Function() {
    override fun forward(input: FunctionInput): Tensor {
        require(input is FunctionInput.MatMulInput) {

        }
        val a = input.tensor1
        val b = input.tensor2

        if (a.rank == 1 && b.rank == 1)
            return dot(a, b)

        var aReshaped: Tensor = a
        if (a.rank == 1) {
            val shape = intArrayOf(1, a.size)
            aReshaped = Reshape.reshape(a, shape)
        }

        var bReshaped: Tensor = b
        if (b.rank == 1) {
            val shape = intArrayOf(b.size, 1)
            bReshaped = Reshape.reshape(b, shape)
        }

        val outputTensor = matrixMultiply(aReshaped, bReshaped)
        if (MicroGrad.gradEnabled) {
            outputTensor.prevTensors = arrayOf(a, b)
            matmulBackward(a, aReshaped, b, bReshaped, outputTensor)
        }
        return outputTensor
    }

    override fun backward(vararg tensors: Tensor) {
        TODO("Not yet implemented")
    }

    private fun dot(a: Tensor, b: Tensor): Tensor {
        require(a.shape.contentEquals(b.shape)) {
            "Input shape contains different contents for tensors ${a.shape.contentToString()} and ${b.shape.contentToString()}"
        }

        var product = 0.0
        for (i in 0 until a.size)
            product += a[i] * b[i]
        val output = Tensor(doubleArrayOf(product), intArrayOf(1), a.requiresGrad || b.requiresGrad)
        if (MicroGrad.gradEnabled) {
            output.prevTensors = arrayOf(a, b)
            dotBackward(a, b, output)
        }

        return output
    }

    private fun dotBackward(a: Tensor, b: Tensor, output: Tensor) {
        if (a.grad == null && a.requiresGrad)
            a.grad = Tensor(DoubleArray(a.size) { 0.0 }, a.shape, false, a.strides)
        if (b.grad == null && b.requiresGrad)
            b.grad = Tensor(DoubleArray(b.size) { 0.0 }, b.shape, false, b.strides)
        output.gradFn = {
            if (a.requiresGrad) {
                accumulateDot(a.grad!!, b, output.grad!!)
            }

            if (b.requiresGrad) {
                accumulateDot(b.grad!!, a, output.grad!!)
            }
        }
    }

    private fun matrixMultiply(a: Tensor, b: Tensor): Tensor {
        require(a.shape.last() == b.shape[b.shape.lastIndex - 1]) {
            ""
        }
        val aBatch = a.shape.copyOfRange(0, a.shape.size - 2)
        val bBatch = b.shape.copyOfRange(0, b.shape.size - 2)
        val batchBroadcastedShape = Tensor.broadcastShape(aBatch, bBatch)
        val n = a.shape[a.shape.lastIndex - 1]
        val p = a.shape[a.rank - 1]
        val m = b.shape.last()
        val targetShape = batchBroadcastedShape + intArrayOf(n, m)
        val aBroadcasted = a.broadcastTo(batchBroadcastedShape + a.shape.copyOfRange(a.shape.size - 2, a.shape.size))
        val bBroadcasted = b.broadcastTo(batchBroadcastedShape + b.shape.copyOfRange(b.shape.size - 2, b.shape.size))
        val outputTensor = Tensor(
            DoubleArray(targetShape.fold(1) { acc, i -> acc * i }) { 0.0 },
            targetShape,
            a.requiresGrad || b.requiresGrad
        )
        val currentBatch = IntArray(batchBroadcastedShape.size) { 0 }
        fun batchRecurse(currentDim: Int) {
            if (currentDim == currentBatch.size)
                matrixMultiply2D(aBroadcasted, bBroadcasted, outputTensor, currentBatch, n, m, p)
            else {
                for (i in 0 until targetShape[currentDim]) {
                    currentBatch[currentDim] = i
                    batchRecurse(currentDim + 1)
                }
            }
        }
        batchRecurse(0)
        return outputTensor
    }

    private fun matrixMultiply2D(a: Tensor, b: Tensor, c: Tensor, batchDims: IntArray, n: Int, m: Int, p: Int) {
        val aIndices = batchDims + IntArray(2) { 0 }
        val bIndices = batchDims + IntArray(2) { 0 }
        val cIndices = batchDims + IntArray(2) { 0 }
        for (i in 0 until n) {
            for (k in 0 until p) {
                aIndices[aIndices.lastIndex - 1] = i
                aIndices[aIndices.lastIndex] = k
                bIndices[bIndices.lastIndex - 1] = k
                cIndices[cIndices.lastIndex - 1] = i
                val a = a.get(*aIndices)

                for (j in 0 until m) {
                    bIndices[bIndices.lastIndex] = j
                    cIndices[cIndices.lastIndex] = j
                    c.set(*cIndices, value = c.get(*cIndices) + a * b.get(*bIndices))
                }
            }
        }
    }

    private fun matmulBackward(a: Tensor, aReshaped: Tensor, b: Tensor, bReshaped: Tensor, output: Tensor) {
        if (a.grad == null)
            a.grad = Tensor(DoubleArray(a.size) { 0.0 }, a.shape, false, a.strides)
        if (b.grad == null)
            b.grad = Tensor(DoubleArray(b.size) { 0.0 }, b.shape, false, b.strides)
        output.gradFn = {
            if (a.requiresGrad) {
                aReshaped.grad = Reshape.reshape(a.grad!!, aReshaped.shape)
                val permutation = IntArray(b.rank) { it }
                permutation[b.rank - 2] = b.rank - 1
                permutation[b.rank - 1] = b.rank - 2

                val permuteInput = FunctionInput.PermuteInput(b, permutation)
                val bPermuted = Permute.forward(permuteInput)
                val chainedGrad = matrixMultiply(output.grad!!, bPermuted)
                accumulateAddition(aReshaped.grad!!, chainedGrad)
            }
            if (b.requiresGrad) {
                bReshaped.grad = Reshape.reshape(b.grad!!, bReshaped.shape)
                val permutation = IntArray(a.rank) { it }
                permutation[b.rank - 2] = a.rank - 1
                permutation[b.rank - 1] = a.rank - 2
                val permuteInput = FunctionInput.PermuteInput(a, permutation)
                val aPermuted = Permute.forward(permuteInput)
                val chainedGrad = matrixMultiply(aPermuted, output.grad!!)
                accumulateAddition(bReshaped.grad!!, chainedGrad)
            }
        }

    }

    operator fun invoke(a: Tensor, b: Tensor): Tensor = forward(FunctionInput.MatMulInput(a, b))

}