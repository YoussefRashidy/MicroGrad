package io.github.youssefrashidy

import io.github.youssefrashidy.tensor.Tensor
import kotlin.random.Random

object MicroGrad {
    var gradEnabled : Boolean = true

    fun noGrad(block : ()->Unit){
        val previousGrad = gradEnabled
        gradEnabled = false
        try {
            block()
        } finally {
            gradEnabled = previousGrad
        }
    }

    private class Stack<T> {
        private val deque = ArrayDeque<T>()

        fun push(item: T) = deque.addLast(item)
        fun pop(): T = deque.removeLast()
        fun peek(): T = deque.last()
        fun isEmpty(): Boolean = deque.isEmpty()
        val size: Int get() = deque.size
    }

    fun backward(loss : Tensor , retainGraph: Boolean): Unit{
        require(loss.size == 1) {
            "backward() can only be called on a scalar loss"
        }

        val stack = Stack<Tensor>()
        val visited = mutableSetOf<Tensor>()
        fun dfsTopoSort(node : Tensor): Unit{
            visited.add(node)
            for(tensor in node.prevTensors){
                if(!visited.contains(tensor)){
                    dfsTopoSort(tensor)
                }
            }
            stack.push(node)
        }
        loss.grad = scalar(1.0,false)
        dfsTopoSort(loss)
        while (!stack.isEmpty()){
            val tensor = stack.pop()
            tensor.gradFn?.apply()
            if(!retainGraph){
                tensor.prevTensors = emptyArray()
                tensor.gradFn = null
            }
        }
    }

    fun tensor(data: DoubleArray , shape : IntArray , requiresGrad : Boolean = gradEnabled ) = Tensor(
        data,
        shape,
        requiresGrad
    )
    fun scalar(value: Double, requiresGrad: Boolean = gradEnabled)= tensor(
            doubleArrayOf(value),
            intArrayOf(1),
            requiresGrad
        )
    fun zeros(shape: IntArray, requiresGrad: Boolean = gradEnabled) = tensor(
            DoubleArray(shape.fold(1) { acc, dim -> acc * dim }),
            shape,
            requiresGrad
        )
    fun ones(shape: IntArray, requiresGrad: Boolean = gradEnabled) = tensor(
        DoubleArray(shape.fold(1) { acc, dim -> acc * dim }){1.0},
        shape,
        requiresGrad
    )
    fun fill(shape: IntArray,value: Double ,requiresGrad: Boolean = gradEnabled) = tensor(
        DoubleArray(shape.fold(1) { acc, dim -> acc * dim }){value},
        shape,
        requiresGrad
    )

    fun rand(shape: IntArray, requiresGrad: Boolean = gradEnabled) = tensor(
        DoubleArray(shape.fold(1) { acc, dim -> acc * dim }) { Random.nextDouble() },
        shape,
        requiresGrad
    )

    fun randn(shape: IntArray, requiresGrad: Boolean = gradEnabled): Tensor {
        val size = shape.fold(1) { acc, dim -> acc * dim }
        val data = DoubleArray(size)

        var i = 0

        while (i < size) {
            val u1 = Random.nextDouble()
            val u2 = Random.nextDouble()

            val radius = kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1))
            val theta = 2.0 * kotlin.math.PI * u2

            data[i] = radius * kotlin.math.cos(theta)

            if (i + 1 < size) {
                data[i + 1] = radius * kotlin.math.sin(theta)
            }

            i += 2
        }

        return tensor(data, shape, requiresGrad)
    }

}