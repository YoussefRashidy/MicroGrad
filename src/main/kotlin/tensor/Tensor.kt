package io.github.youssefrashidy.tensor

import io.github.youssefrashidy.MicroGrad
import io.github.youssefrashidy.function.BackwardFunction
import io.github.youssefrashidy.function.operators.Addition
import io.github.youssefrashidy.function.operators.Division
import io.github.youssefrashidy.function.operators.Multiplication
import io.github.youssefrashidy.function.operators.Subtraction
import kotlin.math.max

class Tensor internal constructor(
    val backedArray: DoubleArray,
    val shape: IntArray,
    val requiresGrad: Boolean = true,
    val strides: IntArray = computeStrides(shape)
) {
    val rank: Int get() = shape.size
    val size: Int get() = backedArray.size
    var grad: Tensor? = null
    var gradFn: BackwardFunction? = null
    var prevTensors: Array<Tensor> = emptyArray()

    operator fun get(vararg indices: Int): Double {
        require(indices.size == rank) {
            "Expected $rank indices, but got ${indices.size}"
        }

        val index = getFlatIndex(*indices)
        return backedArray[index]
    }


    operator fun set(vararg indices: Int, value: Double) {
        require(indices.size == rank) {
            "Expected $rank indices, but got ${indices.size}"
        }
        val index = getFlatIndex(*indices)
        backedArray[index] = value
    }


    fun getFlatIndex(vararg indices: Int): Int {
        var index = 0;
        for (dim in 0 until rank) {
            require(indices[dim] in 0 until shape[dim]) {
                "Index ${indices[dim]} out of bounds for axis $dim with size ${shape[dim]}"
            }
            index += indices[dim] * strides[dim]
        }
        return index
    }

    companion object {
        fun computeStrides(shape: IntArray): IntArray {
            val strides = IntArray(shape.size)
            var currentStride = 1
            for (i in shape.size - 1 downTo 0) {
                strides[i] = currentStride
                currentStride *= shape[i]
            }
            return strides
        }

        fun broadcast(a: Tensor, b: Tensor): Triple<Tensor, Tensor, IntArray> {
            val targetShape = broadcastShape(a.shape, b.shape)
            return Triple(a.broadcastTo(targetShape), b.broadcastTo(targetShape), targetShape)
        }

        fun broadcastShape(aShape: IntArray, bShape: IntArray): IntArray {
            val targetRank = max(aShape.size, bShape.size)
            val targetShape = IntArray(targetRank)

            val aPadding = targetRank - aShape.size
            val bPadding = targetRank - bShape.size

            for (i in targetRank - 1 downTo 0) {
                val aDim = if (i < aPadding) 1 else aShape[i - aPadding]
                val bDim = if (i < bPadding) 1 else bShape[i - bPadding]

                targetShape[i] = when {
                    aDim == bDim -> aDim
                    aDim == 1 -> bDim
                    bDim == 1 -> aDim
                    else -> throw IllegalArgumentException(
                        "Operands could not be broadcast together with shapes " +
                                "${aShape.contentToString()} and ${bShape.contentToString()}"
                    )
                }
            }

            return targetShape
        }
    }

    fun broadcastTo(targetShape: IntArray): Tensor {
        if (this.shape.contentEquals(targetShape))
            return this
        require(targetShape.size >= rank) {
            "Cannot broadcast shape ${shape.contentToString()} to lower rank shape ${targetShape.contentToString()}"
        }
        val viewStrides = IntArray(targetShape.size)
        val rankDiff = targetShape.size - rank
        for (i in targetShape.size - 1 downTo 0) {
            val dim = if (i < rankDiff) 1 else this.shape[i - rankDiff]
            val stride = if (i < rankDiff) 0 else this.strides[i - rankDiff]
            when (dim) {
                targetShape[i] -> viewStrides[i] = stride
                1 -> viewStrides[i] = 0
                else -> throw IllegalArgumentException("Cannot broadcast dimension $dim to ${targetShape[i]} at axis $i")
            }
        }
        return Tensor(this.backedArray, targetShape, this.requiresGrad, viewStrides)
    }

    fun backward(retainGraph : Boolean = false) = MicroGrad.backward(this, retainGraph)

    override fun toString(): String {
        val builder = StringBuilder()

        builder.append("Tensor(shape=")
            .append(shape.contentToString())
            .append(", data=")

        fun appendDimension(currentDim: Int, indices: IntArray) {
            if (currentDim == rank) {
                builder.append(get(*indices))
                return
            }
            builder.append("[")
            for (i in 0 until shape[currentDim]) {
                indices[currentDim] = i
                if (i > 0)
                    builder.append(", ")
                appendDimension(currentDim + 1, indices)
            }
            builder.append("]")
        }
        appendDimension(0, IntArray(rank))
        builder.append(")")

        return builder.toString()
    }

    operator fun plus(other: Tensor) = Addition(this, other)
    operator fun plus(other: Double): Tensor =
        this + MicroGrad.scalar(other)

    operator fun plus(other: Float): Tensor =
        this + MicroGrad.scalar(other.toDouble())

    operator fun plus(other: Int): Tensor =
        this + MicroGrad.scalar(other.toDouble())

    operator fun plus(other: Long): Tensor =
        this + MicroGrad.scalar(other.toDouble())


    operator fun minus(other: Tensor) = Subtraction(this, other)
    operator fun minus(other: Double): Tensor =
        this - MicroGrad.scalar(other)

    operator fun minus(other: Float): Tensor =
        this - MicroGrad.scalar(other.toDouble())

    operator fun minus(other: Int): Tensor =
        this - MicroGrad.scalar(other.toDouble())

    operator fun minus(other: Long): Tensor =
        this - MicroGrad.scalar(other.toDouble())

    operator fun times(other: Tensor) = Multiplication(this, other)
    operator fun times(other: Double): Tensor =
        this * MicroGrad.scalar(other)

    operator fun times(other: Float): Tensor =
        this * MicroGrad.scalar(other.toDouble())

    operator fun times(other: Int): Tensor =
        this * MicroGrad.scalar(other.toDouble())

    operator fun times(other: Long): Tensor =
        this * MicroGrad.scalar(other.toDouble())

    operator fun div(other: Tensor) = Division(this, other)

    operator fun div(other: Double): Tensor =
        this / MicroGrad.scalar(other)

    operator fun div(other: Float): Tensor =
        this / MicroGrad.scalar(other.toDouble())

    operator fun div(other: Int): Tensor =
        this / MicroGrad.scalar(other.toDouble())

    operator fun div(other: Long): Tensor =
        this / MicroGrad.scalar(other.toDouble())

    operator fun unaryMinus(): Tensor = MicroGrad.scalar(-1.0) * this

}