package io.github.youssefrashidy

open class Tensor(val backedArray: DoubleArray , val shape : IntArray , val requiresGrad: Boolean = true ,val strides : IntArray = computeStrides() ) {
    val rank : Int get() = shape.size
    val size : Int get() = backedArray.size
    var grad: Double = 0
    var grad_fn : BackwardFunction? = null
    val prevTensors : Array<Tensor> = emptyArray()

    operator fun get(vararg indices: Int): Double {
        require(indices.size == rank){
            "Expected $rank indices, but got ${indices.size}"
        }

        val index = getFlatIndex(indices)
        return backedArray[index]
    }

    operator fun set(value: Double, vararg indices: Int ) {
        require(indices.size == rank){
            "Expected $rank indices, but got ${indices.size}"
        }
        val index = getFlatIndex(indices)
        backedArray[index] = value
    }

    fun getFlatIndex(vararg indices: Int): Int {
        var index = 0 ;
        for(dim in 0 until rank){
            require(indices[dim] in 0 until shape[dim]){
                "Index ${indices[dim]} out of bounds for axis $dim with size ${shape[dim]}"
            }
            index+= indices[dim] * strides[dim]
        }
        return index
    }

    fun broadcast(a : Tensor , b : Tensor ): Triple<Tensor , Tensor , IntArray> {
        val targetRank = max(a.rank, b.rank)
        val targetShape = IntArray(targetRank)
        aPadding = targetRank - a.rank
        bPadding = targetRank - b.rank
        for(i in targetRank -1 downTo 0){
            val aDim = if(i < aPadding) 1 else a.shape[aPadding - i]
            val bDim = if(i < bPadding) 1 else b.shape[bPadding - i]
            if(aDim == bDim)
                targetShape[i] = aDim
            else if (aDim == 1)
                targetShape[i] = bDim
            else if(bDim == 1)
                targetShape[i] = aDim
            else {
                throw IllegalArgumentException(
                    "Operands could not be broadcast together with shapes " +
                            "${a.shape.contentToString()} and ${b.shape.contentToString()}"
                )
            }
        }
    }

    fun broadcastTo(targetShape: IntArray) {
        if(this.shape.contentEquals(targetShape))
            return this
        require(targetShape.size >= rank) {
            "Cannot broadcast shape ${shape.contentToString()} to lower rank shape ${targetShape.contentToString()}"
        }
        val viewStrides = IntArray(targetShape.size)
        val rankDiff = targetShape.size - rank
        for (i in targetShape.size - 1 downTo 0) {
            val dim = if(i < rankDiff) 1 else this.shape[i - rankDiff]
            val stride = if(i < rankDiff) 0 else this.strides[i - rankDiff]
            if(dim == targetShape[i])
                viewStrides[i] = stride
            else if (dim == 1)
                viewStrides[i] = 0
            else
                throw IllegalArgumentException("Cannot broadcast dimension $dim to ${targetShape[i]} at axis $i\"")
        }
        return Tensor(this.backedArray,targetShape,this.requiresGrad,viewStrides)
    }

    companion object {
        private fun computeStrides() {
            val strides = IntArray(shape.size)
            var currentStride = 1
            for (i in shape.size - 1 downTo 0) {
                strides[i] = currentStride
                currentStride *= shape[i]
            }
            return strides
        }
    }
}