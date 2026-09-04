package io.github.youssefrashidy.function

import io.github.youssefrashidy.tensor.Tensor

interface FunctionInput {
    data class Tensors(val tensors: Array<Tensor>) : FunctionInput{}
    data class ReshapeInput(val tensors: Array<Tensor> , val shape : IntArray) : FunctionInput{}
}