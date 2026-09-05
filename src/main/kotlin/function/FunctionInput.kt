package io.github.youssefrashidy.function

import io.github.youssefrashidy.tensor.Tensor

sealed interface FunctionInput {
    data class Tensors(val tensors: Array<Tensor>) : FunctionInput{}
    data class ReshapeInput(val tensors: Array<Tensor> , val shape : IntArray) : FunctionInput{}
    data class PermuteInput(val tensors: Array<Tensor>, val permute: IntArray) : FunctionInput{}
    data class PowerInput(val tensor: Tensor , val power : Double): FunctionInput{}
    data class SumInput(val tensor: Tensor , val axes : IntArray , val keepDims : Boolean): FunctionInput{}
    data class MeanInput(val tensor: Tensor , val axes : IntArray , val keepDims : Boolean): FunctionInput{}
    data class MaxInput(val tensor: Tensor , val axes : IntArray , val keepDims : Boolean): FunctionInput{}
    data class MinInput(val tensor: Tensor , val axes : IntArray , val keepDims : Boolean): FunctionInput{}
}