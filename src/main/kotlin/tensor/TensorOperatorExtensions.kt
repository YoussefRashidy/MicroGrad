package io.github.youssefrashidy.tensor

import io.github.youssefrashidy.MicroGrad

// Int operators
operator fun Int.plus(other: Tensor): Tensor =
    MicroGrad.scalar(this.toDouble()) + other

operator fun Int.minus(other: Tensor): Tensor =
    MicroGrad.scalar(this.toDouble()) - other

operator fun Int.times(other: Tensor): Tensor =
    MicroGrad.scalar(this.toDouble()) * other

operator fun Int.div(other: Tensor): Tensor =
    MicroGrad.scalar(this.toDouble()) / other

// Long operators
operator fun Long.plus(other: Tensor): Tensor =
    MicroGrad.scalar(this.toDouble()) + other

operator fun Long.minus(other: Tensor): Tensor =
    MicroGrad.scalar(this.toDouble()) - other

operator fun Long.times(other: Tensor): Tensor =
    MicroGrad.scalar(this.toDouble()) * other

operator fun Long.div(other: Tensor): Tensor =
    MicroGrad.scalar(this.toDouble()) / other

// Float operators
operator fun Float.plus(other: Tensor): Tensor =
    MicroGrad.scalar(this.toDouble()) + other

operator fun Float.minus(other: Tensor): Tensor =
    MicroGrad.scalar(this.toDouble()) - other

operator fun Float.times(other: Tensor): Tensor =
    MicroGrad.scalar(this.toDouble()) * other

operator fun Float.div(other: Tensor): Tensor =
    MicroGrad.scalar(this.toDouble()) / other

// Double operators
operator fun Double.plus(other: Tensor): Tensor =
    MicroGrad.scalar(this) + other

operator fun Double.minus(other: Tensor): Tensor =
    MicroGrad.scalar(this) - other

operator fun Double.times(other: Tensor): Tensor =
    MicroGrad.scalar(this) * other

operator fun Double.div(other: Tensor): Tensor =
    MicroGrad.scalar(this) / other