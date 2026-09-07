package io.github.youssefrashidy.function.math

import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.tensor.Tensor
import kotlin.math.E
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertContentEquals

class ExpTest {

    @Test
    fun `computes exponential elementwise`() {
        val input = Tensor(
            doubleArrayOf(0.0, 1.0, 2.0, -1.0),
            intArrayOf(4)
        )

        val output = Exp.forward(
            FunctionInput.Tensors(arrayOf(input))
        )

        assertEquals(1.0, output.backedArray[0], 1e-10)
        assertEquals(E, output.backedArray[1], 1e-10)
        assertEquals(exp(2.0), output.backedArray[2], 1e-10)
        assertEquals(exp(-1.0), output.backedArray[3], 1e-10)
    }

    @Test
    fun `preserves shape`() {
        val input = Tensor(
            doubleArrayOf(
                0.0, 1.0, 2.0,
                3.0, 4.0, 5.0
            ),
            intArrayOf(2, 3)
        )

        val output = Exp.forward(
            FunctionInput.Tensors(arrayOf(input))
        )

        assertContentEquals(
            intArrayOf(2, 3),
            output.shape
        )
    }

    @Test
    fun `backward computes correct gradient`() {
        val input = Tensor(
            doubleArrayOf(0.0, 1.0, 2.0),
            intArrayOf(3)
        )

        val output = Exp.forward(
            FunctionInput.Tensors(arrayOf(input))
        )

        // dL/dz
        output.grad = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            false
        )

        output.gradFn!!.apply()

        // dL/dx = dL/dz * exp(x)
        assertEquals(
            1.0,
            input.grad!!.backedArray[0],
            1e-10
        )

        assertEquals(
            2.0 * E,
            input.grad!!.backedArray[1],
            1e-10
        )

        assertEquals(
            3.0 * exp(2.0),
            input.grad!!.backedArray[2],
            1e-10
        )
    }

    @Test
    fun `does not create gradient when requiresGrad is false`() {
        val input = Tensor(
            doubleArrayOf(0.0, 1.0, 2.0),
            intArrayOf(3),
            requiresGrad = false
        )

        val output = Exp.forward(
            FunctionInput.Tensors(arrayOf(input))
        )

        output.grad = Tensor(
            doubleArrayOf(1.0, 1.0, 1.0),
            intArrayOf(3),
            false
        )

        // There should be no gradFn because input does not require grad.
        assertEquals(null, output.gradFn)
        assertEquals(null, input.grad)
    }
}