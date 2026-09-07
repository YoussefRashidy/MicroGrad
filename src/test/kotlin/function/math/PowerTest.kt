package io.github.youssefrashidy.function.math

import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.tensor.Tensor
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class PowerTest {

    @Test
    fun `computes power elementwise`() {
        val input = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0, 4.0),
            intArrayOf(4)
        )

        val output = Power.forward(
            FunctionInput.PowerInput(input, 2.0)
        )

        assertContentEquals(
            doubleArrayOf(1.0, 4.0, 9.0, 16.0),
            output.backedArray
        )
    }

    @Test
    fun `preserves shape`() {
        val input = Tensor(
            doubleArrayOf(
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
            ),
            intArrayOf(2, 3)
        )

        val output = Power.forward(
            FunctionInput.PowerInput(input, 2.0)
        )

        assertContentEquals(
            intArrayOf(2, 3),
            output.shape
        )
    }

    @Test
    fun `supports non integer power`() {
        val input = Tensor(
            doubleArrayOf(1.0, 4.0, 9.0, 16.0),
            intArrayOf(4)
        )

        val output = Power.forward(
            FunctionInput.PowerInput(input, 0.5)
        )

        assertEquals(1.0, output.backedArray[0], 1e-10)
        assertEquals(2.0, output.backedArray[1], 1e-10)
        assertEquals(3.0, output.backedArray[2], 1e-10)
        assertEquals(4.0, output.backedArray[3], 1e-10)
    }

    @Test
    fun `backward computes correct gradient`() {
        val input = Tensor(
            doubleArrayOf(2.0, 3.0, 4.0),
            intArrayOf(3)
        )

        val output = Power.forward(
            FunctionInput.PowerInput(input, 3.0)
        )

        // dL/dz
        output.grad = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            false
        )

        output.gradFn!!.apply()

        // dL/dx = dL/dz * p * x^(p-1)
        //
        // [1 * 3 * 2²,
        //  2 * 3 * 3²,
        //  3 * 3 * 4²]
        //
        // = [12, 54, 144]

        assertContentEquals(
            doubleArrayOf(12.0, 54.0, 144.0),
            input.grad!!.backedArray
        )
    }

    @Test
    fun `backward works with fractional power`() {
        val input = Tensor(
            doubleArrayOf(4.0, 9.0, 16.0),
            intArrayOf(3)
        )

        val output = Power.forward(
            FunctionInput.PowerInput(input, 0.5)
        )

        output.grad = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            false
        )

        output.gradFn!!.apply()

        // dL/dx = dL/dz * 0.5 * x^(-0.5)
        //
        // [1 * 0.5 / 2,
        //  2 * 0.5 / 3,
        //  3 * 0.5 / 4]
        //
        // = [0.25, 1/3, 0.375]

        assertEquals(
            0.25,
            input.grad!!.backedArray[0],
            1e-10
        )

        assertEquals(
            1.0 / 3.0,
            input.grad!!.backedArray[1],
            1e-10
        )

        assertEquals(
            0.375,
            input.grad!!.backedArray[2],
            1e-10
        )
    }

    @Test
    fun `does not create gradient when requiresGrad is false`() {
        val input = Tensor(
            doubleArrayOf(2.0, 3.0, 4.0),
            intArrayOf(3),
            requiresGrad = false
        )

        val output = Power.forward(
            FunctionInput.PowerInput(input, 2.0)
        )

        assertEquals(null, output.gradFn)
        assertEquals(null, input.grad)
    }
}