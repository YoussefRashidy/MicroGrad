package io.github.youssefrashidy.function.reduction

import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.tensor.Tensor
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class MinTest {

    @Test
    fun `min over one axis`() {
        val input = Tensor(
            doubleArrayOf(
                3.0, 1.0, 2.0,
                6.0, 4.0, 5.0
            ),
            intArrayOf(2, 3)
        )

        val output = Min.forward(
            FunctionInput.MinInput(
                input,
                intArrayOf(1),
                false
            )
        )

        assertContentEquals(
            doubleArrayOf(1.0, 4.0),
            output.backedArray
        )

        assertContentEquals(
            intArrayOf(2),
            output.shape
        )
    }

    @Test
    fun `min with keepDims`() {
        val input = Tensor(
            doubleArrayOf(
                3.0, 1.0, 2.0,
                6.0, 4.0, 5.0
            ),
            intArrayOf(2, 3)
        )

        val output = Min.forward(
            FunctionInput.MinInput(
                input,
                intArrayOf(1),
                true
            )
        )

        assertContentEquals(
            doubleArrayOf(1.0, 4.0),
            output.backedArray
        )

        assertContentEquals(
            intArrayOf(2, 1),
            output.shape
        )
    }

    @Test
    fun `min over multiple axes`() {
        val input = Tensor(
            doubleArrayOf(
                4.0, 2.0,
                8.0, 3.0,

                7.0, 6.0,
                5.0, 1.0
            ),
            intArrayOf(2, 2, 2)
        )

        val output = Min.forward(
            FunctionInput.MinInput(
                input,
                intArrayOf(1, 2),
                false
            )
        )

        // First 2x2 -> min = 2
        // Second 2x2 -> min = 1

        assertContentEquals(
            doubleArrayOf(2.0, 1.0),
            output.backedArray
        )

        assertContentEquals(
            intArrayOf(2),
            output.shape
        )
    }

    @Test
    fun `backward routes gradient only to minimum elements`() {
        val input = Tensor(
            doubleArrayOf(
                3.0, 1.0, 2.0,
                6.0, 4.0, 5.0
            ),
            intArrayOf(2, 3)
        )

        val output = Min.forward(
            FunctionInput.MinInput(
                input,
                intArrayOf(1),
                false
            )
        )

        // dL/dz = [2, 4]
        output.grad = Tensor(
            doubleArrayOf(2.0, 4.0),
            intArrayOf(2),
            false
        )

        output.gradFn!!.apply()

        // First row minimum is at index 1.
        // Second row minimum is at index 4.

        assertContentEquals(
            doubleArrayOf(
                0.0, 2.0, 0.0,
                0.0, 4.0, 0.0
            ),
            input.grad!!.backedArray
        )
    }

    @Test
    fun `backward with keepDims routes gradient correctly`() {
        val input = Tensor(
            doubleArrayOf(
                3.0, 1.0, 2.0,
                6.0, 4.0, 5.0
            ),
            intArrayOf(2, 3)
        )

        val output = Min.forward(
            FunctionInput.MinInput(
                input,
                intArrayOf(1),
                true
            )
        )

        output.grad = Tensor(
            doubleArrayOf(2.0, 4.0),
            intArrayOf(2, 1),
            false
        )

        output.gradFn!!.apply()

        assertContentEquals(
            doubleArrayOf(
                0.0, 2.0, 0.0,
                0.0, 4.0, 0.0
            ),
            input.grad!!.backedArray
        )
    }

    @Test
    fun `min over all dimensions produces scalar`() {
        val input = Tensor(
            doubleArrayOf(
                4.0, 2.0,
                3.0, 1.0
            ),
            intArrayOf(2, 2)
        )

        val output = Min.forward(
            FunctionInput.MinInput(
                input,
                intArrayOf(0, 1),
                false
            )
        )

        assertEquals(1.0, output.backedArray[0])

        assertContentEquals(
            intArrayOf(),
            output.shape
        )
    }

    @Test
    fun `does not create gradient when requiresGrad is false`() {
        val input = Tensor(
            doubleArrayOf(
                3.0, 1.0, 2.0,
                6.0, 4.0, 5.0
            ),
            intArrayOf(2, 3),
            requiresGrad = false
        )

        val output = Min.forward(
            FunctionInput.MinInput(
                input,
                intArrayOf(1),
                false
            )
        )

        assertEquals(null, input.grad)
    }
}