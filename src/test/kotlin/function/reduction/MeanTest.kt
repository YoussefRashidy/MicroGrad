package io.github.youssefrashidy.function.reduction

import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.tensor.Tensor
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class MeanTest {

    @Test
    fun `mean over one axis`() {
        val input = Tensor(
            doubleArrayOf(
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
            ),
            intArrayOf(2, 3)
        )

        val output = Mean.forward(
            FunctionInput.MeanInput(
                input,
                intArrayOf(1),
                false
            )
        )

        assertContentEquals(
            doubleArrayOf(2.0, 5.0),
            output.backedArray
        )

        assertContentEquals(
            intArrayOf(2),
            output.shape
        )
    }

    @Test
    fun `mean with keepDims`() {
        val input = Tensor(
            doubleArrayOf(
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
            ),
            intArrayOf(2, 3)
        )

        val output = Mean.forward(
            FunctionInput.MeanInput(
                input,
                intArrayOf(1),
                true
            )
        )

        assertContentEquals(
            doubleArrayOf(2.0, 5.0),
            output.backedArray
        )

        assertContentEquals(
            intArrayOf(2, 1),
            output.shape
        )
    }

    @Test
    fun `mean over multiple axes`() {
        val input = Tensor(
            doubleArrayOf(
                1.0, 2.0,
                3.0, 4.0,

                5.0, 6.0,
                7.0, 8.0
            ),
            intArrayOf(2, 2, 2)
        )

        val output = Mean.forward(
            FunctionInput.MeanInput(
                input,
                intArrayOf(1, 2),
                false
            )
        )

        // First 2x2:
        // (1 + 2 + 3 + 4) / 4 = 2.5
        //
        // Second 2x2:
        // (5 + 6 + 7 + 8) / 4 = 6.5

        assertContentEquals(
            doubleArrayOf(2.5, 6.5),
            output.backedArray
        )

        assertContentEquals(
            intArrayOf(2),
            output.shape
        )
    }

    @Test
    fun `backward distributes gradient equally across reduced elements`() {
        val input = Tensor(
            doubleArrayOf(
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
            ),
            intArrayOf(2, 3)
        )

        val output = Mean.forward(
            FunctionInput.MeanInput(
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

        // Each row contains 3 elements, so:
        //
        // row 1: 2 / 3
        // row 2: 4 / 3

        assertEquals(2.0 / 3.0, input.grad!!.backedArray[0], 1e-10)
        assertEquals(2.0 / 3.0, input.grad!!.backedArray[1], 1e-10)
        assertEquals(2.0 / 3.0, input.grad!!.backedArray[2], 1e-10)

        assertEquals(4.0 / 3.0, input.grad!!.backedArray[3], 1e-10)
        assertEquals(4.0 / 3.0, input.grad!!.backedArray[4], 1e-10)
        assertEquals(4.0 / 3.0, input.grad!!.backedArray[5], 1e-10)
    }

    @Test
    fun `backward with keepDims`() {
        val input = Tensor(
            doubleArrayOf(
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
            ),
            intArrayOf(2, 3)
        )

        val output = Mean.forward(
            FunctionInput.MeanInput(
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

        assertEquals(2.0 / 3.0, input.grad!!.backedArray[0], 1e-10)
        assertEquals(2.0 / 3.0, input.grad!!.backedArray[1], 1e-10)
        assertEquals(2.0 / 3.0, input.grad!!.backedArray[2], 1e-10)

        assertEquals(4.0 / 3.0, input.grad!!.backedArray[3], 1e-10)
        assertEquals(4.0 / 3.0, input.grad!!.backedArray[4], 1e-10)
        assertEquals(4.0 / 3.0, input.grad!!.backedArray[5], 1e-10)
    }

    @Test
    fun `mean over all dimensions produces scalar`() {
        val input = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0, 4.0),
            intArrayOf(2, 2)
        )

        val output = Mean.forward(
            FunctionInput.MeanInput(
                input,
                intArrayOf(0, 1),
                false
            )
        )

        assertEquals(2.5, output.backedArray[0], 1e-10)

        // Your current representation keeps a rank-0 tensor
        // with an empty shape when all dimensions are reduced.
        assertContentEquals(
            intArrayOf(),
            output.shape
        )
    }
}