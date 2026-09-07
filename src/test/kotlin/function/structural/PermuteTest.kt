package io.github.youssefrashidy.function.structural

import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.tensor.Tensor
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class PermuteTest {

    @Test
    fun `permutes dimensions correctly`() {
        val input = Tensor(
            doubleArrayOf(
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
            ),
            intArrayOf(2, 3)
        )

        val output = Permute.forward(
            FunctionInput.PermuteInput(
                input,
                intArrayOf(1, 0)
            )
        )

        assertContentEquals(
            intArrayOf(3, 2),
            output.shape
        )

        assertEquals(1.0, output[0, 0])
        assertEquals(4.0, output[0, 1])
        assertEquals(2.0, output[1, 0])
        assertEquals(5.0, output[1, 1])
        assertEquals(3.0, output[2, 0])
        assertEquals(6.0, output[2, 1])
    }

    @Test
    fun `permute is a view of the original tensor`() {
        val input = Tensor(
            doubleArrayOf(
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
            ),
            intArrayOf(2, 3)
        )

        val output = Permute.forward(
            FunctionInput.PermuteInput(
                input,
                intArrayOf(1, 0)
            )
        )

        // Both tensors share the same backing array.
        output[1, 0] = 100.0

        assertEquals(100.0, input[0, 1])
        assertEquals(100.0, output[1, 0])
    }

    @Test
    fun `permutes three dimensions correctly`() {
        val input = Tensor(
            doubleArrayOf(
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0,
                7.0, 8.0
            ),
            intArrayOf(2, 2, 2)
        )

        val output = Permute.forward(
            FunctionInput.PermuteInput(
                input,
                intArrayOf(2, 0, 1)
            )
        )

        assertContentEquals(
            intArrayOf(2, 2, 2),
            output.shape
        )

        assertEquals(1.0, output[0, 0, 0])
        assertEquals(5.0, output[0, 1, 0])
        assertEquals(3.0, output[0, 0, 1])
        assertEquals(7.0, output[0, 1, 1])

        assertEquals(2.0, output[1, 0, 0])
        assertEquals(6.0, output[1, 1, 0])
        assertEquals(4.0, output[1, 0, 1])
        assertEquals(8.0, output[1, 1, 1])

    }

    @Test
    fun `backward applies inverse permutation`() {
        val input = Tensor(
            doubleArrayOf(
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
            ),
            intArrayOf(2, 3)
        )

        val output = Permute.forward(
            FunctionInput.PermuteInput(
                input,
                intArrayOf(1, 0)
            )
        )

        // dL/doutput
        output.grad = Tensor(
            doubleArrayOf(
                10.0, 20.0,
                30.0, 40.0,
                50.0, 60.0
            ),
            intArrayOf(3, 2),
            false
        )

        output.gradFn!!.apply()

        // Inverse permutation is also [1, 0].
        //
        // [10 20]       [10 30 50]
        // [30 40]  ->   [20 40 60]
        // [50 60]
        assertContentEquals(
            doubleArrayOf(
                10.0, 30.0, 50.0,
                20.0, 40.0, 60.0
            ),
            input.grad!!.backedArray
        )
    }

    @Test
    fun `backward works with non self inverse permutation`() {
        val input = Tensor(
            doubleArrayOf(
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0
            ),
            intArrayOf(2, 2, 3)
        )

        val permutation = intArrayOf(2, 0, 1)

        val output = Permute.forward(
            FunctionInput.PermuteInput(
                input,
                permutation
            )
        )

        assertContentEquals(
            intArrayOf(3, 2, 2),
            output.shape
        )

        // Fill output gradient with unique values so that
        // the inverse permutation can be verified.
        output.grad = Tensor(
            DoubleArray(output.size) { (it + 1).toDouble() },
            output.shape,
            false
        )

        output.gradFn!!.apply()

        // inverse([2, 0, 1]) = [1, 2, 0]
        //
        // Verify a few positions explicitly through indexing.
        val expected = DoubleArray(input.size)

        for (i in 0 until input.shape[0]) {
            for (j in 0 until input.shape[1]) {
                for (k in 0 until input.shape[2]) {
                    expected[input.getFlatIndex(i, j, k)] =
                        output.grad!![k, i, j]
                }
            }
        }

        assertContentEquals(
            expected,
            input.grad!!.backedArray
        )
    }

    @Test
    fun `invalid permutation throws exception`() {
        val input = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0, 4.0),
            intArrayOf(2, 2)
        )

        assertFailsWith<IllegalArgumentException> {
            Permute.forward(
                FunctionInput.PermuteInput(
                    input,
                    intArrayOf(0, 0)
                )
            )
        }

        assertFailsWith<IllegalArgumentException> {
            Permute.forward(
                FunctionInput.PermuteInput(
                    input,
                    intArrayOf(0, 2)
                )
            )
        }
    }

    @Test
    fun `permutation with wrong rank throws exception`() {
        val input = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0, 4.0),
            intArrayOf(2, 2)
        )

        assertFailsWith<IllegalArgumentException> {
            Permute.forward(
                FunctionInput.PermuteInput(
                    input,
                    intArrayOf(0, 1, 2)
                )
            )
        }
    }

    @Test
    fun `does not create gradient when requiresGrad is false`() {
        val input = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0, 4.0),
            intArrayOf(2, 2),
            requiresGrad = false
        )

        val output = Permute.forward(
            FunctionInput.PermuteInput(
                input,
                intArrayOf(1, 0)
            )
        )

        assertEquals(null, input.grad)
        assertEquals(null, output.gradFn)
    }
}