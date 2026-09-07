package io.github.youssefrashidy.function.linalg

import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.tensor.Tensor
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class MatMulTest {

    private fun assertArrayEquals(
        expected: DoubleArray,
        actual: DoubleArray,
        epsilon: Double = 1e-10
    ) {
        assertEquals(expected.size, actual.size)

        for (i in expected.indices) {
            assertEquals(expected[i], actual[i], epsilon)
        }
    }

    @Test
    fun `2D matrix multiplication`() {
        val a = Tensor(
            doubleArrayOf(
                1.0, 2.0,
                3.0, 4.0
            ),
            intArrayOf(2, 2),
            false
        )

        val b = Tensor(
            doubleArrayOf(
                5.0, 6.0,
                7.0, 8.0
            ),
            intArrayOf(2, 2),
            false
        )

        val output = MatMul.forward(
            FunctionInput.MatMulInput(a, b)
        )

        assertContentEquals(intArrayOf(2, 2), output.shape)

        assertEquals(19.0, output[0, 0])
        assertEquals(22.0, output[0, 1])
        assertEquals(43.0, output[1, 0])
        assertEquals(50.0, output[1, 1])
    }

    @Test
    fun `matrix multiplication with non square matrices`() {
        val a = Tensor(
            doubleArrayOf(
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
            ),
            intArrayOf(2, 3),
            false
        )

        val b = Tensor(
            doubleArrayOf(
                7.0, 8.0,
                9.0, 10.0,
                11.0, 12.0
            ),
            intArrayOf(3, 2),
            false
        )

        val output = MatMul.forward(
            FunctionInput.MatMulInput(a, b)
        )

        assertContentEquals(intArrayOf(2, 2), output.shape)

        assertEquals(58.0, output[0, 0])
        assertEquals(64.0, output[0, 1])
        assertEquals(139.0, output[1, 0])
        assertEquals(154.0, output[1, 1])
    }

    @Test
    fun `dot product`() {
        val a = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            false
        )

        val b = Tensor(
            doubleArrayOf(4.0, 5.0, 6.0),
            intArrayOf(3),
            false
        )

        val output = MatMul.forward(
            FunctionInput.MatMulInput(a, b)
        )

        assertContentEquals(intArrayOf(1), output.shape)
        assertEquals(32.0, output[0])
    }

    @Test
    fun `vector matrix multiplication`() {
        val a = Tensor(
            doubleArrayOf(1.0, 2.0),
            intArrayOf(2),
            false
        )

        val b = Tensor(
            doubleArrayOf(
                3.0, 4.0,
                5.0, 6.0
            ),
            intArrayOf(2, 2),
            false
        )

        val output = MatMul.forward(
            FunctionInput.MatMulInput(a, b)
        )

        /*
         * Internally:
         *
         * [1 2] @ [3 4
         *          5 6]
         *
         * = [13 16]
         *
         * Since your implementation reshapes the vector to [1, 2],
         * the current result shape is [1, 2].
         */

        assertContentEquals(intArrayOf(1, 2), output.shape)

        assertEquals(13.0, output[0, 0])
        assertEquals(16.0, output[0, 1])
    }

    @Test
    fun `matrix vector multiplication`() {
        val a = Tensor(
            doubleArrayOf(
                1.0, 2.0,
                3.0, 4.0
            ),
            intArrayOf(2, 2),
            false
        )

        val b = Tensor(
            doubleArrayOf(5.0, 6.0),
            intArrayOf(2),
            false
        )

        val output = MatMul.forward(
            FunctionInput.MatMulInput(a, b)
        )

        /*
         * [1 2] @ [5] = [17]
         * [3 4]   [6]   [39]
         *
         * Your current implementation reshapes b to [2, 1],
         * so the current result shape is [2, 1].
         */

        assertContentEquals(intArrayOf(2, 1), output.shape)

        assertEquals(17.0, output[0, 0])
        assertEquals(39.0, output[1, 0])
    }

    @Test
    fun `backward for matrix multiplication`() {
        val a = Tensor(
            doubleArrayOf(
                1.0, 2.0,
                3.0, 4.0
            ),
            intArrayOf(2, 2),
            true
        )

        val b = Tensor(
            doubleArrayOf(
                5.0, 6.0,
                7.0, 8.0
            ),
            intArrayOf(2, 2),
            true
        )

        val output = MatMul.forward(
            FunctionInput.MatMulInput(a, b)
        )

        /*
         * Let:
         *
         * A = [1 2]    B = [5 6]
         *     [3 4]        [7 8]
         *
         * dL/dC = [1 2]
         *         [3 4]
         *
         * dL/dA = dL/dC @ B^T
         *
         *        [1 2] @ [5 7] = [17 23]
         *        [3 4]   [6 8]   [39 53]
         *
         * dL/dB = A^T @ dL/dC
         *
         *        [1 3] @ [1 2] = [10 14]
         *        [2 4]   [3 4]   [14 20]
         */

        output.grad = Tensor(
            doubleArrayOf(
                1.0, 2.0,
                3.0, 4.0
            ),
            intArrayOf(2, 2),
            false
        )

        output.gradFn!!.apply()

        assertArrayEquals(
            doubleArrayOf(
                17.0, 23.0,
                39.0, 53.0
            ),
            a.grad!!.backedArray
        )

        assertArrayEquals(
            doubleArrayOf(
                10.0, 14.0,
                14.0, 20.0
            ),
            b.grad!!.backedArray
        )
    }

    @Test
    fun `backward for dot product`() {
        val a = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            true
        )

        val b = Tensor(
            doubleArrayOf(4.0, 5.0, 6.0),
            intArrayOf(3),
            true
        )

        val output = MatMul.forward(
            FunctionInput.MatMulInput(a, b)
        )

        output.grad = Tensor(
            doubleArrayOf(10.0),
            intArrayOf(1),
            false
        )

        output.gradFn!!.apply()

        /*
         * da = dz * b
         * db = dz * a
         */

        assertArrayEquals(
            doubleArrayOf(40.0, 50.0, 60.0),
            a.grad!!.backedArray
        )

        assertArrayEquals(
            doubleArrayOf(10.0, 20.0, 30.0),
            b.grad!!.backedArray
        )
    }

    @Test
    fun `incompatible matrix dimensions fail`() {
        val a = Tensor(
            doubleArrayOf(
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
            ),
            intArrayOf(2, 3),
            false
        )

        val b = Tensor(
            doubleArrayOf(
                1.0, 2.0,
                3.0, 4.0
            ),
            intArrayOf(2, 2),
            false
        )

        kotlin.test.assertFailsWith<IllegalArgumentException> {
            MatMul.forward(
                FunctionInput.MatMulInput(a, b)
            )
        }
    }
}