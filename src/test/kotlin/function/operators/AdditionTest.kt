package io.github.youssefrashidy.function.operators

import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.tensor.Tensor
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class AdditionTest {

    @Test
    fun `addition adds two tensors elementwise`() {
        val a = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            requiresGrad = false
        )

        val b = Tensor(
            doubleArrayOf(4.0, 5.0, 6.0),
            intArrayOf(3),
            requiresGrad = false
        )

        val result = Addition.forward(
            FunctionInput.Tensors(arrayOf(a, b))
        )

        assertArrayEquals(intArrayOf(3), result.shape)
        assertArrayEquals(
            doubleArrayOf(5.0, 7.0, 9.0),
            result.backedArray
        )
    }

    @Test
    fun `addition broadcasts lower rank tensor`() {
        val a = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            requiresGrad = false
        )

        val b = Tensor(
            doubleArrayOf(
                10.0, 20.0, 30.0,
                40.0, 50.0, 60.0
            ),
            intArrayOf(2, 3),
            requiresGrad = false
        )

        val result = Addition.forward(
            FunctionInput.Tensors(arrayOf(a, b))
        )

        assertArrayEquals(intArrayOf(2, 3), result.shape)

        assertArrayEquals(
            doubleArrayOf(
                11.0, 22.0, 33.0,
                41.0, 52.0, 63.0
            ),
            result.backedArray
        )
    }

    @Test
    fun `addition rejects more or fewer than two tensors`() {
        val a = Tensor(
            doubleArrayOf(1.0, 2.0),
            intArrayOf(2),
            requiresGrad = false
        )

        val b = Tensor(
            doubleArrayOf(3.0, 4.0),
            intArrayOf(2),
            requiresGrad = false
        )

        assertThrows(IllegalArgumentException::class.java) {
            Addition.forward(
                FunctionInput.Tensors(arrayOf(a))
            )
        }

        assertThrows(IllegalArgumentException::class.java) {
            Addition.forward(
                FunctionInput.Tensors(arrayOf(a, b, a))
            )
        }
    }

    @Test
    fun `addition rejects incompatible shapes`() {
        val a = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            requiresGrad = false
        )

        val b = Tensor(
            doubleArrayOf(1.0, 2.0),
            intArrayOf(2),
            requiresGrad = false
        )

        assertThrows(IllegalArgumentException::class.java) {
            Addition.forward(
                FunctionInput.Tensors(arrayOf(a, b))
            )
        }
    }

    @Test
    fun `addition backward gives gradient one to both tensors`() {
        val a = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            requiresGrad = true
        )

        val b = Tensor(
            doubleArrayOf(4.0, 5.0, 6.0),
            intArrayOf(3),
            requiresGrad = true
        )

        val output = Addition.forward(
            FunctionInput.Tensors(arrayOf(a, b))
        )

        output.grad = Tensor(
            doubleArrayOf(1.0, 1.0, 1.0),
            intArrayOf(3),
            requiresGrad = false
        )

        Addition.backward(a, b, a, b, output)

        output.gradFn!!.apply()

        assertNotNull(a.grad)
        assertNotNull(b.grad)

        assertArrayEquals(
            doubleArrayOf(1.0, 1.0, 1.0),
            a.grad!!.backedArray
        )

        assertArrayEquals(
            doubleArrayOf(1.0, 1.0, 1.0),
            b.grad!!.backedArray
        )
    }

    @Test
    fun `addition backward handles broadcasting`() {
        val a = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            requiresGrad = true
        )

        val b = Tensor(
            doubleArrayOf(
                10.0, 20.0, 30.0,
                40.0, 50.0, 60.0
            ),
            intArrayOf(2, 3),
            requiresGrad = true
        )

        val (aBroadcasted, bBroadcasted, _) =
            Tensor.broadcast(a, b)

        val output = Addition.forward(
            FunctionInput.Tensors(arrayOf(a, b))
        )

        output.grad = Tensor(
            doubleArrayOf(
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
            ),
            intArrayOf(2, 3),
            requiresGrad = false
        )

        Addition.backward(
            a,
            b,
            aBroadcasted,
            bBroadcasted,
            output
        )

        output.gradFn!!.apply()

        /*
         * a was broadcast over the first dimension.
         *
         * dL/da:
         *
         * [1, 2, 3]
         * [4, 5, 6]
         *
         * The broadcast view maps both rows back to a,
         * so the accumulated gradient is:
         *
         * [5, 7, 9]
         */
        assertArrayEquals(
            doubleArrayOf(5.0, 7.0, 9.0),
            a.grad!!.backedArray
        )

        /*
         * b was not broadcast, therefore its gradient
         * is exactly the output gradient.
         */
        assertArrayEquals(
            doubleArrayOf(
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
            ),
            b.grad!!.backedArray
        )
    }

    @Test
    fun `addition backward respects requiresGrad`() {
        val a = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            requiresGrad = true
        )

        val b = Tensor(
            doubleArrayOf(4.0, 5.0, 6.0),
            intArrayOf(3),
            requiresGrad = false
        )

        val output = Addition.forward(
            FunctionInput.Tensors(arrayOf(a, b))
        )

        output.grad = Tensor(
            doubleArrayOf(1.0, 1.0, 1.0),
            intArrayOf(3),
            requiresGrad = false
        )

        Addition.backward(a, b, a, b, output)

        output.gradFn!!.apply()

        assertNotNull(a.grad)
        assertNull(b.grad)

        assertArrayEquals(
            doubleArrayOf(1.0, 1.0, 1.0),
            a.grad!!.backedArray
        )
    }
}

