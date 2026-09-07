package io.github.youssefrashidy.function.operators

import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.tensor.Tensor
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class SubtractionTest {

    @Test
    fun `subtraction subtracts two tensors elementwise`() {
        val a = Tensor(
            doubleArrayOf(10.0, 20.0, 30.0),
            intArrayOf(3),
            requiresGrad = false
        )

        val b = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            requiresGrad = false
        )

        val result = Subtraction.forward(
            FunctionInput.Tensors(arrayOf(a, b))
        )

        assertArrayEquals(
            intArrayOf(3),
            result.shape
        )

        assertArrayEquals(
            doubleArrayOf(9.0, 18.0, 27.0),
            result.backedArray
        )

        assertArrayEquals(
            arrayOf(a, b),
            result.prevTensors
        )
    }

    @Test
    fun `subtraction broadcasts lower rank tensor`() {
        val a = Tensor(
            doubleArrayOf(
                10.0, 20.0, 30.0,
                40.0, 50.0, 60.0
            ),
            intArrayOf(2, 3),
            requiresGrad = false
        )

        val b = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            requiresGrad = false
        )

        val result = Subtraction.forward(
            FunctionInput.Tensors(arrayOf(a, b))
        )

        assertArrayEquals(
            intArrayOf(2, 3),
            result.shape
        )

        assertArrayEquals(
            doubleArrayOf(
                9.0, 18.0, 27.0,
                39.0, 48.0, 57.0
            ),
            result.backedArray
        )
    }

    @Test
    fun `subtraction rejects incompatible shapes`() {
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
            Subtraction.forward(
                FunctionInput.Tensors(arrayOf(a, b))
            )
        }
    }

    @Test
    fun `subtraction backward gives positive gradient to first tensor`() {
        val a = Tensor(
            doubleArrayOf(10.0, 20.0, 30.0),
            intArrayOf(3),
            requiresGrad = true
        )

        val b = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            requiresGrad = true
        )

        val output = Subtraction.forward(
            FunctionInput.Tensors(arrayOf(a, b))
        )

        output.grad = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            requiresGrad = false
        )

        Subtraction.backward(
            a,
            b,
            a,
            b,
            output
        )

        output.gradFn!!.apply()

        assertArrayEquals(
            doubleArrayOf(1.0, 2.0, 3.0),
            a.grad!!.backedArray
        )
    }

    @Test
    fun `subtraction backward gives negative gradient to second tensor`() {
        val a = Tensor(
            doubleArrayOf(10.0, 20.0, 30.0),
            intArrayOf(3),
            requiresGrad = true
        )

        val b = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            requiresGrad = true
        )

        val output = Subtraction.forward(
            FunctionInput.Tensors(arrayOf(a, b))
        )

        output.grad = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            requiresGrad = false
        )

        Subtraction.backward(
            a,
            b,
            a,
            b,
            output
        )

        output.gradFn!!.apply()

        assertArrayEquals(
            doubleArrayOf(-1.0, -2.0, -3.0),
            b.grad!!.backedArray
        )
    }

    @Test
    fun `subtraction backward handles broadcasting`() {
        val a = Tensor(
            doubleArrayOf(
                10.0, 20.0, 30.0,
                40.0, 50.0, 60.0
            ),
            intArrayOf(2, 3),
            requiresGrad = true
        )

        val b = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            requiresGrad = true
        )

        val (aBroadcasted, bBroadcasted, _) =
            Tensor.broadcast(a, b)

        val output = Subtraction.forward(
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

        Subtraction.backward(
            a,
            b,
            aBroadcasted,
            bBroadcasted,
            output
        )

        output.gradFn!!.apply()

        // a is not broadcasted.
        assertArrayEquals(
            doubleArrayOf(
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
            ),
            a.grad!!.backedArray
        )

        /*
         * b is broadcast over the first dimension.
         *
         * db = -sum(output.grad over broadcast dimension)
         *
         * = -( [1,2,3] + [4,5,6] )
         * = [-5,-7,-9]
         */
        assertArrayEquals(
            doubleArrayOf(-5.0, -7.0, -9.0),
            b.grad!!.backedArray
        )
    }

    @Test
    fun `subtraction backward respects requiresGrad`() {
        val a = Tensor(
            doubleArrayOf(10.0, 20.0, 30.0),
            intArrayOf(3),
            requiresGrad = true
        )

        val b = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            requiresGrad = false
        )

        val output = Subtraction.forward(
            FunctionInput.Tensors(arrayOf(a, b))
        )

        output.grad = Tensor(
            doubleArrayOf(1.0, 1.0, 1.0),
            intArrayOf(3),
            requiresGrad = false
        )

        Subtraction.backward(
            a,
            b,
            a,
            b,
            output
        )

        output.gradFn!!.apply()

        assertNotNull(a.grad)
        assertNull(b.grad)

        assertArrayEquals(
            doubleArrayOf(1.0, 1.0, 1.0),
            a.grad!!.backedArray
        )
    }
}

