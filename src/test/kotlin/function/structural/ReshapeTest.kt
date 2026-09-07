package io.github.youssefrashidy.function.structural

import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.tensor.Tensor
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class ReshapeTest {

    private val eps = 1e-9

    @Test
    fun `forward reshapes 1D tensor into 2D preserving flat data order`() {
        val input = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            intArrayOf(6)
        )

        val output = Reshape.forward(
            FunctionInput.ReshapeInput(arrayOf(input), intArrayOf(2, 3))
        )

        assertArrayEquals(intArrayOf(2, 3), output.shape)
        assertEquals(1.0, output[0, 0], eps)
        assertEquals(2.0, output[0, 1], eps)
        assertEquals(3.0, output[0, 2], eps)
        assertEquals(4.0, output[1, 0], eps)
        assertEquals(5.0, output[1, 1], eps)
        assertEquals(6.0, output[1, 2], eps)
    }

    @Test
    fun `forward throws when element count does not match new shape`() {
        val input = Tensor(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(4))

        assertThrows(IllegalArgumentException::class.java) {
            Reshape.forward(FunctionInput.ReshapeInput(arrayOf(input), intArrayOf(3, 2)))
        }
    }

    @Test
    fun `forward throws when more than one tensor is provided`() {
        val a = Tensor(doubleArrayOf(1.0, 2.0), intArrayOf(2))
        val b = Tensor(doubleArrayOf(3.0, 4.0), intArrayOf(2))

        assertThrows(IllegalArgumentException::class.java) {
            Reshape.forward(FunctionInput.ReshapeInput(arrayOf(a, b), intArrayOf(2)))
        }
    }

    @Test
    fun `output shares backing array with input (view, not copy)`() {
        val input = Tensor(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(4))

        val output = Reshape.forward(
            FunctionInput.ReshapeInput(arrayOf(input), intArrayOf(2, 2))
        )

        assertSame(input.backedArray, output.backedArray)
    }

    @Test
    fun `gradFn reshapes upstream gradient back to input shape and accumulates`() {
        // rank (2) is much smaller than size (6) -- this is the case that will
        // fail if grad buffer is allocated with shape.size instead of tensor.size
        val input = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            intArrayOf(2, 3),
            requiresGrad = true
        )

        val output = Reshape.forward(
            FunctionInput.ReshapeInput(arrayOf(input), intArrayOf(6))
        )

        output.grad = Tensor(
            doubleArrayOf(10.0, 20.0, 30.0, 40.0, 50.0, 60.0),
            output.shape,
            false
        )

        output.gradFn!!.apply()

        assertEquals(6, input.grad!!.backedArray.size)
        assertEquals(10.0, input.grad!![0, 0], eps)
        assertEquals(20.0, input.grad!![0, 1], eps)
        assertEquals(30.0, input.grad!![0, 2], eps)
        assertEquals(40.0, input.grad!![1, 0], eps)
        assertEquals(50.0, input.grad!![1, 1], eps)
        assertEquals(60.0, input.grad!![1, 2], eps)
    }
}