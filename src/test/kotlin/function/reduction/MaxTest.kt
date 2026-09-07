package io.github.youssefrashidy.function.reduction

import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.tensor.Tensor
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class MaxTest {

    private val eps = 1e-9

    @Test
    fun `full reduction of 1D tensor without keepDims returns scalar shaped tensor`() {
        val input = Tensor(doubleArrayOf(3.0, 7.0, 1.0, 5.0), intArrayOf(4))

        val output = Max.forward(
            FunctionInput.MaxInput(input, intArrayOf(0), keepDims = false)
        )

        assertArrayEquals(intArrayOf(), output.shape)
        assertEquals(7.0, output.backedArray[0], eps)
    }

    @Test
    fun `full reduction of 1D tensor with keepDims returns shape 1`() {
        val input = Tensor(doubleArrayOf(3.0, 7.0, 1.0, 5.0), intArrayOf(4))

        val output = Max.forward(
            FunctionInput.MaxInput(input, intArrayOf(0), keepDims = true)
        )

        assertArrayEquals(intArrayOf(1), output.shape)
        assertEquals(7.0, output[0], eps)
    }

    @Test
    fun `max along axis 0 of 2D tensor without keepDims`() {
        // shape (2,3): [[1,5,3],[4,2,6]]
        val input = Tensor(
            doubleArrayOf(1.0, 5.0, 3.0, 4.0, 2.0, 6.0),
            intArrayOf(2, 3)
        )

        val output = Max.forward(
            FunctionInput.MaxInput(input, intArrayOf(0), keepDims = false)
        )

        assertArrayEquals(intArrayOf(3), output.shape)
        assertEquals(4.0, output[0], eps) // max(1,4)
        assertEquals(5.0, output[1], eps) // max(5,2)
        assertEquals(6.0, output[2], eps) // max(3,6)
    }

    @Test
    fun `max along axis 1 of 2D tensor without keepDims`() {
        // shape (2,3): [[1,5,3],[4,2,6]]
        val input = Tensor(
            doubleArrayOf(1.0, 5.0, 3.0, 4.0, 2.0, 6.0),
            intArrayOf(2, 3)
        )

        val output = Max.forward(
            FunctionInput.MaxInput(input, intArrayOf(1), keepDims = false)
        )

        assertArrayEquals(intArrayOf(2), output.shape)
        assertEquals(5.0, output[0], eps) // max(1,5,3)
        assertEquals(6.0, output[1], eps) // max(4,2,6)
    }

    @Test
    fun `max along axis 0 (leading, non-last) of 2D tensor with keepDims preserves rank`() {
        // reduced axis is NOT the last dim -> exercises the recursive-branch fix
        val input = Tensor(
            doubleArrayOf(1.0, 5.0, 3.0, 4.0, 2.0, 6.0),
            intArrayOf(2, 3)
        )

        val output = Max.forward(
            FunctionInput.MaxInput(input, intArrayOf(0), keepDims = true)
        )

        assertArrayEquals(intArrayOf(1, 3), output.shape)
        assertEquals(4.0, output[0, 0], eps)
        assertEquals(5.0, output[0, 1], eps)
        assertEquals(6.0, output[0, 2], eps)
    }

    @Test
    fun `max along axis 1 (last axis) of 2D tensor with keepDims preserves rank`() {
        // reduced axis IS the last dim -> exercises the base-case branch,
        // which still has the un-patched outputIndices write.
        val input = Tensor(
            doubleArrayOf(1.0, 5.0, 3.0, 4.0, 2.0, 6.0),
            intArrayOf(2, 3)
        )

        val output = Max.forward(
            FunctionInput.MaxInput(input, intArrayOf(1), keepDims = true)
        )

        assertArrayEquals(intArrayOf(2, 1), output.shape)
        assertEquals(5.0, output[0, 0], eps)
        assertEquals(6.0, output[1, 0], eps)
    }

    @Test
    fun `max over multiple axes on 3D tensor reduces to correct shape and values`() {
        // shape (2,2,2), values 1..8
        val input = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
            intArrayOf(2, 2, 2)
        )

        // reduce axes 0 and 2, keep axis 1
        val output = Max.forward(
            FunctionInput.MaxInput(input, intArrayOf(0, 2), keepDims = false)
        )

        assertArrayEquals(intArrayOf(2), output.shape)
        // axis1=0: elements (0,0,0)=1,(0,0,1)=2,(1,0,0)=5,(1,0,1)=6 -> max 6
        // axis1=1: elements (0,1,0)=3,(0,1,1)=4,(1,1,0)=7,(1,1,1)=8 -> max 8
        assertEquals(6.0, output[0], eps)
        assertEquals(8.0, output[1], eps)
    }

    @Test
    fun `output records prevTensors and gradFn when input requires grad`() {
        val input = Tensor(doubleArrayOf(1.0, 5.0, 3.0), intArrayOf(3), requiresGrad = true)

        val output = Max.forward(
            FunctionInput.MaxInput(input, intArrayOf(0), keepDims = false)
        )

        assertEquals(1, output.prevTensors.size)
        assertSame(input, output.prevTensors[0])
        assertNotNull(output.gradFn)
        assertNotNull(input.grad)
    }

    @Test
    fun `gradFn routes gradient only to the argmax element`() {
        val input = Tensor(doubleArrayOf(1.0, 5.0, 3.0), intArrayOf(3), requiresGrad = true)

        val output = Max.forward(
            FunctionInput.MaxInput(input, intArrayOf(0), keepDims = false)
        )

        output.grad = Tensor(doubleArrayOf(1.0), output.shape, false)
        output.gradFn!!.apply()

        // only index 1 (the max, value 5.0) should receive gradient
        assertEquals(0.0, input.grad!!.backedArray[0], eps)
        assertEquals(1.0, input.grad!!.backedArray[1], eps)
        assertEquals(0.0, input.grad!!.backedArray[2], eps)
    }

    @Test
    fun `gradFn routes gradient per-row when reducing along axis 1`() {
        // shape (2,3): [[1,5,3],[4,2,6]]
        val input = Tensor(
            doubleArrayOf(1.0, 5.0, 3.0, 4.0, 2.0, 6.0),
            intArrayOf(2, 3),
            requiresGrad = true
        )

        val output = Max.forward(
            FunctionInput.MaxInput(input, intArrayOf(1), keepDims = false)
        )

        output.grad = Tensor(doubleArrayOf(1.0, 1.0), output.shape, false)
        output.gradFn!!.apply()

        // row 0 max is index 1 (value 5.0), row 1 max is index 2 (value 6.0)
        val expected = doubleArrayOf(0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        for (i in expected.indices) {
            assertEquals(expected[i], input.grad!!.backedArray[i], eps)
        }
    }
}