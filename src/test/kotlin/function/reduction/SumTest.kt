package io.github.youssefrashidy.function.reduction

import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.tensor.Tensor
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class SumTest {

    private val eps = 1e-9

    @Test
    fun `full reduction of 1D tensor without keepDims returns scalar shaped tensor`() {
        val input = Tensor(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(4))

        val output = Sum.forward(
            FunctionInput.SumInput(input, intArrayOf(0), keepDims = false)
        )

        assertArrayEquals(intArrayOf(), output.shape)
        assertEquals(10.0, output.backedArray[0], eps)
    }

    @Test
    fun `full reduction of 1D tensor with keepDims returns shape 1`() {
        val input = Tensor(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(4))

        val output = Sum.forward(
            FunctionInput.SumInput(input, intArrayOf(0), keepDims = true)
        )

        assertArrayEquals(intArrayOf(1), output.shape)
        assertEquals(10.0, output[0], eps)
    }

    @Test
    fun `sum along axis 0 of 2D tensor without keepDims`() {
        // shape (2,3): [[1,2,3],[4,5,6]]
        val input = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            intArrayOf(2, 3)
        )

        val output = Sum.forward(
            FunctionInput.SumInput(input, intArrayOf(0), keepDims = false)
        )

        assertArrayEquals(intArrayOf(3), output.shape)
        assertEquals(5.0, output[0], eps)  // 1+4
        assertEquals(7.0, output[1], eps)  // 2+5
        assertEquals(9.0, output[2], eps)  // 3+6
    }

    @Test
    fun `sum along axis 1 of 2D tensor without keepDims`() {
        // shape (2,3): [[1,2,3],[4,5,6]]
        val input = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            intArrayOf(2, 3)
        )

        val output = Sum.forward(
            FunctionInput.SumInput(input, intArrayOf(1), keepDims = false)
        )

        assertArrayEquals(intArrayOf(2), output.shape)
        assertEquals(6.0, output[0], eps)   // 1+2+3
        assertEquals(15.0, output[1], eps)  // 4+5+6
    }

    @Test
    fun `sum along axis 0 of 2D tensor with keepDims preserves rank`() {
        val input = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            intArrayOf(2, 3)
        )

        val output = Sum.forward(
            FunctionInput.SumInput(input, intArrayOf(0), keepDims = true)
        )

        assertArrayEquals(intArrayOf(1, 3), output.shape)
        assertEquals(5.0, output[0, 0], eps)
        assertEquals(7.0, output[0, 1], eps)
        assertEquals(9.0, output[0, 2], eps)
    }

    @Test
    fun `sum over multiple axes on 3D tensor reduces to correct shape and values`() {
        // shape (2,2,2), values 1..8
        val input = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
            intArrayOf(2, 2, 2)
        )

        // reduce axes 0 and 2, keep axis 1
        val output = Sum.forward(
            FunctionInput.SumInput(input, intArrayOf(0, 2), keepDims = false)
        )

        assertArrayEquals(intArrayOf(2), output.shape)
        // axis1=0: elements (0,0,0)=1,(0,0,1)=2,(1,0,0)=5,(1,0,1)=6 -> 14
        // axis1=1: elements (0,1,0)=3,(0,1,1)=4,(1,1,0)=7,(1,1,1)=8 -> 22
        assertEquals(14.0, output[0], eps)
        assertEquals(22.0, output[1], eps)
    }

    @Test
    fun `output records prevTensors and gradFn when input requires grad`() {
        val input = Tensor(doubleArrayOf(1.0, 2.0, 3.0), intArrayOf(3), requiresGrad = true)

        val output = Sum.forward(
            FunctionInput.SumInput(input, intArrayOf(0), keepDims = false)
        )

        assertEquals(1, output.prevTensors.size)
        assertSame(input, output.prevTensors[0])
        assertNotNull(output.gradFn)
        assertNotNull(input.grad)
    }

    @Test
    fun `gradFn broadcasts upstream grad of 1 back to every input element`() {
        val input = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            intArrayOf(2, 3),
            requiresGrad = true
        )

        val output = Sum.forward(
            FunctionInput.SumInput(input, intArrayOf(1), keepDims = false)
        )

        // seed as if this were the final loss node: dL/d(sum_i) = 1 for each row-sum
        output.grad = Tensor(DoubleArray(2) { 1.0 }, output.shape, false)

        output.gradFn!!.apply()

        // d(sum)/dx_ij = 1 for every element regardless of axis reduced
        for (i in 0 until 6) {
            assertEquals(1.0, input.grad!!.backedArray[i], eps)
        }
    }
}