package io.github.youssefrashidy.function.math

import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.tensor.Tensor
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import kotlin.math.abs
import kotlin.math.ln

class LogTest {

    private val eps = 1e-9

    @Test
    fun `forward computes elementwise natural log for 1D tensor`() {
        val input = Tensor(doubleArrayOf(1.0, 2.0, Math.E, 10.0), intArrayOf(4))

        val output = Log.forward(FunctionInput.Tensors(arrayOf(input)))

        assertArrayEquals(input.shape, output.shape)
        assertEquals(ln(1.0), output[0], eps)
        assertEquals(ln(2.0), output[1], eps)
        assertEquals(1.0, output[2], eps) // ln(e) == 1
        assertEquals(ln(10.0), output[3], eps)
    }

    @Test
    fun `forward computes elementwise natural log for 2D tensor`() {
        val input = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0, 4.0),
            intArrayOf(2, 2)
        )

        val output = Log.forward(FunctionInput.Tensors(arrayOf(input)))

        assertEquals(ln(1.0), output[0, 0], eps)
        assertEquals(ln(2.0), output[0, 1], eps)
        assertEquals(ln(3.0), output[1, 0], eps)
        assertEquals(ln(4.0), output[1, 1], eps)
    }

    @Test
    fun `forward throws when a value is zero`() {
        val input = Tensor(doubleArrayOf(1.0, 0.0), intArrayOf(2))

        assertThrows(IllegalArgumentException::class.java) {
            Log.forward(FunctionInput.Tensors(arrayOf(input)))
        }
    }

    @Test
    fun `forward throws when a value is negative`() {
        val input = Tensor(doubleArrayOf(1.0, -3.0), intArrayOf(2))

        assertThrows(IllegalArgumentException::class.java) {
            Log.forward(FunctionInput.Tensors(arrayOf(input)))
        }
    }

    @Test
    fun `forward throws when input list does not contain exactly one tensor`() {
        val a = Tensor(doubleArrayOf(1.0), intArrayOf(1))
        val b = Tensor(doubleArrayOf(2.0), intArrayOf(1))

        assertThrows(IllegalArgumentException::class.java) {
            Log.forward(FunctionInput.Tensors(arrayOf(a, b)))
        }
    }

    @Test
    fun `forward throws on wrong FunctionInput subtype`() {
        // Adjust this to whatever the other FunctionInput variant actually is,
        // e.g. FunctionInput.Scalar(1.0) — placeholder shown for illustration.
        assertThrows(IllegalArgumentException::class.java) {
            Log.forward(FunctionInput.Tensors(emptyArray()))
        }
    }

    @Test
    fun `output records prevTensors and gradFn when input requires grad`() {
        val input = Tensor(doubleArrayOf(2.0, 4.0), intArrayOf(2), requiresGrad = true)

        val output = Log.forward(FunctionInput.Tensors(arrayOf(input)))

        assertEquals(1, output.prevTensors.size)
        assertSame(input, output.prevTensors[0])
        assertNotNull(output.gradFn)
        assertNotNull(input.grad) // grad buffer initialized even before backward runs
    }

    @Test
    fun `gradFn accumulates correct derivative 1 over x`() {
        val input = Tensor(doubleArrayOf(2.0, 4.0, 0.5), intArrayOf(3), requiresGrad = true)
        val output = Log.forward(FunctionInput.Tensors(arrayOf(input)))

        // Seed output grad as if it were the loss (dL/d(output) = 1 for each element)
        output.grad = Tensor(DoubleArray(3) { 1.0 }, output.shape, false)

        output.gradFn!!.apply()

        // d(ln x)/dx = 1/x, chained with upstream grad of 1.0
        assertEquals(1.0 / 2.0, input.grad!![0], eps)
        assertEquals(1.0 / 4.0, input.grad!![1], eps)
        assertEquals(1.0 / 0.5, input.grad!![2], eps)
    }

    @Test
    fun `gradFn is null when input does not require grad`() {
        val input = Tensor(doubleArrayOf(3.0), intArrayOf(1), requiresGrad = false)

        val output = Log.forward(FunctionInput.Tensors(arrayOf(input)))

        assertNull(output.gradFn)
    }
}