package io.github.youssefrashidy.function.operators

import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.tensor.Tensor
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class DivisionTest {

    @Test
    fun `elementwise division`() {
        val a = Tensor(
            doubleArrayOf(10.0, 20.0, 30.0),
            intArrayOf(3)
        )

        val b = Tensor(
            doubleArrayOf(2.0, 4.0, 5.0),
            intArrayOf(3)
        )

        val result = Division.forward(
            FunctionInput.Tensors(arrayOf(a, b))
        )

        assertContentEquals(
            doubleArrayOf(5.0, 5.0, 6.0),
            result.backedArray
        )
    }

    @Test
    fun `broadcasted division`() {
        val a = Tensor(
            doubleArrayOf(
                10.0, 20.0, 30.0,
                40.0, 50.0, 60.0
            ),
            intArrayOf(2, 3)
        )

        val b = Tensor(
            doubleArrayOf(2.0, 5.0, 10.0),
            intArrayOf(3)
        )

        val result = Division.forward(
            FunctionInput.Tensors(arrayOf(a, b))
        )

        assertContentEquals(
            doubleArrayOf(
                5.0, 4.0, 3.0,
                20.0, 10.0, 6.0
            ),
            result.backedArray
        )

        assertContentEquals(
            intArrayOf(2, 3),
            result.shape
        )
    }

    @Test
    fun `division by zero throws exception`() {
        val a = Tensor(
            doubleArrayOf(10.0, 20.0, 30.0),
            intArrayOf(3)
        )

        val b = Tensor(
            doubleArrayOf(2.0, 0.0, 5.0),
            intArrayOf(3)
        )

        assertFailsWith<IllegalArgumentException> {
            Division.forward(
                FunctionInput.Tensors(arrayOf(a, b))
            )
        }
    }

    @Test
    fun `backward computes numerator and denominator gradients`() {
        val a = Tensor(
            doubleArrayOf(10.0, 20.0, 30.0),
            intArrayOf(3)
        )

        val b = Tensor(
            doubleArrayOf(2.0, 4.0, 5.0),
            intArrayOf(3)
        )

        val output = Division.forward(
            FunctionInput.Tensors(arrayOf(a, b))
        )

        // dL/dz
        output.grad = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            false
        )

        output.gradFn!!.apply()

        // dL/da = dL/dz * (1 / b)
        //
        // [1/2, 2/4, 3/5]
        assertContentEquals(
            doubleArrayOf(0.5, 0.5, 0.6),
            a.grad!!.backedArray
        )

        // dL/db = dL/dz * (-a / b²)
        //
        // [1*(-10/4), 2*(-20/16), 3*(-30/25)]
        // = [-2.5, -2.5, -3.6]
        assertContentEquals(
            doubleArrayOf(-2.5, -2.5, -3.6),
            b.grad!!.backedArray
        )
    }

    @Test
    fun `broadcasted backward accumulates denominator gradient`() {
        val a = Tensor(
            doubleArrayOf(
                10.0, 20.0, 30.0,
                40.0, 50.0, 60.0
            ),
            intArrayOf(2, 3)
        )

        val b = Tensor(
            doubleArrayOf(2.0, 5.0, 10.0),
            intArrayOf(3)
        )

        val output = Division.forward(
            FunctionInput.Tensors(arrayOf(a, b))
        )

        // dL/dz
        output.grad = Tensor(
            doubleArrayOf(
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
            ),
            intArrayOf(2, 3),
            false
        )

        output.gradFn!!.apply()

        // dL/da = dL/dz / b
        assertContentEquals(
            doubleArrayOf(
                0.5, 0.4, 0.3,
                2.0, 1.0, 0.6
            ),
            a.grad!!.backedArray
        )

        // b is broadcast, so its gradient accumulates across rows.
        //
        // db[0] = -(1*10/2²) -(4*40/2²)
        //       = -2.5 - 10
        //       = -42.5
        //
        // db[1] = -(2*20/5²) -(5*50/5²)
        //       = -1.6 - 10
        //       = -11.6
        //
        // db[2] = -(3*30/10²) -(6*60/10²)
        //       = -0.9 - 3.6
        //       = -4.5
        assertContentEquals(
            doubleArrayOf(-42.5, -11.6, -4.5),
            b.grad!!.backedArray
        )
    }

    @Test
    fun `does not create gradient for tensor with requiresGrad false`() {
        val a = Tensor(
            doubleArrayOf(10.0, 20.0, 30.0),
            intArrayOf(3),
            requiresGrad = false
        )

        val b = Tensor(
            doubleArrayOf(2.0, 4.0, 5.0),
            intArrayOf(3),
            requiresGrad = true
        )

        val output = Division.forward(
            FunctionInput.Tensors(arrayOf(a, b))
        )

        output.grad = Tensor(
            doubleArrayOf(1.0, 1.0, 1.0),
            intArrayOf(3),
            false
        )

        output.gradFn!!.apply()

        assertEquals(null, a.grad)

        // -(a / b²)
        assertContentEquals(
            doubleArrayOf(
                -2.5,
                -1.25,
                -1.2
            ),
            b.grad!!.backedArray
        )
    }
}