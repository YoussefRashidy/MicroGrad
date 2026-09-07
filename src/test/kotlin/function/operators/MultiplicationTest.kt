package io.github.youssefrashidy.function.operators

import io.github.youssefrashidy.function.FunctionInput
import io.github.youssefrashidy.tensor.Tensor
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class MultiplicationTest {

    @Test
    fun `elementwise multiplication`() {
        val a = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3)
        )
        val b = Tensor(
            doubleArrayOf(4.0, 5.0, 6.0),
            intArrayOf(3)
        )

        val result = Multiplication.forward(
            FunctionInput.Tensors(arrayOf(a, b))
        )

        assertContentEquals(
            doubleArrayOf(4.0, 10.0, 18.0),
            result.backedArray
        )
    }

    @Test
    fun `broadcasted multiplication`() {
        val a = Tensor(
            doubleArrayOf(
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
            ),
            intArrayOf(2, 3)
        )

        val b = Tensor(
            doubleArrayOf(10.0, 20.0, 30.0),
            intArrayOf(3)
        )

        val result = Multiplication.forward(
            FunctionInput.Tensors(arrayOf(a, b))
        )

        assertContentEquals(
            doubleArrayOf(
                10.0, 40.0, 90.0,
                40.0, 100.0, 180.0
            ),
            result.backedArray
        )

        assertContentEquals(
            intArrayOf(2, 3),
            result.shape
        )
    }

    @Test
    fun `incompatible shapes throw exception`() {
        val a = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3)
        )

        val b = Tensor(
            doubleArrayOf(1.0, 2.0),
            intArrayOf(2)
        )

        assertFailsWith<IllegalArgumentException> {
            Multiplication.forward(
                FunctionInput.Tensors(arrayOf(a, b))
            )
        }
    }

    @Test
    fun `backward gives gradient multiplied by other operand`() {
        val a = Tensor(
            doubleArrayOf(2.0, 3.0, 4.0),
            intArrayOf(3)
        )

        val b = Tensor(
            doubleArrayOf(5.0, 6.0, 7.0),
            intArrayOf(3)
        )

        val output = Multiplication.forward(
            FunctionInput.Tensors(arrayOf(a, b))
        )

        // Pretend dL/dz = [1, 2, 3]
        output.grad = Tensor(
            doubleArrayOf(1.0, 2.0, 3.0),
            intArrayOf(3),
            false
        )

        output.gradFn!!.apply()

        // dL/da = dL/dz * b
        assertContentEquals(
            doubleArrayOf(5.0, 12.0, 21.0),
            a.grad!!.backedArray
        )

        // dL/db = dL/dz * a
        assertContentEquals(
            doubleArrayOf(2.0, 6.0, 12.0),
            b.grad!!.backedArray
        )
    }

    @Test
    fun `broadcasted backward accumulates gradients correctly`() {
        val a = Tensor(
            doubleArrayOf(
                2.0, 3.0, 4.0,
                5.0, 6.0, 7.0
            ),
            intArrayOf(2, 3)
        )

        val b = Tensor(
            doubleArrayOf(10.0, 20.0, 30.0),
            intArrayOf(3)
        )

        val output = Multiplication.forward(
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

        // dL/da = dL/dz * b
        assertContentEquals(
            doubleArrayOf(
                10.0, 40.0, 90.0,
                40.0, 100.0, 180.0
            ),
            a.grad!!.backedArray
        )

        // b is broadcast:
        //
        // db[0] = 1*2 + 4*5 = 22
        // db[1] = 2*3 + 5*6 = 36
        // db[2] = 3*4 + 6*7 = 54
        assertContentEquals(
            doubleArrayOf(22.0, 36.0, 54.0),
            b.grad!!.backedArray
        )
    }

    @Test
    fun `does not create gradient for tensor with requiresGrad false`() {
        val a = Tensor(
            doubleArrayOf(2.0, 3.0, 4.0),
            intArrayOf(3),
            requiresGrad = false
        )

        val b = Tensor(
            doubleArrayOf(5.0, 6.0, 7.0),
            intArrayOf(3),
            requiresGrad = true
        )

        val output = Multiplication.forward(
            FunctionInput.Tensors(arrayOf(a, b))
        )

        output.grad = Tensor(
            doubleArrayOf(1.0, 1.0, 1.0),
            intArrayOf(3),
            false
        )

        output.gradFn!!.apply()

        assertEquals(null, a.grad)

        assertContentEquals(
            doubleArrayOf(2.0, 3.0, 4.0),
            b.grad!!.backedArray
        )
    }
}