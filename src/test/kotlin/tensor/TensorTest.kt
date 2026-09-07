package io.github.youssefrashidy.tensor

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class TensorTest {

    @Test
    fun `tensor exposes correct rank and size`() {
        val tensor = Tensor(
            backedArray = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            shape = intArrayOf(2, 3),
            requiresGrad = false
        )

        assertEquals(2, tensor.rank)
        assertEquals(6, tensor.size)
        assertArrayEquals(intArrayOf(2, 3), tensor.shape)
    }

    @Test
    fun `computeStrides computes row major strides`() {
        assertArrayEquals(
            intArrayOf(12, 4, 1),
            Tensor.computeStrides(intArrayOf(2, 3, 4))
        )

        assertArrayEquals(
            intArrayOf(3, 1),
            Tensor.computeStrides(intArrayOf(2, 3))
        )

        assertArrayEquals(
            intArrayOf(1),
            Tensor.computeStrides(intArrayOf(5))
        )
    }

    @Test
    fun `get returns value at multidimensional index`() {
        val tensor = Tensor(
            backedArray = doubleArrayOf(
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
            ),
            shape = intArrayOf(2, 3),
            requiresGrad = false
        )

        assertEquals(1.0, tensor[0, 0])
        assertEquals(3.0, tensor[0, 2])
        assertEquals(4.0, tensor[1, 0])
        assertEquals(6.0, tensor[1, 2])
    }

    @Test
    fun `set changes value at multidimensional index`() {
        val tensor = Tensor(
            backedArray = DoubleArray(6),
            shape = intArrayOf(2, 3),
            requiresGrad = false
        )

        tensor[1, 2] = 42.0

        assertEquals(42.0, tensor[1, 2])
        assertEquals(42.0, tensor.backedArray[5])
    }

    @Test
    fun `getFlatIndex converts multidimensional index correctly`() {
        val tensor = Tensor(
            backedArray = DoubleArray(24),
            shape = intArrayOf(2, 3, 4),
            requiresGrad = false
        )

        assertEquals(0, tensor.getFlatIndex(0, 0, 0))
        assertEquals(1, tensor.getFlatIndex(0, 0, 1))
        assertEquals(4, tensor.getFlatIndex(0, 1, 0))
        assertEquals(12, tensor.getFlatIndex(1, 0, 0))
        assertEquals(23, tensor.getFlatIndex(1, 2, 3))
    }

    @Test
    fun `get rejects wrong number of indices`() {
        val tensor = Tensor(
            backedArray = DoubleArray(6),
            shape = intArrayOf(2, 3),
            requiresGrad = false
        )

        assertThrows(IllegalArgumentException::class.java) {
            tensor[0]
        }

        assertThrows(IllegalArgumentException::class.java) {
            tensor[0, 1, 2]
        }
    }

    @Test
    fun `getFlatIndex rejects out of bounds indices`() {
        val tensor = Tensor(
            backedArray = DoubleArray(6),
            shape = intArrayOf(2, 3),
            requiresGrad = false
        )

        assertThrows(IllegalArgumentException::class.java) {
            tensor[2, 0]
        }

        assertThrows(IllegalArgumentException::class.java) {
            tensor[0, 3]
        }

        assertThrows(IllegalArgumentException::class.java) {
            tensor[-1, 0]
        }
    }

    @Test
    fun `broadcastShape computes compatible target shape`() {
        assertArrayEquals(
            intArrayOf(2, 3),
            Tensor.broadcastShape(
                intArrayOf(2, 3),
                intArrayOf(3)
            )
        )

        assertArrayEquals(
            intArrayOf(2, 3, 4),
            Tensor.broadcastShape(
                intArrayOf(2, 1, 4),
                intArrayOf(3, 4)
            )
        )

        assertArrayEquals(
            intArrayOf(2, 3, 4),
            Tensor.broadcastShape(
                intArrayOf(1, 3, 4),
                intArrayOf(2, 1, 4)
            )
        )
    }

    @Test
    fun `broadcastShape rejects incompatible shapes`() {
        assertThrows(IllegalArgumentException::class.java) {
            Tensor.broadcastShape(
                intArrayOf(2, 3),
                intArrayOf(4, 3)
            )
        }
    }

    @Test
    fun `broadcastTo returns same tensor when shape is unchanged`() {
        val tensor = Tensor(
            backedArray = doubleArrayOf(1.0, 2.0, 3.0),
            shape = intArrayOf(3),
            requiresGrad = false
        )

        val broadcasted = tensor.broadcastTo(intArrayOf(3))

        assertSame(tensor, broadcasted)
    }

    @Test
    fun `broadcastTo creates a zero stride view for broadcast dimensions`() {
        val tensor = Tensor(
            backedArray = doubleArrayOf(1.0, 2.0, 3.0),
            shape = intArrayOf(3),
            requiresGrad = false
        )

        val broadcasted = tensor.broadcastTo(intArrayOf(2, 3))

        assertArrayEquals(
            intArrayOf(0, 1),
            broadcasted.strides
        )

        assertEquals(1.0, broadcasted[0, 0])
        assertEquals(2.0, broadcasted[0, 1])
        assertEquals(3.0, broadcasted[0, 2])

        assertEquals(1.0, broadcasted[1, 0])
        assertEquals(2.0, broadcasted[1, 1])
        assertEquals(3.0, broadcasted[1, 2])
    }

    @Test
    fun `broadcastTo shares backing storage`() {
        val tensor = Tensor(
            backedArray = doubleArrayOf(1.0, 2.0, 3.0),
            shape = intArrayOf(3),
            requiresGrad = false
        )

        val broadcasted = tensor.broadcastTo(intArrayOf(2, 3))

        tensor[1] = 99.0

        assertEquals(99.0, broadcasted[0, 1])
        assertEquals(99.0, broadcasted[1, 1])
    }

    @Test
    fun `broadcastTo rejects lower rank target`() {
        val tensor = Tensor(
            backedArray = DoubleArray(6),
            shape = intArrayOf(2, 3),
            requiresGrad = false
        )

        assertThrows(IllegalArgumentException::class.java) {
            tensor.broadcastTo(intArrayOf(3))
        }
    }

    @Test
    fun `broadcast creates views with common target shape`() {
        val a = Tensor(
            backedArray = doubleArrayOf(1.0, 2.0, 3.0),
            shape = intArrayOf(3),
            requiresGrad = false
        )

        val b = Tensor(
            backedArray = doubleArrayOf(10.0, 20.0, 30.0, 40.0, 50.0, 60.0),
            shape = intArrayOf(2, 3),
            requiresGrad = false
        )

        val (aBroadcasted, bBroadcasted, shape) =
            Tensor.broadcast(a, b)

        assertArrayEquals(intArrayOf(2, 3), shape)
        assertArrayEquals(intArrayOf(2, 3), aBroadcasted.shape)
        assertArrayEquals(intArrayOf(2, 3), bBroadcasted.shape)

        assertEquals(1.0, aBroadcasted[0, 0])
        assertEquals(1.0, aBroadcasted[1, 0])
        assertEquals(3.0, aBroadcasted[0, 2])
        assertEquals(3.0, aBroadcasted[1, 2])
    }
}

