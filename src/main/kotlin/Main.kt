import io.github.youssefrashidy.MicroGrad
import io.github.youssefrashidy.function.reduction.Sum

fun main() {

    val x = MicroGrad.tensor(
        doubleArrayOf(2.0, 3.0),
        intArrayOf(2),
        requiresGrad = true
    )

    val y = MicroGrad.tensor(
        doubleArrayOf(4.0, 5.0),
        intArrayOf(2),
        requiresGrad = true
    )

    val a = x * y          // [8, 15]
    val b = a + x           // [10, 18]
    val c = b * y           // [40, 90]
    val d = c + a           // [48, 105]
    val e = x * x           // [4, 9]   <- x reused
    val f = e + d           // [52, 114]
    val g = y * y           // [16, 25]  <- y reused
    val h = f + g           // [68, 139]

    println("h = ${h.backedArray.contentToString()}")

    val loss = Sum(h, intArrayOf(0), false)
    println("loss = ${loss.backedArray.contentToString()}")   // expect 207.0

    loss.backward()
}