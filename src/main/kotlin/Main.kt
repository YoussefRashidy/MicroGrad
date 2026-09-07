import io.github.youssefrashidy.MicroGrad
import io.github.youssefrashidy.function.math.Exp
import io.github.youssefrashidy.function.math.Log
import io.github.youssefrashidy.function.math.Power
import io.github.youssefrashidy.function.linalg.MatMul
import io.github.youssefrashidy.function.reduction.Sum
import io.github.youssefrashidy.function.reduction.Mean
import io.github.youssefrashidy.tensor.*   // pulls in the Int/Double <-> Tensor operators

fun main() {
    // 2 samples, 3 features
    val X = MicroGrad.tensor(doubleArrayOf(1.0, 2.0, -1.0, 0.0, 1.0, 3.0), intArrayOf(2, 3))
    val y = MicroGrad.tensor(doubleArrayOf(1.0, 0.0), intArrayOf(2, 1))

    val w = MicroGrad.tensor(doubleArrayOf(0.5, -0.2, 0.1), intArrayOf(3, 1), requiresGrad = true)
    val b = MicroGrad.scalar(0.2, requiresGrad = true)

    // --- forward ---
    val z = MatMul(X, w) + b                       // (2,1); bias broadcasts over the batch
    val p = 1 / (1 + Exp(-z))                       // sigmoid, built from primitives
    val bce = -(y * Log(p) + (1 - y) * Log(1 - p))  // elementwise BCE, shape (2,1)
    val meanLoss = Mean(bce, intArrayOf(0), false)  // reduce over the batch axis

    val lambda = 0.1
    val reg = lambda * Sum(Power(w, 2.0), intArrayOf(0, 1), false)  // w reused here too

    val total = meanLoss + reg
    println("loss = ${total.backedArray[0]}")

    total.backward()
    println("w.grad = ${w.grad?.backedArray?.contentToString()}")
    println("b.grad = ${b.grad?.backedArray?.contentToString()}")
}