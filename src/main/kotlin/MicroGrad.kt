package io.github.youssefrashidy

object MicroGrad {
    var gradEnabled : Boolean = true

    fun noGrad(block : ()->Unit){
        val previousGrad = gradEnabled
        gradEnabled = false
        try {
            block()
        } finally {
            gradEnabled = previousGrad
        }
    }
}