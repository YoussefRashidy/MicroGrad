# MicroGrad — Kotlin

A small, from-scratch **reverse-mode automatic differentiation engine for n-dimensional tensors**, written entirely in Kotlin.

MicroGrad combines a **strided tensor system**, **NumPy-style broadcasting**, **zero-copy views**, and a **define-by-run computation graph** to provide automatic differentiation similar in spirit to PyTorch and micrograd.

The project is intentionally small and explicit: tensors, operations, broadcasting, graph construction, and backpropagation are implemented from the ground up rather than delegated to an existing numerical or autograd library.

> Package root: `io.github.youssefrashidy`

## Features

- **N-dimensional strided tensors** backed by flat `DoubleArray` storage
- **Reverse-mode automatic differentiation**
- **Define-by-run computation graphs**
- **Gradient accumulation** across shared subgraphs
- **NumPy-style broadcasting**
- **Broadcasting-aware backpropagation**
- **Zero-copy tensor views** for reshape, permutation, transpose, squeeze, and unsqueeze
- **Matrix multiplication** with batched dimensions and broadcasting
- **Reduction operations** including `sum`, `mean`, `max`, and `min`
- **Mathematical operations** including `exp`, `log`, and arbitrary-power operations
- **Operator overloading** for natural tensor expressions
- **Mixed scalar/tensor arithmetic** with `Int`, `Long`, `Float`, and `Double`
- **`noGrad` execution context**
- **Optional graph retention** for repeated backward passes
- Random tensor initialization with uniform and Gaussian distributions

## Quick Start

A simple scalar loss can be built directly from tensor operations:

```kotlin
import io.github.youssefrashidy.MicroGrad
import io.github.youssefrashidy.function.reduction.Sum
import io.github.youssefrashidy.function.linalg.MatMul

fun main() {
    val x = MicroGrad.randn(
        shape = intArrayOf(2, 3),
        requiresGrad = true
    )

    val w = MicroGrad.randn(
        shape = intArrayOf(3, 4),
        requiresGrad = true
    )

    val bias = MicroGrad.ones(
        shape = intArrayOf(4),
        requiresGrad = true
    )

    // Matrix multiplication
    val logits = MatMul(x, w)

    // Broadcasting: [2, 4] + [4]
    val shifted = logits + bias

    // Nonlinear transformation
    val activated = shifted.exp()

    // Reduction to a scalar loss
    val loss = Sum(
        activated,
        axes = intArrayOf(0, 1),
        keepDims = false
    )

    loss.backward()

    println("loss = ${loss[0]}")
    println("dx = ${x.grad?.backedArray?.contentToString()}")
    println("dw = ${w.grad?.backedArray?.contentToString()}")
    println("db = ${bias.grad?.backedArray?.contentToString()}")
}
```

This small example exercises several parts of the engine at once:

**matrix multiplication → broadcasting → elementwise operation → reduction → reverse-mode backpropagation**

MicroGrad also supports ordinary tensor expressions through operator overloading:

```kotlin
val y = ((x * x) + 2.0 * x - 1.0).exp()
val loss = Sum(y, intArrayOf(0), keepDims = false)

loss.backward()
```

## How It Works

MicroGrad uses a **define-by-run computation graph**.

Each differentiable operation produces a `Tensor` and records the information required to propagate gradients later:

```text
Input tensors
     │
     ▼
  Operation
     │
     ▼
 Output tensor
     │
     ├── prevTensors
     └── gradFn
```

When `backward()` is called, MicroGrad:

1. Starts from the scalar loss.
2. Traverses `prevTensors` to construct a topological ordering of the graph.
3. Seeds the loss gradient with `1`.
4. Walks the graph in reverse topological order.
5. Executes each node's gradient function.
6. Accumulates the resulting gradients into its operands.

This allows shared tensors to participate in multiple branches of a computation graph while correctly accumulating their gradients.

For example:

```kotlin
val y = x * x + x
```

contains multiple paths from `x` to `y`. MicroGrad accumulates the contribution from each path rather than replacing the previous gradient.

## Tensor Representation

`Tensor` uses a flat `DoubleArray` together with a shape and stride description:

```text
Tensor
├── data      → DoubleArray
├── shape     → IntArray
└── strides   → IntArray
```

This representation makes it possible to implement tensor views without copying the underlying data.

Operations such as:

```text
reshape
permute
transpose
squeeze
unsqueeze
broadcastTo
```

can therefore return tensors that reference the same storage.

### Broadcasting

MicroGrad implements NumPy-style broadcasting.

For example:

```text
A: [2, 3]
B: [3]

A + B → [2, 3]
```

Broadcasting is represented using **stride-0 dimensions**, allowing the broadcasted tensor to remain a view rather than allocating a new buffer.

The backward pass performs the corresponding reduction back to the original operand shape so that broadcasting is handled correctly during gradient accumulation.

## Automatic Differentiation

Every differentiable operation records:

```text
prevTensors
gradFn
```

The forward pass computes the output, while the associated gradient function describes how the output gradient should be propagated to its inputs.

Conceptually:

```text
             forward
A ───────────────► B
                   │
                   │ gradFn
                   ▼
             backward
```

Gradient accumulation is performed in-place on the operand's gradient buffer, allowing gradients from multiple downstream paths to be combined naturally.

## Operations

### Elementwise Operations

```text
+
-
*
/
unary -
```

All support broadcasting and scalar/tensor arithmetic.

Examples:

```kotlin
val y1 = x + 2.0
val y2 = 3.0 * x
val y3 = x / 4
val y4 = -x
```

### Mathematical Operations

```text
Exp
Log
Power
```

Examples:

```kotlin
val y = x.exp()
val y = x.log()
val y = x.pow(3.0)
```

`Power` supports an arbitrary `Double` exponent.

### Reductions

```text
Sum
Mean
Max
Min
```

Reductions accept one or more axes and support `keepDims`.

```kotlin
val total = Sum(x, intArrayOf(0, 1), keepDims = false)
val mean = Mean(x, intArrayOf(1), keepDims = true)
```

For `Max` and `Min`, the backward pass routes gradients only to the selected extremum.

### Structural Operations

```text
Reshape
Permute
Transpose
Squeeze
UnSqueeze
```

These operations are implemented using tensor views where possible, avoiding unnecessary copies.

### Linear Algebra

```text
MatMul
```

`MatMul` supports:

- vector × vector dot products
- matrix × matrix multiplication
- batched matrix multiplication
- broadcasting of batch dimensions

For example:

```text
A: [2, 3]
B: [3, 4]

A @ B → [2, 4]
```

and higher-rank tensors can use broadcasted batch dimensions.

## API Design

Operations are represented as `Function` objects.

Each operation follows the same general structure:

```text
Function
   │
   ├── forward(...)
   └── backward(...)
```

The library uses singleton operation objects where the operation itself is stateless:

```kotlin
Addition(...)
Multiplication(...)
Exp(...)
Sum(...)
MatMul(...)
```

`FunctionInput` is a sealed hierarchy used to provide operation-specific metadata while keeping the `Function` interface uniform.

Common gradient accumulation logic is centralized in `Function`, allowing individual operations to focus on their local derivatives.

## No-Grad Mode

Computations can be executed without constructing an autograd graph:

```kotlin
MicroGrad.noGrad {
    val y = model(input)
}
```

This is useful for inference and other computations where gradients are not required.

## Project Structure

```text
src/
├── main/
│   └── kotlin/
│       ├── Main.kt
│       ├── MicroGrad.kt
│       │
│       ├── tensor/
│       │   ├── Tensor.kt
│       │   └── TensorOperatorExtensions.kt
│       │
│       └── function/
│           ├── Function.kt
│           ├── FunctionInput.kt
│           ├── BackwardFunction.kt
│           │
│           ├── operators/
│           │   ├── Addition.kt
│           │   ├── Subtraction.kt
│           │   ├── Multiplication.kt
│           │   └── Division.kt
│           │
│           ├── math/
│           │   ├── Exp.kt
│           │   ├── Log.kt
│           │   └── Power.kt
│           │
│           ├── reduction/
│           │   ├── Sum.kt
│           │   ├── Mean.kt
│           │   ├── Max.kt
│           │   └── Min.kt
│           │
│           ├── structural/
│           │   ├── Reshape.kt
│           │   ├── Permute.kt
│           │   ├── Transpose.kt
│           │   ├── Squeeze.kt
│           │   └── UnSqueeze.kt
│           │
│           └── linalg/
│               └── MatMul.kt
│
└── test/
    └── kotlin/
        └── ...
```

## Testing

The project contains unit tests covering the tensor implementation and individual operations, including:

```text
Tensor
Addition
Subtraction
Multiplication
Division
Exp
Log
Power
Sum
Mean
Max
Min
Reshape
Permute
MatMul
```

The tests cover forward computation as well as gradient behavior for supported operations.

## Building

MicroGrad is a Kotlin/JVM project and can be built using the standard Kotlin build ecosystem.

With Gradle:

```bash
./gradlew build
```

Run the test suite:

```bash
./gradlew test
```

Run the example application:

```bash
./gradlew run
```

## Design Goals

MicroGrad is primarily a learning and experimentation project.

The goal is not to reproduce the scale or performance of frameworks such as PyTorch. Instead, the project focuses on making the fundamental mechanisms behind tensor computation and automatic differentiation explicit:

```text
Tensor representation
        ↓
Broadcasting & views
        ↓
Operations
        ↓
Computation graph
        ↓
Topological traversal
        ↓
Reverse-mode differentiation
        ↓
Gradient accumulation
```

The implementation is intentionally small enough to explore these mechanisms directly while still supporting many of the features required by a practical tensor/autograd system.

## License

This project is licensed under the [MIT License](LICENSE).