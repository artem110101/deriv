# Simple Automatic Differentiation in Scala 3 with Dual Numbers

Recently, I had an opportunity to finally catch up on one of the most fascinating technologies I've come across: [JAX](https://docs.jax.dev/en/latest/).

*JAX* has an incredibly friendly user-facing API, which packs significant power and is a state-of-the-art accelerator for array processing in Python.

Interestingly, the JAX team has put an incredible amount of effort to make JAX contributor-friendly: they've create [Autodidax: JAX core from scratch](Autodidax: JAX core from scratch) guide.

# Background

The ability to compute the derivative (i.e., the gradient) of the loss function with respect to a model’s parameters is at the core of deep learning. By feeding these gradients backwards through the network (backpropagation), the model can iteratively adjust its parameters and ultimately learn from data.

Today, I want to explore the simplest possible way to do automatic differentiation in Scala 3 using *Dual Numbers*. While JAX comes with incredible powerful forward and reverse modes, we will focus on simplicity for today.

# Code

While Python would be a natural fit for an example like this, I would like to explore a Scala 3 code sample with operator overloading.

Let's define simple case class that follows *Dual Numbers* strategy: (value, derivative)

```
final case class Dual(value: Double, deriv: Double) {
  def +(that: Dual): Dual = Dual(this.value + that.value, this.deriv + that.deriv)
  def -(that: Dual): Dual = Dual(this.value - that.value, this.deriv - that.deriv)
  def *(that: Dual): Dual = Dual(
    this.value * that.value,
    this.deriv * that.value + this.value * that.deriv
  )
  def +(c: Double): Dual = this + Dual(c, 0.0)
  def -(c: Double): Dual = this - Dual(c, 0.0)
  def *(c: Double): Dual = this * Dual(c, 0.0)
  def unary_- : Dual = Dual(-this.value, -this.deriv)
}
```
Now, let's define few more functions that we need

```
given Conversion[Double, Dual] = d => Dual(d, 0.0)

def sin(d: Dual): Dual = Dual(math.sin(d.value), math.cos(d.value) * d.deriv)
def cos(d: Dual): Dual = Dual(math.cos(d.value), -math.sin(d.value) * d.deriv)

def deriv(f: Dual => Dual): Double => Double = {
  (x: Double) =>
    // Seed the computation with the input value x and derivative 1 (dx/dx=1)
    val inputDual = Dual(x, 1.0)

    // Evaluate the function f using Dual number arithmetic.
    val resultDual = f(inputDual)

    // The derivative is the 'deriv' component of the result.
    resultDual.deriv
}
```
Now let's run few examples:

## Example 1: `f(x) = x * x`
```
val f1 = (x: Dual) => x * x
val df1 = deriv(f1)
println(s"f1(3.0)  = ${f1(Dual(3.0, 0.0)).value}")
println(s"f1'(3.0) = ${df1(3.0)}")
println(s"f1'(5.0) = ${df1(5.0)}")
```
*Output*

```
f1(3.0)  = 9.0
f1'(3.0) = 6.0
f1'(5.0) = 10.0
```

## Example 2: `f(x) = sin(x)*`
```
val f2 = (x: Dual) => sin(x)
val df2 = deriv(f2)
println(s"f2(Pi/2)  = ${f2(Dual(Pi/2, 0.0)).value}")
println(s"f2'(Pi/2) = ${df2(Pi/2)}")
println(s"f2'(0.0)  = ${df2(0.0)}")
```
*Output*

```
f2(Pi/2)  = 1.0
f2'(Pi/2) = 6.123233995736766E-17
f2'(0.0)  = 1.0
```

## Example 3: `f(x) = 3.0 * cos(x) - x`

```
val f3 = (x: Dual) => 3.0 * cos(x) - x
val df3 = deriv(f3)
val x_val = Pi
println(s"f3($x_val)  = ${f3(Dual(x_val, 0.0)).value}")
println(s"f3'($x_val) = ${df3(x_val)}")
val x_val2 = 0.0
println(s"f3($x_val2)  = ${f3(Dual(x_val2, 0.0)).value}")
println(s"f3'($x_val2) = ${df3(x_val2)}")
```
*Output*
```
f3(3.141592653589793)  = -6.141592653589793
f3'(3.141592653589793) = -1.0000000000000004
f3(0.0)  = 3.0
f3'(0.0) = -1.0
```

## Example 4: `f(x) = x*x + 2*x + 1`
```
val f4 = (x: Dual) => x * x + 2.0 * x + 1.0
val df4 = deriv(f4)
println(s"f4(1.0)  = ${f4(Dual(1.0, 0.0)).value}")
println(s"f4'(1.0) = ${df4(1.0)}")
println(s"f4'(5.0) = ${df4(5.0)}")
```
*Output*
```
f4(1.0) = 4.0
f4'(1.0) = 4.0
f4'(5.0) = 12.0
```

While this is the simplest possible example of a forward pass, using scalar (single) numbers for clarity, real-world applications, especially in deep learning using libraries like JAX, typically involve operations on vectors, matrices, and higher-dimensional arrays (tensors). The underlying automatic differentiation principles demonstrated here extend to these structures, and efficient libraries like JAX leverage vectorized APIs to perform calculations across entire arrays. Furthermore, beyond this simple forward mode, JAX implements incredibly powerful techniques to support the reverse mode (often more efficient for typical deep learning gradient calculations), which we will explore in the next article.

References:
 - [Deep Learning with JAX](https://www.manning.com/books/deep-learning-with-jax) (by Grigory Sapunov)
 - [Autodidax: JAX core from scratch](https://docs.jax.dev/en/latest/autodidax.html) (by JAX team)
 - [Deep Learning with Python, Third Edition](https://www.manning.com/books/deep-learning-with-python-third-edition)[https://www.manning.com/books/deep-learning-with-python-third-edition] (by François Chollet)