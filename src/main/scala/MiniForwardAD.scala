import scala.math.{sin, cos, Pi}

object MiniForwardAD {

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

  given Conversion[Double, Dual] = d => Dual(d, 0.0)

  def sin(d: Dual): Dual = Dual(math.sin(d.value), math.cos(d.value) * d.deriv)
  def cos(d: Dual): Dual = Dual(math.cos(d.value), -math.sin(d.value) * d.deriv)

  def deriv(f: Dual => Dual): Double => Double = {
    (x: Double) => {
      // Seed the computation with the input value x and derivative 1 (dx/dx=1)
      val inputDual = Dual(x, 1.0)
      // Evaluate the function f using Dual number arithmetic.
      val resultDual = f(inputDual)
      // The derivative is the 'deriv' component of the result.
      resultDual.deriv
    }
  }

  def main(args: Array[String]): Unit = {
    // Example 1: f(x) = x*x
    // f'(x) = 2x
    val f1 = (x: Dual) => x * x
    val df1 = deriv(f1)
    println(s"f1(x) = x*x")
    println(s"f1(3.0)  = ${f1(Dual(3.0, 0.0)).value}") // Expected: 9.0
    println(s"f1'(3.0) = ${df1(3.0)}")               // Expected: 6.0
    println(s"f1'(5.0) = ${df1(5.0)}")               // Expected: 10.0
    println("-" * 20)

    // Example 2: f(x) = sin(x)
    // f'(x) = cos(x)
    val f2 = (x: Dual) => sin(x)
    val df2 = deriv(f2)
    println(s"f2(x) = sin(x)")
    println(s"f2(Pi/2)  = ${f2(Dual(Pi/2, 0.0)).value}") // Expected: 1.0
    println(s"f2'(Pi/2) = ${df2(Pi/2)}")               // Expected: cos(Pi/2) = 0.0
    println(s"f2'(0.0)  = ${df2(0.0)}")                // Expected: cos(0.0) = 1.0
    println("-" * 20)

    // Example 3: f(x) = 3.0 * cos(x) - x
    // f'(x) = -3.0 * sin(x) - 1
    val f3 = (x: Dual) => 3.0 * cos(x) - x // Already correct
    val df3 = deriv(f3)
    println(s"f3(x) = 3*cos(x) - x")
    val x_val = Pi
    println(s"f3($x_val)  = ${f3(Dual(x_val, 0.0)).value}") // Expected: 3*(-1) - Pi ≈ -6.14159
    println(s"f3'($x_val) = ${df3(x_val)}")               // Expected: -3*sin(Pi) - 1 = -1.0
    val x_val2 = 0.0
    println(s"f3($x_val2)  = ${f3(Dual(x_val2, 0.0)).value}") // Expected: 3*cos(0) - 0 = 3.0
    println(s"f3'($x_val2) = ${df3(x_val2)}")               // Expected: -3*sin(0) - 1 = -1.0
    println("-" * 20)

    // Example 4: Polynomial f(x) = x*x + 2*x + 1
    // f'(x) = 2x + 2
    val f4 = (x: Dual) => x * x + 2.0 * x + 1.0
    val df4 = deriv(f4)
    println(s"f4(x) = x*x + 2*x + 1")
    println(s"f4(1.0)  = ${f4(Dual(1.0, 0.0)).value}") // Expected: 4.0
    println(s"f4'(1.0) = ${df4(1.0)}")              // Expected: 4.0
    println(s"f4'(5.0) = ${df4(5.0)}")              // Expected: 12.0
    println("-" * 20)
  }
}

@main def runAD(): Unit = {
  MiniForwardAD.main(Array())
}