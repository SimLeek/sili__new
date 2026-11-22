import Base

# LAWS: why sili's fp4 weight + fp4 importance can train (with stochastic
# rounding), and how much larger an fp4 model must be to hold the same weights
# exactly. See README.md for the setting (sili__new fp4quant.hpp,
# delta_csr_types.hpp synapse policies) and for what is assumed, not proved.

# LAW 1: stochastic rounding is unbiased on one grid cell.
# The cell is [a, a + w], w = d + e, and x = a + d sits inside it. Stochastic
# rounding goes down to a with probability e/w and up to a + w with
# probability d/w. Times w, the expectation is a*e + (a + w)*d, which equals
# x*w: E[round(x)] = x. So an update far smaller than the fp4 spacing still
# moves the stored weight by exactly that update on average.
law sr_unbiased:
  for +a: Nat
  for +d: Nat
  for +e: Nat
  {Nat.add(Nat.mul(a, e), Nat.mul(Nat.add(a, Nat.add(d, e)), d)) == Nat.mul(Nat.add(a, d), Nat.add(d, e)) : Nat}

# LAW 1b: the rounding noise. The error is -d with probability e/w and +e
# with probability d/w, so w*E[err^2] = e*d^2 + d*e^2 = (d*e)*w: the variance is
# d*e, at most w^2/4 (largest at the cell centre). Zero-mean (LAW 1) plus
# bounded variance is what SGD needs: training stays unbiased, only noisier.
law sr_second_moment:
  for +d: Nat
  for +e: Nat
  {Nat.add(Nat.mul(e, Nat.mul(d, d)), Nat.mul(d, Nat.mul(e, e))) == Nat.mul(Nat.mul(d, e), Nat.add(d, e)) : Nat}

# Base-5 digits. Each digit value 0..4 is exactly representable in E2M1
# (fp4 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 times a scale of 1/2 or 1).
type Digit is Data:
  D0{}
  D1{}
  D2{}
  D3{}
  D4{}

def Fp4.digit(d: Digit) -> Nat:
  match d:
    case D0{}:
      0n
    case D1{}:
      1n
    case D2{}:
      2n
    case D3{}:
      3n
    case D4{}:
      4n

# value([d0, d1, ...]) = d0 + 5*(d1 + 5*(...)): fp4 synapse j carries digit dj
# and its row scale is 5^j (one rank-1 scale channel, S = 5^j).
def Fp4.value(xs: +List<Digit>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case Con{h, t}:
      Nat.add(Fp4.digit(h), Nat.mul(5n, Fp4.value(t)))

def Fp4.inc(xs: +List<Digit>) -> +List<Digit>:
  match xs:
    case Nil{}:
      D1{} <> Nil{}
    case Con{h, t}:
      match h:
        case D0{}:
          D1{} <> t
        case D1{}:
          D2{} <> t
        case D2{}:
          D3{} <> t
        case D3{}:
          D4{} <> t
        case D4{}:
          D0{} <> Fp4.inc(t)

def Fp4.encode(n: Nat) -> +List<Digit>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      Fp4.inc(Fp4.encode(p))

# LAW 2: incrementing the digit list adds exactly one (carry is exact).
law inc_value:
  for +xs: +List<Digit>
  {Fp4.value(Fp4.inc(xs)) == 1n+Fp4.value(xs) : Nat}

# LAW 3, THE SIZE BOUND: every natural number n is exactly the value of a list
# of fp4-representable digits, so any integer weight (any b-bit fixed-point
# weight, sign handled by a mirrored copy) is exactly a sum of fp4 synapses
# with row scales 5^j. b bits need ceil(b / log2(5)) = ceil(b / 2.32) synapses.
law encode_value:
  for +n: Nat
  {Fp4.value(Fp4.encode(n)) == n : Nat}
