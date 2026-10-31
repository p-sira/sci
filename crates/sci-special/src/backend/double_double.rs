//! Shared double-double arithmetic for cancellation-sensitive series.
use core::ops::{Add, Div, Mul, Neg, Sub};
#[allow(unused_imports)]
use sci_core::Float;

#[derive(Clone, Copy)]
pub(crate) struct Dd {
    pub(crate) hi: f64,
    pub(crate) lo: f64,
}
impl From<f64> for Dd {
    fn from(hi: f64) -> Self {
        Self { hi, lo: 0.0 }
    }
}
#[inline(always)]
fn two_sum(a: f64, b: f64) -> Dd {
    let hi = a + b;
    let v = hi - a;
    Dd {
        hi,
        lo: (a - (hi - v)) + (b - v),
    }
}
/// Error-free product: `a*b = hi + lo` exactly, barring overflow or underflow.
#[inline(always)]
fn two_prod(a: f64, b: f64) -> Dd {
    let hi = a * b;
    Dd {
        hi,
        lo: a.mul_add(b, -hi),
    }
}
/// Dekker split for bounded double-double series. Callers keep |a| below
/// 1e280, so multiplying by 2^27+1 cannot overflow.
#[inline(always)]
pub(crate) fn split(a: f64) -> (f64, f64) {
    let c = 134_217_729.0 * a;
    let hi = c - (c - a);
    (hi, a - hi)
}

/// A fixed multiplier for bounded double-double series. Splitting its leading
/// part once avoids repeated FMA dispatch. Operands and intermediate products
/// must stay within the normal binary64 range and below 1e280 in magnitude.
#[derive(Clone, Copy)]
pub(crate) struct DdFactor {
    value: Dd,
    hi: f64,
    lo: f64,
}
impl DdFactor {
    #[inline]
    pub(crate) fn new(value: Dd) -> Self {
        let (hi, lo) = split(value.hi);
        Self { value, hi, lo }
    }
    #[inline(always)]
    pub(crate) fn mul(self, a: Dd) -> Dd {
        let p = a.hi * self.value.hi;
        let (hi, lo) = split(a.hi);
        let error = ((hi * self.hi - p) + hi * self.lo + lo * self.hi) + lo * self.lo;
        let lo = error + (a.hi * self.value.lo + a.lo * self.value.hi);
        let sum = p + lo;
        Dd {
            hi: sum,
            lo: lo - (sum - p),
        }
    }
}

impl Dd {
    /// Exact sum of two binary64 values.
    #[inline(always)]
    pub(crate) fn sum(a: f64, b: f64) -> Self {
        two_sum(a, b)
    }
    /// Quotient by a double-double divisor, to about 2^-104 relative
    /// accuracy. One correction step suffices because `hi - q*divisor.hi`
    /// is exact; cheaper than `/`, which forms the residual with full
    /// double-double subtraction and addition.
    #[inline(always)]
    pub(crate) fn div_dd(self, rhs: Self) -> Self {
        let q = self.hi / rhs.hi;
        let product = rhs.mul_f64(q);
        let residual = (self.hi - product.hi) + (self.lo - product.lo);
        let correction = residual / rhs.hi;
        let hi = q + correction;
        Self {
            hi,
            lo: correction - (hi - q),
        }
    }
    /// Sum with a binary64 value; cheaper than promoting it to `Dd`.
    #[inline(always)]
    pub(crate) fn add_f64(self, rhs: f64) -> Self {
        let leading = two_sum(self.hi, rhs);
        two_sum(leading.hi, leading.lo + self.lo)
    }
    /// Product with a binary64 factor; cheaper than promoting it to `Dd`.
    #[inline(always)]
    pub(crate) fn mul_f64(self, rhs: f64) -> Self {
        let product = two_prod(self.hi, rhs);
        two_sum(product.hi, product.lo + self.lo * rhs)
    }
    /// Sum of two values of the same sign. Without cancellation, one
    /// error-free addition of the leading parts suffices; cheaper than `+`.
    #[inline(always)]
    pub(crate) fn add_same_sign(self, rhs: Self) -> Self {
        let leading = two_sum(self.hi, rhs.hi);
        let hi = leading.hi;
        let lo = leading.lo + (self.lo + rhs.lo);
        let sum = hi + lo;
        Self {
            hi: sum,
            lo: lo - (sum - hi),
        }
    }
    /// Product by an exact integer of magnitude < 2^26. The split products
    /// are exact, avoiding runtime FMA dispatch. Requires |self.hi| < 1e280.
    #[inline(always)]
    pub(crate) fn mul_small(self, n: f64) -> Self {
        let p = self.hi * n;
        let (hi, lo) = split(self.hi);
        let lo = ((hi * n - p) + lo * n) + self.lo * n;
        let sum = p + lo;
        Self {
            hi: sum,
            lo: lo - (sum - p),
        }
    }
    /// Quotient by an exact nonzero integer of magnitude < 2^26.
    /// Requires |self.hi / n| < 1e280.
    #[inline(always)]
    pub(crate) fn div_small(self, n: f64) -> Self {
        let q = self.hi / n;
        let p = q * n;
        let (hi, lo) = split(q);
        let error = (hi * n - p) + lo * n;
        let correction = (((self.hi - p) - error) + self.lo) / n;
        let sum = q + correction;
        Self {
            hi: sum,
            lo: correction - (sum - q),
        }
    }
}
impl Add for Dd {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let leading = two_sum(self.hi, rhs.hi);
        let trailing = two_sum(self.lo, rhs.lo);
        let partial = two_sum(leading.hi, leading.lo + trailing.hi);
        two_sum(partial.hi, partial.lo + trailing.lo)
    }
}
impl Neg for Dd {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}
impl Sub for Dd {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}
impl Mul for Dd {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let product = two_prod(self.hi, rhs.hi);
        two_sum(product.hi, product.lo + self.hi * rhs.lo + self.lo * rhs.hi)
    }
}
impl Div for Dd {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let q = Dd::from(self.hi / rhs.hi);
        let residual = self - rhs * q;
        q + Dd::from((residual.hi + residual.lo) / rhs.hi)
    }
}
