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
fn two_sum(a: f64, b: f64) -> Dd {
    let hi = a + b;
    let v = hi - a;
    Dd {
        hi,
        lo: (a - (hi - v)) + (b - v),
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
        let product = self.hi * rhs.hi;
        let error = self.hi.mul_add(rhs.hi, -product) + self.hi * rhs.lo + self.lo * rhs.hi;
        two_sum(product, error)
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
