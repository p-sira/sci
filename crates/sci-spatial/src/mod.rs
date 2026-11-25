pub mod distance;
mod spatial;
pub use distance::*;
pub use spatial::*;

use crate::{Error, Result, error::is_finite_slice};

const MAX_ENTRIES: usize = 1_000_000;
const MAX_WORK: usize = 10_000_000;

fn bound(size: Option<usize>, limit: usize) -> Result<()> {
    if size.is_none_or(|n| n > limit) {
        Err(Error::InvalidInput("geometry work limit exceeded"))
    } else {
        Ok(())
    }
}

fn points(x: &[Vec<f64>]) -> Result<usize> {
    let d = x.first().map_or(0, Vec::len);
    bound(x.len().checked_mul(d.max(1)), MAX_ENTRIES)?;
    for row in x {
        if row.len() != d {
            return Err(Error::DimensionMismatch);
        }
        is_finite_slice(row)?;
    }
    Ok(d)
}

fn pair(u: &[f64], v: &[f64]) -> Result<()> {
    if u.len() != v.len() {
        return Err(Error::DimensionMismatch);
    }
    bound(Some(u.len()), MAX_ENTRIES)?;
    is_finite_slice(u)?;
    is_finite_slice(v)
}

fn evaluated(value: f64) -> Result<f64> {
    if value.is_infinite() {
        Err(Error::NonFiniteEvaluation)
    } else {
        Ok(value)
    }
}
