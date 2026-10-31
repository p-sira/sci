//! Shared fixture parsers and comparisons; tolerance values stay in each suite.
// Each integration test compiles this module separately and uses a subset of its helpers.
#![allow(dead_code)]

/// Call a function with fixture scalars parsed into its parameter types.
///
/// For example: `parsed_call!(sci::special::gamma::polygamma; args[0], args[1])`.
/// Parsing each argument separately preserves integer parameters and special f64 values.
#[allow(unused_macros)]
macro_rules! parsed_call {
    ($function:path; $($argument:expr),+ $(,)?) => {
        $function($($crate::common::number(&$argument)),+)
    };
}
#[allow(unused_imports)]
pub(crate) use parsed_call;

/// Parse fixture arguments and normalize a scalar result to a one-element vector.
#[allow(unused_macros)]
macro_rules! scalar_call {
    ($function:path; $($argument:expr),+ $(,)?) => {
        vec![$crate::common::parsed_call!($function; $($argument),+)]
    };
}
#[allow(unused_imports)]
pub(crate) use scalar_call;

/// Parse fixture arguments and normalize a pair result to a two-element vector.
/// The first tuple element precedes the second in the output.
#[allow(unused_macros)]
macro_rules! pair_call {
    ($function:path; $($argument:expr),+ $(,)?) => {{
        let (a, b) = $crate::common::parsed_call!($function; $($argument),+);
        vec![a, b]
    }};
}
#[allow(unused_imports)]
pub(crate) use pair_call;

/// Concatenate a pair of sequences, placing the first sequence before the second.
/// Pass the resolved pair expression, e.g. `concat_pair!(parsed_call!(f; x).unwrap())`.
#[allow(unused_macros)]
macro_rules! concat_pair {
    ($pair:expr $(,)?) => {{
        let (a, b) = $pair;
        [a, b].concat()
    }};
}
#[allow(unused_imports)]
pub(crate) use concat_pair;

use std::{fmt::Debug, str::FromStr};

/// Skip the CSV header, retaining row text for case diagnostics.
pub fn lines(csv: &str) -> impl Iterator<Item = &str> + Clone {
    csv.lines().skip(1)
}

/// Split an unquoted fixture row, preserving empty fields.
pub fn fields(row: &str) -> Vec<&str> {
    row.split(',').collect()
}

/// Read unquoted CSV rows, ignoring blank lines after the header.
pub fn rows(csv: &str) -> impl Iterator<Item = Vec<&str>> {
    lines(csv).filter(|line| !line.is_empty()).map(fields)
}

#[track_caller]
pub fn number<T: FromStr>(value: &str) -> T
where
    T::Err: Debug,
{
    value
        .parse()
        .unwrap_or_else(|error| panic!("invalid fixture value {value:?}: {error:?}"))
}

pub fn integer(value: &str) -> usize {
    number(value)
}

/// Optional scalars use the literal `none`.
pub fn optional<T: FromStr>(value: &str) -> Option<T>
where
    T::Err: Debug,
{
    (value != "none").then(|| number(value))
}

/// Vectors use semicolon-separated values; an empty field is an empty vector.
pub fn vector(value: &str) -> Vec<f64> {
    if value.is_empty() {
        vec![]
    } else {
        value.split(';').map(number).collect()
    }
}

pub fn indices(value: &str) -> Vec<usize> {
    if value.is_empty() {
        vec![]
    } else {
        value.split(';').map(number).collect()
    }
}

/// Matrices use colon-separated rows; an empty field is an empty matrix.
pub fn matrix(value: &str) -> Vec<Vec<f64>> {
    if value.is_empty() {
        vec![]
    } else {
        matrix_rows(value)
    }
}

/// Row-oriented fixtures retain an empty field as one empty row.
pub fn matrix_rows(value: &str) -> Vec<Vec<f64>> {
    value.split(':').map(vector).collect()
}

/// Parse a CSV row with quoted fields and doubled quote escaping.
pub fn csv_row(row: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = row.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' if quoted && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                fields.push(std::mem::take(&mut field));
            }
            _ => field.push(ch),
        }
    }
    assert!(!quoted);
    fields.push(field);
    fields
}

/// Compare using the suite's existing absolute-plus-relative reference bound.
/// `approx` uses the supplied absolute bound without a symmetric relative scale.
/// NaNs match only NaNs; infinities must have the same sign. Signed zero is
/// compared explicitly by suites where its sign is part of the contract.
pub fn matches(actual: f64, expected: f64, atol: f64, rtol: f64) -> bool {
    if expected.is_nan() {
        actual.is_nan()
    } else if expected.is_infinite() {
        actual == expected
    } else {
        actual.is_finite()
            && approx::abs_diff_eq!(actual, expected, epsilon = atol + rtol * expected.abs())
    }
}

#[track_caller]
pub fn assert_close(actual: f64, expected: f64, atol: f64, rtol: f64, context: &str) {
    assert!(
        matches(actual, expected, atol, rtol),
        "{context}: expected {expected:.17e}, got {actual:.17e}; atol={atol:e}, rtol={rtol:e}"
    );
}

/// Compare values using an absolute-plus-relative reference bound.
#[track_caller]
pub fn close(actual: f64, expected: f64, atol: f64, rtol: f64) {
    assert_close(actual, expected, atol, rtol, "numerical comparison");
}

/// Compare finite values, rejecting matching NaNs and infinities as well.
#[track_caller]
pub fn finite_close(actual: f64, expected: f64, atol: f64, rtol: f64) {
    assert!(
        actual.is_finite() && expected.is_finite(),
        "expected finite values, got {actual} and {expected}"
    );
    close(actual, expected, atol, rtol);
}

/// Compare with a relative bound floored at the smallest positive normal value.
#[track_caller]
pub fn relative_close(actual: f64, expected: f64, rtol: f64) {
    close(
        actual,
        expected,
        rtol * expected.abs().max(f64::MIN_POSITIVE),
        0.0,
    );
}

/// Compare matrix shapes and entries with an absolute-plus-relative bound.
#[track_caller]
pub fn matrix_close(actual: &[Vec<f64>], expected: &[Vec<f64>], atol: f64, rtol: f64) {
    assert_eq!(actual.len(), expected.len(), "row count");
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.len(), expected.len(), "column count");
        for (&actual, &expected) in actual.iter().zip(expected) {
            finite_close(actual, expected, atol, rtol);
        }
    }
}

/// Preserve nonzero references, using only relative error below `small`.
/// The bound is floored at the smallest subnormal value for tiny references.
/// Larger references use `tolerance * (1 + abs(expected))`; NaNs and signed
/// infinities follow `assert_close`.
#[track_caller]
pub fn assert_nonzero_close(actual: f64, expected: f64, tolerance: f64, small: f64, context: &str) {
    let bound = if expected != 0.0 && expected.abs() < small {
        (tolerance * expected.abs()).max(f64::from_bits(1))
    } else {
        tolerance + tolerance * expected.abs()
    };
    if expected.is_finite() && expected != 0.0 {
        assert_ne!(
            actual, 0.0,
            "{context}: nonzero reference underflowed to zero"
        );
    }
    assert_close(actual, expected, bound, 0.0, context);
}
