//! Parsers for checked-in SciPy fixtures; numerical comparison policies stay in each suite.
// Each integration test compiles this module separately and uses a subset of its helpers.
#![allow(dead_code)]

/// Call a function with fixture scalars parsed into its parameter types.
///
/// For example: `parsed_call!(sci::special::polygamma; args[0], args[1])`.
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
