#![allow(non_upper_case_globals)]

mod data;
use crate::{Error, Result};
pub use data::*;

/// One entry in the CODATA physical-constant table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicalConstant {
    /// Exact lookup key used by SciPy.
    pub name: &'static str,
    /// Numerical value in the specified SI unit.
    pub value: f64,
    /// SI unit string; empty for dimensionless quantities.
    pub unit: &'static str,
    /// Absolute standard uncertainty; zero for exact constants.
    pub uncertainty: f64,
}

fn lookup(key: &str) -> Result<&'static PhysicalConstant> {
    PHYSICAL_CONSTANTS
        .binary_search_by_key(&key, |entry| entry.name)
        .map(|index| &PHYSICAL_CONSTANTS[index])
        .map_err(|_| Error::InvalidInput("unknown physical constant"))
}

/// Looks up the numerical value of a physical constant.
///
/// ## Parameters
///
/// `key` is an exact, case-sensitive SciPy/CODATA table name.
/// ## Returns
///
/// The pinned CODATA value as `f64`, in the unit returned by [`unit()`].
/// ## Errors
///
/// Unknown keys return [`Error::InvalidInput`].
/// ## Numerical behavior
///
/// Decimal database values are rounded to `f64`; uncertainties are preserved.
/// # Examples
///
/// ```
/// assert_eq!(sci_constants::value("speed of light in vacuum")?, 299792458.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy constants](https://docs.scipy.org/doc/scipy/reference/constants.html).
pub fn value(key: &str) -> Result<f64> {
    Ok(lookup(key)?.value)
}

/// Looks up the SI unit of a physical constant.
///
/// ## Parameters
///
/// `key` is an exact case-sensitive table name.
/// ## Returns
///
/// A static unit string; an empty string means dimensionless.
/// ## Errors
///
/// Unknown keys return [`Error::InvalidInput`].
/// # Examples
///
/// ```
/// assert_eq!(sci_constants::unit("speed of light in vacuum")?, "m s^-1");
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy constants](https://docs.scipy.org/doc/scipy/reference/constants.html).
pub fn unit(key: &str) -> Result<&'static str> {
    Ok(lookup(key)?.unit)
}

/// Returns relative standard uncertainty `u / value` for a physical constant.
///
/// ## Parameters
///
/// `key` is an exact case-sensitive table name; `u` is absolute uncertainty.
/// ## Returns
///
/// The dimensionless SciPy precision convention, zero for exact constants.
/// ## Errors
///
/// Unknown keys return [`Error::InvalidInput`].
/// ## Numerical behavior
///
/// Uses `f64` division of the stored CODATA values.
/// # Examples
///
/// ```
/// assert_eq!(sci_constants::precision("speed of light in vacuum")?, 0.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy precision](https://docs.scipy.org/doc/scipy/reference/generated/scipy.constants.precision.html).
pub fn precision(key: &str) -> Result<f64> {
    let entry = lookup(key)?;
    Ok(entry.uncertainty / entry.value)
}

/// Finds constant names containing a case-insensitive substring.
///
/// ## Parameters
///
/// `None` returns all current keys. SciPy historical keys absent from its
/// current CODATA set are excluded, matching SciPy's `find` behavior.
/// ## Returns
///
/// Alphabetically sorted static names. No match gives an empty vector.
/// # Examples
///
/// ```
/// assert!(sci_constants::find(Some("speed of light")).contains(&"speed of light in vacuum"));
/// ```
/// # References
///
/// [SciPy find](https://docs.scipy.org/doc/scipy/reference/generated/scipy.constants.find.html).
pub fn find(substring: Option<&str>) -> Vec<&'static str> {
    let substring = substring.unwrap_or("").to_lowercase();
    CURRENT_NAMES
        .iter()
        .copied()
        .filter(|name| name.to_lowercase().contains(&substring))
        .collect()
}

/// Temperature scales accepted by [`convert_temperature`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TemperatureScale {
    /// Celsius: `K = C + 273.15`.
    Celsius,
    /// Kelvin, the SI absolute temperature scale.
    Kelvin,
    /// Fahrenheit: `K = (F - 32) * 5/9 + 273.15`.
    Fahrenheit,
    /// Rankine: `K = R * 5/9`.
    Rankine,
}

/// Converts temperature by an affine transformation through kelvin.
///
/// ## Parameters
///
/// `temperature` uses `from` units; the result uses `to` units. Scales use
/// an enum instead of SciPy's string names and aliases.
/// ## Domain
///
/// All `f64` values are accepted, including negative absolute temperatures.
/// NaN and infinities propagate, as in SciPy.
/// ## Returns
///
/// Converted scalar temperature.
/// ## Numerical behavior
///
/// Two affine `f64` transformations; identical scales return the input exactly.
/// # Examples
///
/// ```
/// use sci_constants::{convert_temperature, TemperatureScale::*};
/// assert_eq!(convert_temperature(0.0, Celsius, Kelvin), 273.15);
/// ```
/// # References
///
/// [SciPy convert_temperature](https://docs.scipy.org/doc/scipy/reference/generated/scipy.constants.convert_temperature.html).
pub fn convert_temperature(temperature: f64, from: TemperatureScale, to: TemperatureScale) -> f64 {
    use TemperatureScale::*;
    if from == to {
        return temperature;
    }
    let kelvin = match from {
        Celsius => temperature + zero_Celsius,
        Kelvin => temperature,
        Fahrenheit => (temperature - 32.0) * (5.0 / 9.0) + zero_Celsius,
        Rankine => temperature * (5.0 / 9.0),
    };
    match to {
        Celsius => kelvin - zero_Celsius,
        Kelvin => kelvin,
        Fahrenheit => (kelvin - zero_Celsius) * (9.0 / 5.0) + 32.0,
        Rankine => kelvin * (9.0 / 5.0),
    }
}

/// Converts wavelength in metres to frequency in hertz: `ν = c / λ`.
///
/// ## Parameters
///
/// `wavelength` is a scalar vacuum wavelength in metres.
/// ## Domain
///
/// IEEE division applies: signed zero gives signed infinity, infinity gives
/// signed zero, NaN propagates, and negative inputs give negative frequencies.
/// ## Returns
///
/// Frequency in hertz.
/// ## Numerical behavior
///
/// One `f64` division using the exact SI speed of light.
/// # Examples
///
/// ```
/// assert_eq!(sci_constants::lambda2nu(1.0), 299792458.0);
/// ```
/// # References
///
/// [SciPy lambda2nu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.constants.lambda2nu.html).
pub fn lambda2nu(wavelength: f64) -> f64 {
    c / wavelength
}

/// Converts frequency in hertz to wavelength in metres: `λ = c / ν`.
///
/// ## Parameters
///
/// `frequency` is a scalar vacuum frequency in hertz.
/// ## Domain
///
/// IEEE division applies: signed zero gives signed infinity, infinity gives
/// signed zero, NaN propagates, and negative inputs give negative wavelengths.
/// ## Returns
///
/// Wavelength in metres.
/// ## Numerical behavior
///
/// One `f64` division using the exact SI speed of light.
/// # Examples
///
/// ```
/// assert_eq!(sci_constants::nu2lambda(299792458.0), 1.0);
/// ```
/// # References
///
/// [SciPy nu2lambda](https://docs.scipy.org/doc/scipy/reference/generated/scipy.constants.nu2lambda.html).
pub fn nu2lambda(frequency: f64) -> f64 {
    c / frequency
}
