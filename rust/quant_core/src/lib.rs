//! Optional native kernels for the dipcatcher research harness.
//!
//! Safe Rust only: this crate contains no `unsafe` blocks. Order routing and
//! live-trading paths are intentionally absent.

mod book;
mod hashing;
mod returns;
mod rolling;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

fn as_slice<'a>(values: &'a PyReadonlyArray1<'a, f64>) -> PyResult<&'a [f64]> {
    values.as_slice().map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "expected a contiguous float64 buffer ({err})"
        ))
    })
}

fn check_len(flat: &[f64], rows: usize, cols: usize, name: &str) -> PyResult<()> {
    let expected = rows.saturating_mul(cols);
    if flat.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} length {got} != {rows} * {cols}",
            got = flat.len()
        )));
    }
    Ok(())
}

#[pyfunction]
fn rolling_mean<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    window: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let series = as_slice(&values)?;
    Ok(PyArray1::from_vec(
        py,
        rolling::rolling_mean_1d(series, window),
    ))
}

#[pyfunction]
fn rolling_std<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    window: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let series = as_slice(&values)?;
    Ok(PyArray1::from_vec(
        py,
        rolling::rolling_std_1d(series, window),
    ))
}

#[pyfunction]
fn rolling_mean_2d<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    rows: usize,
    cols: usize,
    window: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let flat = as_slice(&values)?;
    check_len(flat, rows, cols, "rolling_mean_2d")?;
    Ok(PyArray1::from_vec(
        py,
        rolling::rolling_rows(flat, rows, cols, window, false),
    ))
}

#[pyfunction]
fn rolling_std_2d<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    rows: usize,
    cols: usize,
    window: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let flat = as_slice(&values)?;
    check_len(flat, rows, cols, "rolling_std_2d")?;
    Ok(PyArray1::from_vec(
        py,
        rolling::rolling_rows(flat, rows, cols, window, true),
    ))
}

#[pyfunction]
fn ema<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let series = as_slice(&values)?;
    Ok(PyArray1::from_vec(py, rolling::ema(series, window)))
}

#[pyfunction]
fn rsi<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let series = as_slice(&values)?;
    Ok(PyArray1::from_vec(py, rolling::rsi(series, window)))
}

#[pyfunction]
fn bollinger<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    window: usize,
    num_sd: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let series = as_slice(&values)?;
    let [mid, upper, lower, pct_b, bandwidth] = rolling::bollinger(series, window, num_sd);
    let out = PyDict::new(py);
    out.set_item("mid", PyArray1::from_vec(py, mid))?;
    out.set_item("upper", PyArray1::from_vec(py, upper))?;
    out.set_item("lower", PyArray1::from_vec(py, lower))?;
    out.set_item("pct_b", PyArray1::from_vec(py, pct_b))?;
    out.set_item("bandwidth", PyArray1::from_vec(py, bandwidth))?;
    Ok(out)
}

#[pyfunction]
fn simple_returns<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let series = as_slice(&prices)?;
    Ok(PyArray1::from_vec(py, returns::simple_returns_1d(series)))
}

#[pyfunction]
fn simple_returns_2d<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let flat = as_slice(&prices)?;
    check_len(flat, rows, cols, "simple_returns_2d")?;
    Ok(PyArray1::from_vec(
        py,
        returns::simple_returns_2d(flat, rows, cols),
    ))
}

#[pyfunction]
fn wealth_index<'py>(
    py: Python<'py>,
    rets: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let series = as_slice(&rets)?;
    Ok(PyArray1::from_vec(py, returns::wealth_index(series)))
}

#[pyfunction]
fn turnover(weights: PyReadonlyArray1<'_, f64>, prev: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let w = as_slice(&weights)?;
    let p = as_slice(&prev)?;
    if w.len() != p.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "turnover weight length mismatch: {} vs {}",
            w.len(),
            p.len()
        )));
    }
    Ok(returns::turnover(w, p))
}

#[pyfunction]
fn turnover_series<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray1<'py, f64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let flat = as_slice(&weights)?;
    check_len(flat, rows, cols, "turnover_series")?;
    Ok(PyArray1::from_vec(
        py,
        returns::turnover_series(flat, rows, cols),
    ))
}

#[pyfunction]
fn book_features<'py>(
    py: Python<'py>,
    bid_px: PyReadonlyArray1<'py, f64>,
    bid_sz: PyReadonlyArray1<'py, f64>,
    ask_px: PyReadonlyArray1<'py, f64>,
    ask_sz: PyReadonlyArray1<'py, f64>,
    rows: usize,
    depth: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let bp = as_slice(&bid_px)?;
    let bs = as_slice(&bid_sz)?;
    let ap = as_slice(&ask_px)?;
    let az = as_slice(&ask_sz)?;
    check_len(bp, rows, depth, "bid_px")?;
    check_len(bs, rows, depth, "bid_sz")?;
    check_len(ap, rows, depth, "ask_px")?;
    check_len(az, rows, depth, "ask_sz")?;
    let cols = book::book_features(bp, bs, ap, az, rows, depth);
    let names = [
        "mid",
        "spread",
        "microprice",
        "imbalance_top",
        "imbalance_depth",
        "bid_depth",
        "ask_depth",
        "bid_log_size_slope",
        "ask_log_size_slope",
        "bid_log_price_slope",
        "ask_log_price_slope",
        "bid_mean_log_tick_spacing",
        "ask_mean_log_tick_spacing",
    ];
    let out = PyDict::new(py);
    for (name, values) in names.iter().zip(cols) {
        out.set_item(name, PyArray1::from_vec(py, values))?;
    }
    Ok(out)
}

#[pyfunction]
fn hash_bytes(data: &[u8]) -> String {
    hashing::sha256_hex(data)
}

#[pyfunction]
fn hash_many(py: Python<'_>, chunks: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    // One buffer, then hash outside the GIL. A Vec per 64-byte chunk was
    // most of the overhead against CPython's hashlib, and 3.13 closed the
    // rest of the gap. Split the hash across cores when the runner has them;
    // std threads only, no extra crate.
    let mut buf: Vec<u8> = Vec::new();
    let mut lens: Vec<usize> = Vec::new();
    for item in chunks.try_iter()? {
        let item = item?;
        let blob = item.cast::<PyBytes>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("hash_many expects a sequence of bytes")
        })?;
        let bytes = blob.as_bytes();
        buf.extend_from_slice(bytes);
        lens.push(bytes.len());
    }
    let count = lens.len();
    Ok(py.detach(move || hash_owned(&buf, &lens, count)))
}

fn hash_owned(buf: &[u8], lens: &[usize], count: usize) -> Vec<String> {
    if count == 0 {
        return Vec::new();
    }
    let mut offsets = Vec::with_capacity(count);
    let mut cursor = 0usize;
    for len in lens {
        offsets.push(cursor);
        cursor += *len;
    }
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 8);
    if workers == 1 || count < 512 {
        return (0..count)
            .map(|index| hashing::sha256_hex(&buf[offsets[index]..offsets[index] + lens[index]]))
            .collect();
    }
    let span = count.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut joins = Vec::new();
        for worker in 0..workers {
            let start = worker * span;
            if start >= count {
                break;
            }
            let end = (start + span).min(count);
            let bytes: &[u8] = buf;
            let offs: &[usize] = &offsets;
            let sizes: &[usize] = lens;
            joins.push(scope.spawn(move || {
                (start..end)
                    .map(|index| {
                        hashing::sha256_hex(&bytes[offs[index]..offs[index] + sizes[index]])
                    })
                    .collect::<Vec<_>>()
            }));
        }
        let mut out = Vec::with_capacity(count);
        for join in joins {
            out.extend(join.join().expect("hash worker"));
        }
        out
    })
}

#[pymodule]
fn quant_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rolling_mean, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_mean_2d, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std_2d, m)?)?;
    m.add_function(wrap_pyfunction!(ema, m)?)?;
    m.add_function(wrap_pyfunction!(rsi, m)?)?;
    m.add_function(wrap_pyfunction!(bollinger, m)?)?;
    m.add_function(wrap_pyfunction!(simple_returns, m)?)?;
    m.add_function(wrap_pyfunction!(simple_returns_2d, m)?)?;
    m.add_function(wrap_pyfunction!(wealth_index, m)?)?;
    m.add_function(wrap_pyfunction!(turnover, m)?)?;
    m.add_function(wrap_pyfunction!(turnover_series, m)?)?;
    m.add_function(wrap_pyfunction!(book_features, m)?)?;
    m.add_function(wrap_pyfunction!(hash_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(hash_many, m)?)?;
    Ok(())
}
