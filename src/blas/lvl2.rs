//! Level 2 BLAS: matrix-vector operations
use super::lvl1;
use super::{Diag, Layout, Transpose, UpLo};
use num::complex::{Complex32, Complex64, ComplexFloat};
use num::{Float, One, Zero};
use std::ops::{AddAssign, DivAssign, Mul, MulAssign, SubAssign};

/// Performs the operation:
///  * NoTrans    - y := alpha * A  * x + beta * y
///  * Trans      - y := alpha * A' * x + beta * y
///  * ConjTrans  - y := alpha * conj(A') * x + beta * y
/// Where `A` is a general band matrix
///
/// # Arguments
///  * `layout` - the layout of the matrix `A`
///  * `trans` - whether the matrix should be transposed and/or conjugated
///  * `m` - the number of rows in the `A` matrix
///  * `n` - the number of columns in the `A` matrix
///  * `kl` - the number of sub-diagonals of the matrix `A`
///  * `ku` - the number of super-diagonals of the matrix `A`
///  * `alpha` - the value of the scalar alpha
///  * `a` - array of size `lda * n`, in _compressed form_ (see below for details)
///  * `lda`- the leading dimension of `a`
///  * `x` - array, of size at least (1 + (n - 1) * incx) (NoTrans) and at least
///          (1 + (m - 1) * incx) (Trans/TransConjugate)
///  * `incx` - the increment for the elements of `x`
///  * `beta` - the scalar beta
///  * `y` - the output array, size at least (1 + (m - 1) * incy) (NoTrans) and at least
///          (1 + (n - 1) * incy) otherwise
///  * `incy` - the increments for the elements of `y`
///
/// # Compressed form
/// TODO
pub fn real_gbmv<T>(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    x: &[T],
    incx: usize,
    beta: T,
    y: &mut [T],
    incy: usize,
) where
    T: Float + lvl1::Scal + lvl1::Dot + lvl1::Axpy + AddAssign<T>,
{
    debug_assert!(lda >= kl + ku + 1);
    debug_assert!(incy > 0, "incy must be non-zero");
    let (ny, nx) = match trans {
        Transpose::NoTrans => (m, n),
        _ => (n, m),
    };
    if nx.is_zero() || ny.is_zero() {
        return;
    }
    debug_assert!(
        x.len() >= (1 + (nx - 1) * incx).max(1),
        "x must have size at least {}",
        (1 + (nx - 1) * incx).max(1)
    );
    debug_assert!(
        y.len() >= 1 + (ny - 1) * incy,
        "y must have size at least {}",
        1 + (ny - 1) * incy
    );
    debug_assert!(
        match layout {
            Layout::RowMajor => lda >= n.max(1),
            Layout::ColMajor => lda >= m.max(1),
        },
        "A has inconsistent lead dimension `lda`"
    );
    debug_assert!(
        match layout {
            Layout::RowMajor => a.len() >= lda * m,
            Layout::ColMajor => a.len() >= lda * n,
        },
        "A has insufficient size"
    );
    if ny.is_zero() {
        return;
    }
    if !beta.is_one() {
        T::scal(m, beta, y, incy);
    }
    if nx == 0 || alpha.is_zero() {
        return;
    }
    match (layout, trans) {
        (Layout::ColMajor, Transpose::NoTrans)
        | (Layout::RowMajor, Transpose::Trans)
        | (Layout::RowMajor, Transpose::ConjTrans) => {
            let mut ia = ku;
            let mut ix = 0;
            let mut iy = 0;
            let mut ndiag = m - ku;
            for _ in 0..ku {
                T::axpy(ndiag, alpha * x[ix], &a[ia..], 1, &mut y[iy..], incy);
                ndiag += 1;
                ia += lda - 1;
                ix += incx;
            }
            for _ in ku..5 {
                T::axpy(ndiag, alpha * x[ix], &a[ia..], 1, &mut y[iy..], incy);
                ndiag -= 1;
                ia += lda;
                ix += incx;
                iy += incy;
            }
        }
        (Layout::RowMajor, Transpose::NoTrans)
        | (Layout::ColMajor, Transpose::Trans)
        | (Layout::ColMajor, Transpose::ConjTrans) => {
            let mut ia = kl;
            let mut ix = 0;
            let mut iy = 0;
            let mut ndiag = ku + 1;
            for _ in 0..kl {
                y[iy] += alpha * T::dot(ndiag, &a[ia..], 1, &x[ix..], incx);
                iy += incy;
                ndiag += 1;
                ia += lda - 1;
            }
            for _ in kl..m {
                y[iy] += alpha * T::dot(ndiag, &a[ia..], 1, &x[ix..], incx);
                iy += incy;
                ia += lda;
                ix += incx;
                ndiag -= 1;
            }
        }
    }
}
pub fn sgbmv(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    incx: usize,
    beta: f32,
    y: &mut [f32],
    incy: usize,
) {
    real_gbmv(
        layout, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
    );
}
pub fn dgbmv(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    incx: usize,
    beta: f64,
    y: &mut [f64],
    incy: usize,
) {
    real_gbmv(
        layout, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
    );
}
/// Performs the operation:
///  * NoTrans    - y := alpha * A  * x + beta * y
///  * Trans      - y := alpha * A' * x + beta * y
///  * ConjTrans  - y := alpha * conj(A') * x + beta * y
/// Where `A` is a general band matrix
///
/// # Arguments
///  * `layout` - the layout of the matrix `A`
///  * `trans` - whether the matrix should be transposed and/or conjugated
///  * `m` - the number of rows in the `A` matrix
///  * `n` - the number of columns in the `A` matrix
///  * `kl` - the number of sub-diagonals of the matrix `A`
///  * `ku` - the number of super-diagonals of the matrix `A`
///  * `alpha` - the value of the scalar alpha
///  * `a` - array of size `lda * n`, in _compressed form_ (see below for details)
///  * `lda`- the leading dimension of `a`
///  * `x` - array, of size at least (1 + (n - 1) * incx) (NoTrans) and at least
///          (1 + (m - 1) * incx) (Trans/TransConjugate)
///  * `incx` - the increment for the elements of `x`
///  * `beta` - the scalar beta
///  * `y` - the output array, size at least (1 + (m - 1) * incy) (NoTrans) and at least
///          (1 + (n - 1) * incy) otherwise
///  * `incy` - the increments for the elements of `y`
///
/// # Compressed form
/// `a` is
fn complex_gbmv<T>(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    x: &[T],
    incx: usize,
    beta: T,
    y: &mut [T],
    incy: usize,
) where
    T: ComplexFloat + lvl1::Scal + lvl1::Dotc + lvl1::Axpyc + AddAssign<T>,
{
    debug_assert!(lda >= kl + ku + 1);
    debug_assert!(incy > 0, "incy must be non-zero");
    let (ny, nx) = match trans {
        Transpose::NoTrans => (m, n),
        _ => (n, m),
    };
    debug_assert!(
        x.len() >= (1 + (nx - 1) * incx).max(1),
        "x must have size at least {}",
        (1 + (nx - 1) * incx).max(1)
    );
    debug_assert!(
        y.len() >= 1 + (ny - 1) * incy,
        "y must have size at least {}",
        1 + (ny - 1) * incy
    );
    debug_assert!(
        match layout {
            Layout::RowMajor => lda >= n.max(1),
            Layout::ColMajor => lda >= m.max(1),
        },
        "A has inconsistent lead dimension `lda`"
    );
    debug_assert!(
        match layout {
            Layout::RowMajor => a.len() >= lda * m,
            Layout::ColMajor => a.len() >= lda * n,
        },
        "A has insufficient size"
    );
    if ny.is_zero() {
        return;
    }
    if !beta.is_one() {
        T::scal(m, beta, y, incy);
    }
    if nx == 0 || alpha.is_zero() {
        return;
    }
    match (layout, trans) {
        (Layout::ColMajor, Transpose::NoTrans) | (Layout::RowMajor, Transpose::Trans) => {
            let mut ia = ku;
            let mut ix = 0;
            let mut iy = 0;
            let mut ndiag = m - ku;
            for _ in 0..ku {
                T::axpyu(ndiag, alpha * x[ix], &a[ia..], 1, &mut y[iy..], incy);
                ndiag += 1;
                ia += lda - 1;
                ix += incx;
            }
            for _ in ku..5 {
                T::axpyu(ndiag, alpha * x[ix], &a[ia..], 1, &mut y[iy..], incy);
                ndiag -= 1;
                ia += lda;
                ix += incx;
                iy += incy;
            }
        }
        (Layout::RowMajor, Transpose::ConjTrans) => {
            let mut ia = ku;
            let mut ix = 0;
            let mut iy = 0;
            let mut ndiag = m - ku;
            for _ in 0..ku {
                T::axpyc(ndiag, alpha * x[ix], &a[ia..], 1, &mut y[iy..], incy);
                ndiag += 1;
                ia += lda - 1;
                ix += incx;
            }
            for _ in ku..5 {
                T::axpyc(ndiag, alpha * x[ix], &a[ia..], 1, &mut y[iy..], incy);
                ndiag -= 1;
                ia += lda;
                ix += incx;
                iy += incy;
            }
        }
        (Layout::RowMajor, Transpose::NoTrans) | (Layout::ColMajor, Transpose::Trans) => {
            let mut ia = kl;
            let mut ix = 0;
            let mut iy = 0;
            let mut ndiag = ku + 1;
            for _ in 0..kl {
                y[iy] += alpha * T::dotu(ndiag, &a[ia..], 1, &x[ix..], incx);
                iy += incy;
                ndiag += 1;
                ia += lda - 1;
            }
            for _ in kl..m {
                y[iy] += alpha * T::dotu(ndiag, &a[ia..], 1, &x[ix..], incx);
                iy += incy;
                ia += lda;
                ix += incx;
                ndiag -= 1;
            }
        }
        (Layout::ColMajor, Transpose::ConjTrans) => {
            let mut ia = kl;
            let mut ix = 0;
            let mut iy = 0;
            let mut ndiag = ku + 1;
            for _ in 0..kl {
                y[iy] += alpha * T::dotc(ndiag, &a[ia..], 1, &x[ix..], incx);
                iy += incy;
                ndiag += 1;
                ia += lda - 1;
            }
            for _ in kl..m {
                y[iy] += alpha * T::dotc(ndiag, &a[ia..], 1, &x[ix..], incx);
                iy += incy;
                ia += lda;
                ix += incx;
                ndiag -= 1;
            }
        }
    }
}
pub fn cgbmv(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    x: &[Complex32],
    incx: usize,
    beta: Complex32,
    y: &mut [Complex32],
    incy: usize,
) {
    complex_gbmv(
        layout, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
    );
}
pub fn zgbmv(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    x: &[Complex64],
    incx: usize,
    beta: Complex64,
    y: &mut [Complex64],
    incy: usize,
) {
    complex_gbmv(
        layout, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
    );
}
/// Performs the operation:
///  * NoTrans           - y := alpha * A  * x + beta * y
///  * Trans/ConjTrans   - y := alpha * A' * x + beta * y
/// Where `A` is a symmetric band matrix
///
/// # Arguments
///  * `layout` - the layout of the matrix `A`
///  * `trans` - whether the matrix should be transposed and/or conjugated
///  * `m` - the number of rows in the `A` matrix
///  * `n` - the number of columns in the `A` matrix
///  * `k` - the number of non-diagonals of the matrix `A`
///  * `alpha` - the value of the scalar alpha
///  * `a` - array of size `lda * n`, in _compressed form_ (see below for details)
///  * `lda`- the leading dimension of `a`
///  * `x` - array, of size at least (1 + (n - 1) * incx) (NoTrans) and at least
///          (1 + (m - 1) * incx) (Trans/TransConjugate)
///  * `incx` - the increment for the elements of `x`
///  * `beta` - the scalar beta
///  * `y` - the output array, size at least (1 + (m - 1) * incy) (NoTrans) and at least
///          (1 + (n - 1) * incy) otherwise
///  * `incy` - the increments for the elements of `y`
pub fn sbmv<T>(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    x: &[T],
    incx: usize,
    beta: T,
    y: &mut [T],
    incy: usize,
) where
    T: Float,
{
    // TODO
    unimplemented!("TODO")
}
pub fn ssbmv(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    incx: usize,
    beta: f32,
    y: &mut [f32],
    incy: usize,
) {
    sbmv(
        layout, trans, m, n, k, alpha, a, lda, x, incx, beta, y, incy,
    );
}
pub fn dsbmv(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    incx: usize,
    beta: f64,
    y: &mut [f64],
    incy: usize,
) {
    sbmv(
        layout, trans, m, n, k, alpha, a, lda, x, incx, beta, y, incy,
    );
}
/// Performs the operation:
///  * NoTrans    - y := alpha * A  * x + beta * y
///  * Trans      - y := alpha * A' * x + beta * y
///  * ConjTrans  - y := alpha * conj(A') * x + beta * y
/// Where `A` is a hermitian band matrix
///
/// # Arguments
///  * `layout` - the layout of the matrix `A`
///  * `trans` - whether the matrix should be transposed and/or conjugated
///  * `m` - the number of rows in the `A` matrix
///  * `n` - the number of columns in the `A` matrix
///  * `k` - the number of non-diagonals of the matrix `A`
///  * `alpha` - the value of the scalar alpha
///  * `a` - array of size `lda * n`, in _compressed form_ (see below for details)
///  * `lda`- the leading dimension of `a`
///  * `x` - array, of size at least (1 + (n - 1) * incx) (NoTrans) and at least
///          (1 + (m - 1) * incx) (Trans/TransConjugate)
///  * `incx` - the increment for the elements of `x`
///  * `beta` - the scalar beta
///  * `y` - the output array, size at least (1 + (m - 1) * incy) (NoTrans) and at least
///          (1 + (n - 1) * incy) otherwise
///  * `incy` - the increments for the elements of `y`
///
/// # Compressed form
/// `a` is
fn hbmv<T>(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    x: &[T],
    incx: usize,
    beta: T,
    y: &mut [T],
    incy: usize,
) where
    T: ComplexFloat,
{
    // TODO
    unimplemented!("TODO")
}
pub fn chbmv(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    x: &[Complex32],
    incx: usize,
    beta: Complex32,
    y: &mut [Complex32],
    incy: usize,
) {
    hbmv(
        layout, trans, m, n, k, alpha, a, lda, x, incx, beta, y, incy,
    );
}
pub fn zhbmv(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    x: &[Complex64],
    incx: usize,
    beta: Complex64,
    y: &mut [Complex64],
    incy: usize,
) {
    hbmv(
        layout, trans, m, n, k, alpha, a, lda, x, incx, beta, y, incy,
    );
}
pub fn gemv<T>(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    x: &[T],
    incx: usize,
    beta: T,
    y: &mut [T],
    incy: usize,
) where
    T: One + Zero + lvl1::Scal + lvl1::Dot + Mul<T, Output = T> + PartialEq + AddAssign<T> + Copy,
{
    debug_assert!(incy > 0, "incy must be non-zero");
    let (ny, nx) = match trans {
        Transpose::NoTrans => (m, n),
        _ => (n, m),
    };
    if nx.is_zero() || ny.is_zero() {
        return;
    }
    debug_assert!(
        x.len() >= (1 + (nx - 1) * incx).max(1),
        "x must have size at least {}",
        (1 + (nx - 1) * incx).max(1)
    );
    debug_assert!(
        y.len() >= 1 + (ny - 1) * incy,
        "y must have size at least {}",
        1 + (ny - 1) * incy
    );
    debug_assert!(
        match layout {
            Layout::RowMajor => lda >= n.max(1),
            Layout::ColMajor => lda >= m.max(1),
        },
        "A has inconsistent lead dimension `lda`"
    );
    debug_assert!(
        match layout {
            Layout::RowMajor => a.len() >= lda * m,
            Layout::ColMajor => a.len() >= lda * n,
        },
        "A has insufficient size"
    );
    if !beta.is_one() {
        T::scal(ny, beta, y, incy);
    }
    if alpha.is_zero() {
        return;
    }
    match (layout, trans) {
        (Layout::RowMajor, Transpose::NoTrans)
        | (Layout::ColMajor, Transpose::Trans | Transpose::ConjTrans) => {
            y.into_iter()
                .step_by(incy)
                .take(ny)
                .enumerate()
                .for_each(|(i, yi)| {
                    *yi += alpha * T::dot(nx, &a[i * lda..], 1, x, incx);
                });
        }
        (Layout::ColMajor, Transpose::NoTrans)
        | (Layout::RowMajor, Transpose::Trans | Transpose::ConjTrans) => {
            y.into_iter()
                .step_by(incy)
                .take(ny)
                .enumerate()
                .for_each(|(i, yi)| {
                    *yi += alpha * T::dot(nx, &a[i..], lda, x, incx);
                });
        }
    }
}
pub fn gemvc<T>(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    x: &[T],
    incx: usize,
    beta: T,
    y: &mut [T],
    incy: usize,
) where
    T: One + Zero + lvl1::Scal + lvl1::Dotc + Mul<T, Output = T> + PartialEq + AddAssign<T> + Copy,
{
    debug_assert!(incy > 0, "incy must be non-zero");
    let (ny, nx) = match trans {
        Transpose::NoTrans => (m, n),
        _ => (n, m),
    };
    debug_assert!(
        x.len() >= (1 + (nx - 1) * incx).max(1),
        "x must have size at least {}",
        (1 + (nx - 1) * incx).max(1)
    );
    debug_assert!(
        y.len() >= 1 + (ny - 1) * incy,
        "y must have size at least {}",
        1 + (ny - 1) * incy
    );
    debug_assert!(
        match layout {
            Layout::RowMajor => lda >= n.max(1),
            Layout::ColMajor => lda >= m.max(1),
        },
        "A has inconsistent lead dimension `lda`"
    );
    debug_assert!(
        match layout {
            Layout::RowMajor => a.len() >= lda * m,
            Layout::ColMajor => a.len() >= lda * n,
        },
        "A has insufficient size"
    );
    if !beta.is_one() {
        T::scal(ny, beta, y, incy);
    }
    if alpha.is_zero() {
        return;
    }
    match (layout, trans) {
        (Layout::RowMajor, Transpose::NoTrans) | (Layout::ColMajor, Transpose::Trans) => {
            y.into_iter()
                .step_by(incy)
                .take(ny)
                .enumerate()
                .for_each(|(i, yi)| {
                    *yi += alpha * T::dotu(nx, &a[i * lda..], 1, x, incx);
                });
        }
        (Layout::ColMajor, Transpose::ConjTrans) => {
            y.into_iter()
                .step_by(incy)
                .take(ny)
                .enumerate()
                .for_each(|(i, yi)| {
                    *yi += alpha * T::dotc(nx, &a[i * lda..], 1, x, incx);
                });
        }
        (Layout::ColMajor, Transpose::NoTrans) | (Layout::RowMajor, Transpose::Trans) => {
            y.into_iter()
                .step_by(incy)
                .take(ny)
                .enumerate()
                .for_each(|(i, yi)| {
                    *yi += alpha * T::dotu(nx, &a[i..], lda, x, incx);
                });
        }
        (Layout::RowMajor, Transpose::ConjTrans) => {
            y.into_iter()
                .step_by(incy)
                .take(ny)
                .enumerate()
                .for_each(|(i, yi)| {
                    *yi += alpha * T::dotc(nx, &a[i..], lda, x, incx);
                });
        }
    }
}

/// Computes a matrix-vector product using a general matrix
///  y := alpha * A * x + beta * y  [NoTrans]
///  y := alpha * A' * x + beta * y [Trans/ConjTrans]
pub fn sgemv(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    incx: usize,
    beta: f32,
    y: &mut [f32],
    incy: usize,
) {
    gemv(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
}
/// Computes a matrix-vector product using a general matrix
///  y := alpha * A * x + beta * y  [NoTrans]
///  y := alpha * A' * x + beta * y [Trans/ConjTrans]
pub fn dgemv(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    incx: usize,
    beta: f64,
    y: &mut [f64],
    incy: usize,
) {
    gemv(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
}
/// Computes a matrix-vector product using a general matrix
///  y := alpha * A * x + beta * y  [NoTrans]
///  y := alpha * A' * x + beta * y [Trans]
///  y := alpha * conj(A') * x + beta * y [ConjTrans]
pub fn cgemv(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    x: &[Complex32],
    incx: usize,
    beta: Complex32,
    y: &mut [Complex32],
    incy: usize,
) {
    gemvc(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
}
/// Computes a matrix-vector product using a general matrix
///  y := alpha * A * x + beta * y  [NoTrans]
///  y := alpha * A' * x + beta * y [Trans]
///  y := alpha * conj(A') * x + beta * y [ConjTrans]
pub fn zgemv(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    x: &[Complex64],
    incx: usize,
    beta: Complex64,
    y: &mut [Complex64],
    incy: usize,
) {
    gemvc(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
}

pub fn symv<T>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    x: &[T],
    incx: usize,
    beta: T,
    y: &mut [T],
    incy: usize,
) where
    T: One + Zero + lvl1::Scal + lvl1::Dot + Mul<T, Output = T> + PartialEq + AddAssign<T> + Copy,
{
    debug_assert!(incy > 0, "incy must be non-zero");
    if !beta.is_one() {
        T::scal(n, beta, y, incy);
    }
    if alpha.is_zero() || n.is_zero() {
        return;
    }
    debug_assert!(
        x.len() >= (1 + (n - 1) * incx).max(1),
        "x must have size at least {}",
        (1 + (n - 1) * incx).max(1)
    );
    debug_assert!(
        y.len() >= 1 + (n - 1) * incy,
        "y must have size at least {}",
        1 + (n - 1) * incy
    );
    debug_assert!(lda >= n.max(1), "A has inconsistent lead dimension `lda`");
    debug_assert!(a.len() >= lda * n, "A has insufficient size");
    match (layout, uplo) {
        (Layout::RowMajor, UpLo::Lower) | (Layout::ColMajor, UpLo::Upper) => {
            y.into_iter()
                .step_by(incy)
                .take(n)
                .enumerate()
                .for_each(|(i, yi)| {
                    *yi += alpha * T::dot(n - i, &a[i * lda + i..], lda, &x[i * incx..], incx);
                    *yi += alpha * T::dot(i, &a[i * lda..], 1, x, incx);
                });
        }
        (Layout::ColMajor, UpLo::Lower) | (Layout::RowMajor, UpLo::Upper) => {
            y.into_iter()
                .step_by(incy)
                .take(n)
                .enumerate()
                .for_each(|(i, yi)| {
                    *yi += alpha * T::dot(n - i, &a[i * lda + i..], 1, &x[i * incx..], incx);
                    *yi += alpha * T::dot(i, &a[i..], lda, x, incx);
                });
        }
    }
}

pub fn ssymv(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    incx: usize,
    beta: f32,
    y: &mut [f32],
    incy: usize,
) {
    symv(layout, uplo, n, alpha, a, lda, x, incx, beta, y, incy)
}

pub fn dsymv(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    incx: usize,
    beta: f64,
    y: &mut [f64],
    incy: usize,
) {
    symv(layout, uplo, n, alpha, a, lda, x, incx, beta, y, incy)
}
/// Packed symmetric matrix . value
///   y := alpha * A * x + beta * y
pub fn spmv<T>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: T,
    ap: &[T],
    x: &[T],
    incx: usize,
    beta: T,
    y: &mut [T],
    incy: usize,
) where
    T: One + Zero + lvl1::Scal + lvl1::Dot + Mul<T, Output = T> + PartialEq + AddAssign<T> + Copy,
{
    debug_assert!(incy > 0, "incy must be non-zero");
    if !beta.is_one() {
        T::scal(n, beta, y, incy);
    }
    if alpha.is_zero() || n.is_zero() {
        return;
    }
    debug_assert!(
        x.len() >= 1 + (n - 1) * incx,
        "x must have size at least {}",
        1 + (n - 1) * incx
    );
    debug_assert!(
        y.len() >= 1 + (n - 1) * incy,
        "y must have size at least {}",
        1 + (n - 1) * incy
    );
    debug_assert!(ap.len() >= n * (n + 1) / 2, "A has insufficient size");
    match (layout, uplo) {
        (Layout::RowMajor, UpLo::Upper) | (Layout::ColMajor, UpLo::Lower) => {
            let mut start = 0;
            for (i, yi) in y.into_iter().step_by(incy).take(n).enumerate() {
                *yi += T::dot(n - i, &ap[start..], 1, &x[i..], incx);
                let mut u = i;
                for k in 0..i {
                    *yi += ap[u] * x[k];
                    u += n - k - 1;
                }
                start += n - i;
            }
        }
        (Layout::RowMajor, UpLo::Lower) | (Layout::ColMajor, UpLo::Upper) => {
            let mut start = 0;
            for (i, yi) in y.into_iter().step_by(incy).take(n).enumerate() {
                *yi += T::dot(i + 1, &ap[start..], 1, &x, incx);
                let mut u = start + 2 * i + 1;
                for k in i + 1..n {
                    *yi += ap[u] * x[k];
                    u += k + 1;
                }
                start += i + 1;
            }
        }
    }
}
pub fn sspmv(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: f32,
    ap: &[f32],
    x: &[f32],
    incx: usize,
    beta: f32,
    y: &mut [f32],
    incy: usize,
) {
    spmv(layout, uplo, n, alpha, ap, x, incx, beta, y, incy);
}
pub fn dspmv(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: f64,
    ap: &[f64],
    x: &[f64],
    incx: usize,
    beta: f64,
    y: &mut [f64],
    incy: usize,
) {
    spmv(layout, uplo, n, alpha, ap, x, incx, beta, y, incy);
}
pub fn hemv<T>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    x: &[T],
    incx: usize,
    beta: T,
    y: &mut [T],
    incy: usize,
) where
    T: ComplexFloat + lvl1::Scal + lvl1::Dotc + MulAssign<T> + AddAssign<T>,
{
    debug_assert!(incy > 0, "incy must be non-zero");
    if !beta.is_one() {
        T::scal(n, beta, y, incy);
    }
    if alpha.is_zero() || n.is_zero() {
        return;
    }
    debug_assert!(
        x.len() >= (1 + (n - 1) * incx).max(1),
        "x must have size at least {}",
        (1 + (n - 1) * incx).max(1)
    );
    debug_assert!(
        y.len() >= 1 + (n - 1) * incy,
        "y must have size at least {}",
        1 + (n - 1) * incy
    );
    debug_assert!(lda >= n.max(1), "A has inconsistent lead dimension `lda`");
    debug_assert!(a.len() >= lda * n, "A has insufficient size");
    match (layout, uplo) {
        (Layout::RowMajor, UpLo::Lower) | (Layout::ColMajor, UpLo::Upper) => {
            y.into_iter()
                .step_by(incy)
                .take(n)
                .enumerate()
                .for_each(|(i, yi)| {
                    *yi += alpha * T::dotu(n - i, &a[i * lda + i..], lda, &x[i * incx..], incx);
                    *yi += alpha * T::dotu(i, &a[i * lda..], 1, x, incx);
                });
        }
        (Layout::ColMajor, UpLo::Lower) | (Layout::RowMajor, UpLo::Upper) => {
            y.into_iter()
                .step_by(incy)
                .take(n)
                .enumerate()
                .for_each(|(i, yi)| {
                    *yi += alpha * T::dotu(n - i, &a[i * lda + i..], 1, &x[i * incx..], incx);
                    *yi += alpha * T::dotu(i, &a[i..], lda, x, incx);
                });
        }
    }
}
pub fn chemv(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    x: &[Complex32],
    incx: usize,
    beta: Complex32,
    y: &mut [Complex32],
    incy: usize,
) {
    hemv(layout, uplo, n, alpha, a, lda, x, incx, beta, y, incy)
}
pub fn zhemv(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    x: &[Complex64],
    incx: usize,
    beta: Complex64,
    y: &mut [Complex64],
    incy: usize,
) {
    hemv(layout, uplo, n, alpha, a, lda, x, incx, beta, y, incy)
}
/// Performs a Matrix/vector product on a packed hermitian matrix
///  y := alpha * A * x + beta * y
/// Where A is provided as a _packed_ hermitian matrix
pub fn hpmv<T>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: T,
    ap: &[T],
    x: &[T],
    incx: usize,
    beta: T,
    y: &mut [T],
    incy: usize,
) where
    T: ComplexFloat + lvl1::Scal + lvl1::Dotc + AddAssign<T>,
{
    debug_assert!(incy > 0, "incy must be non-zero");
    if !beta.is_one() {
        T::scal(n, beta, y, incy);
    }
    if alpha.is_zero() || n.is_zero() {
        return;
    }
    debug_assert!(
        x.len() >= 1 + (n - 1) * incx,
        "x must have size at least {}",
        1 + (n - 1) * incx
    );
    debug_assert!(
        y.len() >= 1 + (n - 1) * incy,
        "y must have size at least {}",
        1 + (n - 1) * incy
    );
    debug_assert!(ap.len() >= n * (n + 1) / 2, "A has insufficient size");
    match (layout, uplo) {
        (Layout::RowMajor, UpLo::Upper) => {
            let mut start = 0;
            for (i, yi) in y.into_iter().step_by(incy).take(n).enumerate() {
                *yi += T::dotu(n - i, &ap[start..], 1, &x[i..], incx);
                let mut u = i;
                for k in 0..i {
                    *yi += ap[u].conj() * x[k];
                    u += n - k - 1;
                }
                start += n - i;
            }
        }
        (Layout::ColMajor, UpLo::Lower) => {
            let mut start = 0;
            for (i, yi) in y.into_iter().step_by(incy).take(n).enumerate() {
                *yi += T::dotc(n - i, &ap[start..], 1, &x[i..], incx);
                let mut u = i;
                for k in 0..i {
                    *yi += ap[u] * x[k];
                    u += n - k - 1;
                }
                start += n - i;
            }
        }
        (Layout::RowMajor, UpLo::Lower) => {
            let mut start = 0;
            for (i, yi) in y.into_iter().step_by(incy).take(n).enumerate() {
                *yi += T::dotu(i + 1, &ap[start..], 1, &x, incx);
                let mut u = start + 2 * i + 1;
                for k in i + 1..n {
                    *yi += ap[u].conj() * x[k];
                    u += k + 1;
                }
                start += i + 1;
            }
        }
        (Layout::ColMajor, UpLo::Upper) => {
            let mut start = 0;
            for (i, yi) in y.into_iter().step_by(incy).take(n).enumerate() {
                *yi += T::dotc(i + 1, &ap[start..], 1, &x, incx);
                let mut u = start + 2 * i + 1;
                for k in i + 1..n {
                    *yi += ap[u] * x[k];
                    u += k + 1;
                }
                start += i + 1;
            }
        }
    }
}
/// Performs a Matrix/vector product on a packed hermitian matrix
///  y := alpha * A * x + beta * y
/// Where A is provided as a _packed_ hermitian matrix
pub fn chpmv(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: Complex32,
    ap: &[Complex32],
    x: &[Complex32],
    incx: usize,
    beta: Complex32,
    y: &mut [Complex32],
    incy: usize,
) {
    hpmv(layout, uplo, n, alpha, ap, x, incx, beta, y, incy)
}
/// Performs a Matrix/vector product on a packed hermitian matrix
///  y := alpha * A * x + beta * y
/// Where A is provided as a _packed_ hermitian matrix
pub fn zhpmv(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: Complex64,
    ap: &[Complex64],
    x: &[Complex64],
    incx: usize,
    beta: Complex64,
    y: &mut [Complex64],
    incy: usize,
) {
    hpmv(layout, uplo, n, alpha, ap, x, incx, beta, y, incy)
}
/// Rank-1 update of a general matrix
///   A := alpha * x * y' + A
pub fn ger<T>(
    layout: Layout,
    m: usize,
    n: usize,
    alpha: T,
    x: &[T],
    incx: usize,
    y: &[T],
    incy: usize,
    a: &mut [T],
    lda: usize,
) where
    T: lvl1::Axpy + Float,
{
    debug_assert!(
        x.len() >= (1 + (m - 1) * incx).max(1),
        "x must have size at least {}",
        (1 + (m - 1) * incx).max(1)
    );
    debug_assert!(
        y.len() >= 1 + (n - 1) * incy,
        "y must have size at least {}",
        1 + (n - 1) * incy
    );
    debug_assert!(
        match layout {
            Layout::RowMajor => lda >= n.max(1),
            Layout::ColMajor => lda >= m.max(1),
        },
        "A has inconsistent lead dimension `lda`"
    );
    debug_assert!(
        match layout {
            Layout::RowMajor => a.len() >= lda * m,
            Layout::ColMajor => a.len() >= lda * n,
        },
        "A has insufficient size"
    );
    match layout {
        Layout::RowMajor if incx.is_zero() => {
            let x0 = x[0];
            a.chunks_mut(lda).take(m).for_each(|chunk| {
                T::axpy(n, alpha * x0, y, incy, chunk, 1);
            })
        }
        Layout::RowMajor => a
            .chunks_mut(lda)
            .zip(x.into_iter().step_by(incx))
            .take(m)
            .for_each(|(chunk, &xi)| {
                T::axpy(n, alpha * xi, y, incy, chunk, 1);
            }),
        Layout::ColMajor if incy.is_zero() => {
            let y0 = y[0];
            a.chunks_mut(lda).take(n).for_each(|chunk| {
                T::axpy(n, alpha * y0, x, incx, chunk, 1);
            });
        }
        Layout::ColMajor => a
            .chunks_mut(lda)
            .zip(y.into_iter().step_by(incy))
            .take(n)
            .for_each(|(chunk, &yj)| {
                T::axpy(n, alpha * yj, x, incx, chunk, 1);
            }),
    }
}

/// Rank-1 update of a general matrix
///  A := alpha * x * y' + A
pub fn sger(
    layout: Layout,
    m: usize,
    n: usize,
    alpha: f32,
    x: &[f32],
    incx: usize,
    y: &[f32],
    incy: usize,
    a: &mut [f32],
    lda: usize,
) {
    ger(layout, m, n, alpha, x, incx, y, incy, a, lda)
}
pub fn dger(
    layout: Layout,
    m: usize,
    n: usize,
    alpha: f64,
    x: &[f64],
    incx: usize,
    y: &[f64],
    incy: usize,
    a: &mut [f64],
    lda: usize,
) {
    ger(layout, m, n, alpha, x, incx, y, incy, a, lda)
}
pub fn geru<T>(
    layout: Layout,
    m: usize,
    n: usize,
    alpha: T,
    x: &[T],
    incx: usize,
    y: &[T],
    incy: usize,
    a: &mut [T],
    lda: usize,
) where
    T: ComplexFloat + lvl1::Axpy,
{
    debug_assert!(
        x.len() >= (1 + (m - 1) * incx).max(1),
        "x must have size at least {}",
        (1 + (m - 1) * incx).max(1)
    );
    debug_assert!(
        y.len() >= 1 + (n - 1) * incy,
        "y must have size at least {}",
        1 + (n - 1) * incy
    );
    debug_assert!(
        match layout {
            Layout::RowMajor => lda >= n.max(1),
            Layout::ColMajor => lda >= m.max(1),
        },
        "A has inconsistent lead dimension `lda`"
    );
    debug_assert!(
        match layout {
            Layout::RowMajor => a.len() >= lda * m,
            Layout::ColMajor => a.len() >= lda * n,
        },
        "A has insufficient size"
    );
    match layout {
        Layout::RowMajor if incx.is_zero() => {
            let x0 = x[0];
            a.chunks_mut(lda).take(m).for_each(|chunk| {
                T::axpy(n, alpha * x0, y, incy, chunk, 1);
            })
        }
        Layout::RowMajor => a
            .chunks_mut(lda)
            .zip(x.into_iter().step_by(incx))
            .take(m)
            .for_each(|(chunk, &xi)| {
                T::axpy(n, alpha * xi, y, incy, chunk, 1);
            }),
        Layout::ColMajor if incy.is_zero() => {
            let y0 = y[0];
            a.chunks_mut(lda).take(n).for_each(|chunk| {
                T::axpy(n, alpha * y0, x, incx, chunk, 1);
            });
        }
        Layout::ColMajor => a
            .chunks_mut(lda)
            .zip(y.into_iter().step_by(incy))
            .take(n)
            .for_each(|(chunk, &yj)| {
                T::axpy(n, alpha * yj, x, incx, chunk, 1);
            }),
    }
}
pub fn gerc<T>(
    layout: Layout,
    m: usize,
    n: usize,
    alpha: T,
    x: &[T],
    incx: usize,
    y: &[T],
    incy: usize,
    a: &mut [T],
    lda: usize,
) where
    T: ComplexFloat + lvl1::Axpyc,
{
    debug_assert!(
        x.len() >= (1 + (m - 1) * incx).max(1),
        "x must have size at least {}",
        (1 + (m - 1) * incx).max(1)
    );
    debug_assert!(
        y.len() >= 1 + (n - 1) * incy,
        "y must have size at least {}",
        1 + (n - 1) * incy
    );
    debug_assert!(
        match layout {
            Layout::RowMajor => lda >= n.max(1),
            Layout::ColMajor => lda >= m.max(1),
        },
        "A has inconsistent lead dimension `lda`"
    );
    debug_assert!(
        match layout {
            Layout::RowMajor => a.len() >= lda * m,
            Layout::ColMajor => a.len() >= lda * n,
        },
        "A has insufficient size"
    );
    match layout {
        Layout::RowMajor if incx.is_zero() => {
            let x0 = x[0];
            a.chunks_mut(lda).take(m).for_each(|chunk| {
                T::axpyc(n, alpha * x0, y, incy, chunk, 1);
            })
        }
        Layout::RowMajor => a
            .chunks_mut(lda)
            .zip(x.into_iter().step_by(incx))
            .take(m)
            .for_each(|(chunk, &xi)| {
                T::axpyc(n, alpha * xi, y, incy, chunk, 1);
            }),
        Layout::ColMajor if incy.is_zero() => {
            let y0 = y[0];
            a.chunks_mut(lda).take(n).for_each(|chunk| {
                T::axpyc(n, alpha * y0, x, incx, chunk, 1);
            });
        }
        Layout::ColMajor => a
            .chunks_mut(lda)
            .zip(y.into_iter().step_by(incy))
            .take(n)
            .for_each(|(chunk, &yj)| {
                T::axpyc(n, alpha * yj, x, incx, chunk, 1);
            }),
    }
}
pub fn cgeru(
    layout: Layout,
    m: usize,
    n: usize,
    alpha: Complex32,
    x: &[Complex32],
    incx: usize,
    y: &[Complex32],
    incy: usize,
    a: &mut [Complex32],
    lda: usize,
) {
    geru(layout, m, n, alpha, x, incx, y, incy, a, lda)
}
pub fn cgerc(
    layout: Layout,
    m: usize,
    n: usize,
    alpha: Complex32,
    x: &[Complex32],
    incx: usize,
    y: &[Complex32],
    incy: usize,
    a: &mut [Complex32],
    lda: usize,
) {
    gerc(layout, m, n, alpha, x, incx, y, incy, a, lda)
}
pub fn zgeru(
    layout: Layout,
    m: usize,
    n: usize,
    alpha: Complex64,
    x: &[Complex64],
    incx: usize,
    y: &[Complex64],
    incy: usize,
    a: &mut [Complex64],
    lda: usize,
) {
    geru(layout, m, n, alpha, x, incx, y, incy, a, lda)
}
pub fn zgerc(
    layout: Layout,
    m: usize,
    n: usize,
    alpha: Complex64,
    x: &[Complex64],
    incx: usize,
    y: &[Complex64],
    incy: usize,
    a: &mut [Complex64],
    lda: usize,
) {
    gerc(layout, m, n, alpha, x, incx, y, incy, a, lda)
}
/// Rank-1 update of a symmetric matrix:
///   A := alpha * x * x' + A
pub fn syr<T>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: T,
    x: &[T],
    incx: usize,
    a: &mut [T],
    lda: usize,
) where
    T: Float + lvl1::Shift + lvl1::Axpy,
{
    debug_assert!(lda >= 1.max(n));
    if alpha.is_zero() || n.is_zero() {
        return;
    }
    debug_assert!(x.len() >= (1 + (n - 1) * incx).max(1));
    debug_assert!(a.len() >= n * lda);
    if incx.is_zero() {
        let prod = alpha * x[0] * x[0];
        match (layout, uplo) {
            (Layout::RowMajor, UpLo::Lower) | (Layout::ColMajor, UpLo::Upper) => {
                for i in 0..n {
                    T::shift(i + 1, prod, &mut a[i * lda..], 1);
                }
            }
            (Layout::ColMajor, UpLo::Lower) | (Layout::RowMajor, UpLo::Upper) => {
                for i in 0..n {
                    T::shift(n - i, prod, &mut a[i * lda + i..], 1);
                }
            }
        }
    } else {
        match (layout, uplo) {
            (Layout::RowMajor, UpLo::Lower) | (Layout::ColMajor, UpLo::Upper) => a
                .chunks_mut(lda)
                .take(n)
                .enumerate()
                .for_each(|(i, chunk)| {
                    T::axpy(i+1, alpha * x[i * incx], x, incx, chunk, 1);
                }),
            (Layout::ColMajor, UpLo::Lower) | (Layout::RowMajor, UpLo::Upper) => a
                .chunks_mut(lda)
                .take(n)
                .enumerate()
                .for_each(|(i, chunk)| {
                    T::axpy(
                        n - i,
                        alpha * x[i * incx],
                        &x[i * incx..],
                        incx,
                        &mut chunk[i..],
                        1
                    );
                }),
        }
    }
}
/// Rank-1 update of a symmetric matrix:
///   A := alpha * x * x' + A
pub fn ssyr(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: f32,
    x: &[f32],
    incx: usize,
    a: &mut [f32],
    lda: usize,
) {
    syr(layout, uplo, n, alpha, x, incx, a, lda)
}
/// Rank-1 update of a symmetric matrix:
///   A := alpha * x * x' + A
pub fn dsyr(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: f64,
    x: &[f64],
    incx: usize,
    a: &mut [f64],
    lda: usize,
) {
    syr(layout, uplo, n, alpha, x, incx, a, lda)
}
/// Rank-1 update of a hermitian matrix:
///   A := alpha * x * conj(x') + A
pub fn her<T>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: T,
    x: &[T],
    incx: usize,
    a: &mut [T],
    lda: usize,
) where
    T: ComplexFloat + lvl1::Axpy + lvl1::Shift,
{
    debug_assert!(lda >= 1.max(n));
    if alpha.is_zero() || n.is_zero() {
        return;
    }
    debug_assert!(x.len() >= (1 + (n - 1) * incx).max(1));
    debug_assert!(a.len() >= n * lda);
    if incx.is_zero() {
        let prod = alpha * x[0] * x[0].conj();
        match (layout, uplo) {
            (Layout::RowMajor, UpLo::Lower) | (Layout::ColMajor, UpLo::Upper) => {
                for i in 0..n {
                    T::shift(i + 1, prod, &mut a[lda * i..], 1);
                }
            }
            (Layout::ColMajor, UpLo::Lower) | (Layout::RowMajor, UpLo::Upper) => {
                for i in 0..n {
                    T::shift(n - i, prod, &mut a[lda * i + i..], 1);
                }
            }
        }
    } else {
        match (layout, uplo) {
            (Layout::RowMajor, UpLo::Lower) => x
                .into_iter()
                .step_by(incx)
                .take(n)
                .enumerate()
                .for_each(|(i, xi)| {
                    T::axpy(
                        n - i,
                        alpha * xi.conj(),
                        &x[i * incx..],
                        incx,
                        &mut a[i * lda + i..],
                        lda,
                    );
                }),
            (Layout::ColMajor, UpLo::Upper) => x
                .into_iter()
                .step_by(incx)
                .take(n)
                .enumerate()
                .for_each(|(i, xi)| {
                    T::axpy(i + 1, alpha * xi.conj(), x, incx, &mut a[i * lda..], 1);
                }),
            (Layout::ColMajor, UpLo::Lower) => x
                .into_iter()
                .step_by(incx)
                .take(n)
                .enumerate()
                .for_each(|(i, xi)| {
                    T::axpy(
                        n - i,
                        alpha * xi.conj(),
                        &x[i * incx..],
                        incx,
                        &mut a[i * lda + i..],
                        1,
                    );
                }),
            (Layout::RowMajor, UpLo::Upper) => x
                .into_iter()
                .step_by(incx)
                .take(n)
                .enumerate()
                .for_each(|(i, xi)| {
                    T::axpy(i + 1, alpha * xi.conj(), x, incx, &mut a[i..], lda);
                }),
        }
    }
}
/// Rank-1 update of a hermitian matrix:
///   A := alpha * x * conj(x') + A
pub fn cher(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: Complex32,
    x: &[Complex32],
    incx: usize,
    a: &mut [Complex32],
    lda: usize,
) {
    her(layout, uplo, n, alpha, x, incx, a, lda)
}
/// Rank-1 update of a hermitian matrix:
///   A := alpha * x * conj(x') + A
pub fn zher(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: Complex64,
    x: &[Complex64],
    incx: usize,
    a: &mut [Complex64],
    lda: usize,
) {
    her(layout, uplo, n, alpha, x, incx, a, lda)
}
/// Performs a rank-1 update of a symmetric packed matrix.
///   A := alpha*x*x'+ A,
pub fn spr<T>(layout: Layout, uplo: UpLo, n: usize, alpha: T, x: &[T], incx: usize, ap: &mut [T])
where
    T: Float + lvl1::Shift + lvl1::Axpy,
{
    if alpha.is_zero() || n.is_zero() {
        return;
    }
    debug_assert!(x.len() >= 1 + (n - 1) * incx);
    debug_assert!(ap.len() >= n * (n - 1) / 2);
    if incx.is_zero() {
        T::shift(n * (n - 1) / 2, alpha * x[0] * x[0], ap, 1);
    } else {
        match (layout, uplo) {
            (Layout::RowMajor, UpLo::Lower) | (Layout::ColMajor, UpLo::Upper) => {
                let mut start = 0;
                for (i, &xi) in x.into_iter().step_by(incx).take(n).enumerate() {
                    T::axpy(i + 1, alpha * xi, &x[..], incx, &mut ap[start..], 1);
                    start += i + 1;
                }
            }
            (Layout::ColMajor, UpLo::Lower) | (Layout::RowMajor, UpLo::Upper) => {
                let mut start = 0;
                for (i, &xi) in x.into_iter().step_by(incx).take(n).enumerate() {
                    T::axpy(n - i, alpha * xi, &x[i * incx..], incx, &mut ap[start..], 1);
                    start += n - i;
                }
            }
        }
    }
}
pub fn sspr(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: f32,
    x: &[f32],
    incx: usize,
    ap: &mut [f32],
) {
    spr(layout, uplo, n, alpha, x, incx, ap);
}
pub fn dspr(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: f64,
    x: &[f64],
    incx: usize,
    ap: &mut [f64],
) {
    spr(layout, uplo, n, alpha, x, incx, ap);
}

/// Computes a rank-2 update of a symmetric packed matrix.
///   A := alpha*x*y'+ alpha*y*x' + A,
pub fn spr2<T>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: T,
    x: &[T],
    incx: usize,
    y: &[T],
    incy: usize,
    ap: &mut [T],
) where
    T: Float + lvl1::Shift + lvl1::Axpy,
{
    if n.is_zero() {
        return;
    }
    debug_assert!(x.len() >= 1 + (n - 1) * incx);
    debug_assert!(y.len() >= 1 + (n - 1) * incy);
    if incx.is_zero() && incy.is_zero() {
        T::shift(
            n * (n - 1) / 2,
            alpha * x[0] * y[0] + alpha * y[0] * x[0],
            ap,
            1,
        );
    } else if incy.is_zero() {
        let ay = alpha * y[0];
        match (layout, uplo) {
            (Layout::RowMajor, UpLo::Lower) | (Layout::ColMajor, UpLo::Upper) => {
                let mut start = 0;
                for (i, &xi) in x.into_iter().step_by(incx).take(n).enumerate() {
                    T::shift(i + 1, xi * ay, &mut ap[start..], 1);
                    T::axpy(i + 1, ay, &x[..], incx, &mut ap[start..], 1);
                    start += i + 1;
                }
            }
            (Layout::ColMajor, UpLo::Lower) | (Layout::RowMajor, UpLo::Upper) => {
                let mut start = 0;
                for (i, &xi) in x.into_iter().step_by(incx).take(n).enumerate() {
                    T::shift(n - i, ay * xi, &mut ap[start..], 1);
                    T::axpy(n - i, ay, &x[i * incx..], incx, &mut ap[start..], 1);
                    start += n - i;
                }
            }
        }
    } else if incx.is_zero() {
        spr2(layout, uplo, n, alpha, y, incy, x, incx, ap);
    } else {
        match (layout, uplo) {
            (Layout::RowMajor, UpLo::Lower) | (Layout::ColMajor, UpLo::Upper) => {
                let mut start = 0;
                for (i, (&xi, &yi)) in x
                    .into_iter()
                    .step_by(incx)
                    .zip(y.into_iter().step_by(incy))
                    .take(n)
                    .enumerate()
                {
                    T::axpy(i + 1, alpha * xi, &y[..], incy, &mut ap[start..], 1);
                    T::axpy(i + 1, alpha * yi, &x[..], incx, &mut ap[start..], 1);
                    start += i + 1;
                }
            }
            (Layout::ColMajor, UpLo::Lower) | (Layout::RowMajor, UpLo::Upper) => {
                let mut start = 0;
                for (i, (&xi, &yi)) in x
                    .into_iter()
                    .step_by(incx)
                    .zip(y.into_iter().step_by(incy))
                    .take(n)
                    .enumerate()
                {
                    T::axpy(n - i, alpha * xi, &y[i * incy..], incy, &mut ap[start..], 1);
                    T::axpy(n - i, alpha * yi, &x[i * incx..], incx, &mut ap[start..], 1);
                    start += n - i;
                }
            }
        }
    }
}
pub fn sspr2(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: f32,
    x: &[f32],
    incx: usize,
    y: &[f32],
    incy: usize,
    ap: &mut [f32],
) {
    spr2(layout, uplo, n, alpha, x, incx, y, incy, ap);
}
pub fn dspr2(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: f64,
    x: &[f64],
    incx: usize,
    y: &[f64],
    incy: usize,
    ap: &mut [f64],
) {
    spr2(layout, uplo, n, alpha, x, incx, y, incy, ap);
}
pub fn syr2<T>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: T,
    x: &[T],
    incx: usize,
    y: &[T],
    incy: usize,
    a: &mut [T],
    lda: usize,
) where
    T: Float + lvl1::Axpy + lvl1::Shift,
{
    debug_assert!(lda >= 1.max(n));
    if alpha.is_zero() || n.is_zero() {
        return;
    }
    debug_assert!(x.len() >= (1 + (n - 1) * incx).max(1));
    debug_assert!(y.len() >= (1 + (n - 1) * incy).max(1));
    debug_assert!(a.len() >= n * lda);
    if incx.is_zero() && incy.is_zero() {
        let axy = alpha * x[0] * y[0] + alpha * y[0] * x[0];
        match (layout, uplo) {
            (Layout::RowMajor, UpLo::Lower) | (Layout::ColMajor, UpLo::Upper) => {
                for i in 0..n {
                    T::shift(i + 1, axy, &mut a[i * lda..], 1);
                }
            }
            (Layout::ColMajor, UpLo::Lower) | (Layout::RowMajor, UpLo::Upper) => {
                for i in 0..n {
                    T::shift(n - i, axy, &mut a[i * lda + i..], 1);
                }
            }
        }
    } else if incx.is_zero() {
        let ax0 = alpha * x[0];
        match (layout, uplo) {
            (Layout::RowMajor, UpLo::Lower) | (Layout::ColMajor, UpLo::Upper) => y
                .into_iter()
                .step_by(incy)
                .take(n)
                .enumerate()
                .for_each(|(i, &yi)| {
                    T::shift(i + 1, ax0 * yi, &mut a[i * lda..], 1);
                    T::axpy(i + 1, ax0, y, incy, &mut a[i * lda..], 1);
                }),
            (Layout::ColMajor, UpLo::Lower) | (Layout::RowMajor, UpLo::Upper) => y
                .into_iter()
                .step_by(incy)
                .take(n)
                .enumerate()
                .for_each(|(i, &yi)| {
                    T::shift(n - i, ax0 * yi, &mut a[i * lda + i..], 1);
                    T::axpy(n - i, ax0, &y[i * incy..], incy, &mut a[i * lda + i..], 1);
                }),
        }
    } else if incy.is_zero() {
        // Symmetrical operation: flipping x and y
        syr2(layout, uplo, n, alpha, y, incy, x, incx, a, lda);
    } else {
        match (layout, uplo) {
            (Layout::RowMajor, UpLo::Lower) | (Layout::ColMajor, UpLo::Upper) => a
                .chunks_mut(lda)
                .take(n)
                .enumerate()
                .for_each(|(i, chunk)| {
                    T::axpy(i+1, alpha * y[i * incy], x, incx, chunk, 1);
                    T::axpy(i+1, alpha * x[i * incx], y, incy, chunk, 1);
                }),
            (Layout::ColMajor, UpLo::Lower) | (Layout::RowMajor, UpLo::Upper) => a
                .chunks_mut(lda)
                .take(n)
                .enumerate()
                .for_each(|(i, chunk)| {
                    T::axpy(
                        n - i,
                        alpha * y[i * incy],
                        &x[i * incx..],
                        incx,
                        &mut chunk[i..],
                        1
                    );
                    T::axpy(
                        n - i,
                        alpha * x[i * incx],
                        &y[i * incy..],
                        incy,
                        &mut chunk[i..],
                        1
                    );
                }),
        }
    }
}
/// Rank-2 update of a symmetric matrix:
///   A := alpha * x * y' + alpha * y * x' + A
pub fn ssyr2(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: f32,
    x: &[f32],
    incx: usize,
    y: &[f32],
    incy: usize,
    a: &mut [f32],
    lda: usize,
) {
    syr2(layout, uplo, n, alpha, x, incx, y, incy, a, lda)
}
/// Rank-2 update of a symmetric matrix:
///   A := alpha * x * y' + alpha * y * x' + A
pub fn dsyr2(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: f64,
    x: &[f64],
    incx: usize,
    y: &[f64],
    incy: usize,
    a: &mut [f64],
    lda: usize,
) {
    syr2(layout, uplo, n, alpha, x, incx, y, incy, a, lda)
}
pub fn her2<T>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: T,
    x: &[T],
    incx: usize,
    y: &[T],
    incy: usize,
    a: &mut [T],
    lda: usize,
) where
    T: ComplexFloat + lvl1::Shift + lvl1::Axpy,
{
    debug_assert!(lda >= 1.max(n));
    if alpha.is_zero() || n.is_zero() {
        return;
    }
    debug_assert!(x.len() >= (1 + (n - 1) * incx).max(1));
    debug_assert!(y.len() >= (1 + (n - 1) * incy).max(1));
    debug_assert!(a.len() >= n * lda);
    if incx.is_zero() && incy.is_zero() {
        let prod = alpha * x[0] * y[0].conj() + alpha.conj() * y[0] * x[0].conj();
        match (layout, uplo) {
            (Layout::RowMajor, UpLo::Lower) | (Layout::ColMajor, UpLo::Upper) => {
                for i in 0..n {
                    T::shift(i + 1, prod, &mut a[lda * i..], 1);
                }
            }
            (Layout::ColMajor, UpLo::Lower) | (Layout::RowMajor, UpLo::Upper) => {
                for i in 0..n {
                    T::shift(n - i, prod, &mut a[lda * i + i..], 1);
                }
            }
        }
    } else if incx.is_zero() {
        let (x0, x0bar) = (x[0], x[0].conj());
        match (layout, uplo) {
            (Layout::RowMajor, UpLo::Lower) => y
                .into_iter()
                .step_by(incy)
                .take(n)
                .enumerate()
                .for_each(|(i, &yi)| {
                    T::axpy(
                        n - i,
                        alpha.conj() * x0bar,
                        &y[i * incx..],
                        incy,
                        &mut a[i * lda + i..],
                        lda,
                    );
                    T::shift(n - i, alpha * yi.conj() * x0, &mut a[i * lda + i..], lda);
                }),
            (Layout::ColMajor, UpLo::Upper) => y
                .into_iter()
                .step_by(incy)
                .take(n)
                .enumerate()
                .for_each(|(i, yi)| {
                    T::axpy(i + 1, alpha.conj() * x0bar, y, incy, &mut a[i * lda..], 1);
                    T::shift(i + 1, alpha * yi.conj() * x0, &mut a[i * lda..], 1);
                }),
            (Layout::ColMajor, UpLo::Lower) => y
                .into_iter()
                .step_by(incy)
                .take(n)
                .enumerate()
                .for_each(|(i, yi)| {
                    T::axpy(
                        n - i,
                        alpha.conj() * x0bar,
                        &y[i * incy..],
                        incy,
                        &mut a[i * lda + i..],
                        1,
                    );
                    T::shift(n - i, alpha * yi.conj() * x0, &mut a[i * lda + i..], 1);
                }),
            (Layout::RowMajor, UpLo::Upper) => y
                .into_iter()
                .step_by(incy)
                .take(n)
                .enumerate()
                .for_each(|(i, yi)| {
                    T::axpy(i + 1, alpha.conj() * x0bar, y, incy, &mut a[i..], lda);
                    T::shift(i + 1, alpha * yi.conj() * x0, &mut a[i..], lda);
                }),
        }
    } else if incy.is_zero() {
        her2(layout, uplo, n, alpha, y, incy, x, incx, a, lda);
    } else {
        match (layout, uplo) {
            (Layout::RowMajor, UpLo::Lower) => x
                .into_iter()
                .step_by(incx)
                .zip(y.into_iter().step_by(incy))
                .take(n)
                .enumerate()
                .for_each(|(i, (xi, yi))| {
                    T::axpy(
                        n - i,
                        alpha.conj() * xi.conj(),
                        &y[i * incx..],
                        incy,
                        &mut a[i * lda + i..],
                        lda,
                    );
                    T::axpy(
                        n - i,
                        alpha * yi.conj(),
                        &x[i * incx..],
                        incx,
                        &mut a[i * lda + i..],
                        lda,
                    );
                }),
            (Layout::ColMajor, UpLo::Upper) => x
                .into_iter()
                .step_by(incx)
                .zip(y.into_iter().step_by(incy))
                .take(n)
                .enumerate()
                .for_each(|(i, (xi, yi))| {
                    T::axpy(
                        i + 1,
                        alpha.conj() * xi.conj(),
                        y,
                        incy,
                        &mut a[i * lda..],
                        1,
                    );
                    T::axpy(i + 1, alpha * yi.conj(), x, incx, &mut a[i * lda..], 1);
                }),
            (Layout::ColMajor, UpLo::Lower) => x
                .into_iter()
                .step_by(incx)
                .zip(y.into_iter().step_by(incy))
                .take(n)
                .enumerate()
                .for_each(|(i, (xi, yi))| {
                    T::axpy(
                        n - i,
                        alpha.conj() * xi.conj(),
                        &y[i * incy..],
                        incy,
                        &mut a[i * lda + i..],
                        1,
                    );
                    T::axpy(
                        n - i,
                        alpha * yi.conj(),
                        &x[i * incx..],
                        incx,
                        &mut a[i * lda + i..],
                        1,
                    );
                }),
            (Layout::RowMajor, UpLo::Upper) => x
                .into_iter()
                .step_by(incx)
                .zip(y.into_iter().step_by(incy))
                .take(n)
                .enumerate()
                .for_each(|(i, (xi, yi))| {
                    T::axpy(i + 1, alpha.conj() * xi.conj(), y, incy, &mut a[i..], lda);
                    T::axpy(i + 1, alpha * yi.conj(), x, incx, &mut a[i..], lda);
                }),
        }
    }
}
/// Rank-2 update of a hermitian matrix:
///   A := alpha * x * conj(y') + conj(alpha) * y * conj(x') + A
pub fn cher2(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: Complex32,
    x: &[Complex32],
    incx: usize,
    y: &[Complex32],
    incy: usize,
    a: &mut [Complex32],
    lda: usize,
) {
    her2(layout, uplo, n, alpha, x, incx, y, incy, a, lda)
}
/// Rank-2 update of a hermitian matrix:
///   A := alpha * x * conj(y') + conj(alpha) * y * conj(x') + A
pub fn zher2(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: Complex64,
    x: &[Complex64],
    incx: usize,
    y: &[Complex64],
    incy: usize,
    a: &mut [Complex64],
    lda: usize,
) {
    her2(layout, uplo, n, alpha, x, incx, y, incy, a, lda)
}
/// Performs a rank-1 update of a hermitian packed matrix.
///   A := alpha * x * conj(x') + A,
pub fn hpr<T>(layout: Layout, uplo: UpLo, n: usize, alpha: T, x: &[T], incx: usize, ap: &mut [T])
where
    T: ComplexFloat + lvl1::Shift + lvl1::Axpyc,
{
    if alpha.is_zero() || n.is_zero() {
        return;
    }
    debug_assert!(x.len() >= 1 + (n - 1) * incx);
    debug_assert!(ap.len() >= n * (n - 1) / 2);
    if incx.is_zero() {
        T::shift(n * (n - 1) / 2, alpha * x[0] * x[0].conj(), ap, 1);
    } else {
        match (layout, uplo) {
            (Layout::RowMajor, UpLo::Lower) => {
                let mut start = 0;
                for (i, &xi) in x.into_iter().step_by(incx).take(n).enumerate() {
                    T::axpyc(i + 1, alpha * xi, &x[..], incx, &mut ap[start..], 1);
                    start += i + 1;
                }
            }
            (Layout::ColMajor, UpLo::Upper) => {
                let mut start = 0;
                for (i, &xi) in x.into_iter().step_by(incx).take(n).enumerate() {
                    T::axpyu(i + 1, alpha * xi.conj(), &x[..], incx, &mut ap[start..], 1);
                    start += i + 1;
                }
            }
            (Layout::ColMajor, UpLo::Lower) => {
                let mut start = 0;
                for (i, &xi) in x.into_iter().step_by(incx).take(n).enumerate() {
                    T::axpyu(
                        n - i,
                        alpha * xi.conj(),
                        &x[i * incx..],
                        incx,
                        &mut ap[start..],
                        1,
                    );
                    start += n - i;
                }
            }
            (Layout::RowMajor, UpLo::Upper) => {
                let mut start = 0;
                for (i, &xi) in x.into_iter().step_by(incx).take(n).enumerate() {
                    T::axpyc(n - i, alpha * xi, &x[i * incx..], incx, &mut ap[start..], 1);
                    start += n - i;
                }
            }
        }
    }
}
pub fn chpr(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: Complex32,
    x: &[Complex32],
    incx: usize,
    ap: &mut [Complex32],
) {
    hpr(layout, uplo, n, alpha, x, incx, ap);
}
pub fn zhpr(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: Complex64,
    x: &[Complex64],
    incx: usize,
    ap: &mut [Complex64],
) {
    hpr(layout, uplo, n, alpha, x, incx, ap);
}
/// Computes a rank-2 update of a hermitian packed matrix.
///   A := alpha * x * conj(y') + conj(alpha) * y * conj(x') + A,
pub fn hpr2<T>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: T,
    x: &[T],
    incx: usize,
    y: &[T],
    incy: usize,
    ap: &mut [T],
) where
    T: ComplexFloat + lvl1::Shift + lvl1::Axpyc,
{
    if alpha.is_zero() || n.is_zero() {
        return;
    }
    debug_assert!(x.len() >= 1 + (n - 1) * incx);
    debug_assert!(ap.len() >= n * (n - 1) / 2);
    if incx.is_zero() && incy.is_zero() {
        T::shift(
            n * (n - 1) / 2,
            alpha * x[0] * y[0].conj() + alpha.conj() * y[0] * x[0].conj(),
            ap,
            1,
        );
    } else {
        match (layout, uplo) {
            (Layout::RowMajor, UpLo::Lower) => {
                let mut start = 0;
                for (i, (&xi, &yi)) in x
                    .into_iter()
                    .step_by(incx)
                    .zip(y.into_iter().step_by(incy))
                    .take(n)
                    .enumerate()
                {
                    T::axpyc(i + 1, alpha * xi, &y[..], incy, &mut ap[start..], 1);
                    T::axpyc(i + 1, alpha.conj() * yi, &x[..], incx, &mut ap[start..], 1);
                    start += i + 1;
                }
            }
            (Layout::ColMajor, UpLo::Upper) => {
                let mut start = 0;
                for (i, (&xi, &yi)) in x
                    .into_iter()
                    .step_by(incx)
                    .zip(y.into_iter().step_by(incy))
                    .take(n)
                    .enumerate()
                {
                    T::axpyc(i + 1, alpha * xi, &y[..], incy, &mut ap[start..], 1);
                    T::axpyc(i + 1, alpha.conj() * yi, &x[..], incx, &mut ap[start..], 1);
                    start += i + 1;
                }
            }
            (Layout::ColMajor, UpLo::Lower) => {
                let mut start = 0;
                for (i, (&xi, &yi)) in x
                    .into_iter()
                    .step_by(incx)
                    .zip(y.into_iter().step_by(incy))
                    .take(n)
                    .enumerate()
                {
                    T::axpyc(n - i, alpha * xi, &y[i * incx..], incy, &mut ap[start..], 1);
                    T::axpyc(
                        n - i,
                        alpha.conj() * yi,
                        &x[i * incx..],
                        incx,
                        &mut ap[start..],
                        1,
                    );
                    start += n - i;
                }
            }
            (Layout::RowMajor, UpLo::Upper) => {
                let mut start = 0;
                for (i, (&xi, &yi)) in x
                    .into_iter()
                    .step_by(incx)
                    .zip(y.into_iter().step_by(incy))
                    .take(n)
                    .enumerate()
                {
                    T::axpyc(n - i, alpha * xi, &y[i * incx..], incy, &mut ap[start..], 1);
                    T::axpyc(
                        n - i,
                        alpha.conj() * yi,
                        &x[i * incx..],
                        incx,
                        &mut ap[start..],
                        1,
                    );
                    start += n - i;
                }
            }
        }
    }
}
pub fn chpr2(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: Complex32,
    x: &[Complex32],
    incx: usize,
    y: &[Complex32],
    incy: usize,
    ap: &mut [Complex32],
) {
    hpr2(layout, uplo, n, alpha, x, incx, y, incy, ap);
}
pub fn zhpr2(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    alpha: Complex64,
    x: &[Complex64],
    incx: usize,
    y: &[Complex64],
    incy: usize,
    ap: &mut [Complex64],
) {
    hpr2(layout, uplo, n, alpha, x, incx, y, incy, ap);
}
/// Real triangular matrix . vector product
///  x := A * x  [NoTrans]
///  x := A' * x  [Trans | TransConj]
fn real_trmv<T>(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    a: &[T],
    lda: usize,
    x: &mut [T],
    incx: usize,
) where
    T: Float + lvl1::Dot + AddAssign<T>,
{
    if n.is_zero() {
        return;
    }
    debug_assert!(incx >= 1);
    debug_assert!(x.len() >= 1 + (n - 1) * incx);
    debug_assert!(lda >= n.max(1));
    debug_assert!(a.len() >= n * lda);
    match diag {
        Diag::Unit => match (layout, uplo, trans) {
            (Layout::RowMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::Trans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = 1;
                let mut ix = 0;
                for i in 0..n - 1 {
                    let new = T::dot(n - i - 1, &a[ia..], 1, &x[ix + incx..], incx);
                    x[ix] += new;
                    ia += lda + 1;
                    ix += incx;
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::Trans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ia = n * lda;
                let mut ix = n * incx;
                for i in 0..n - 1 {
                    ia -= lda;
                    ix -= incx;
                    let new = T::dot(n - i - 1, &a[ia..], 1, &x[..], incx);
                    x[ix] += new;
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::Trans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = lda;
                let mut ix = 0;
                for i in 0..n - 1 {
                    let new = T::dot(n - i - 1, &a[ia..], lda, &x[ix + incx..], incx);
                    x[ix] += new;
                    ia += lda + 1;
                    ix += incx;
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::Trans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ia = lda;
                let mut ix = n * incx;
                for i in 0..n {
                    ia -= 1;
                    ix -= incx;
                    let new = T::dot(n - i - 1, &a[ia..], lda, x, incx);
                    x[ix] += new;
                }
            }
        },
        Diag::NonUnit => match (layout, uplo, trans) {
            (Layout::RowMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::Trans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = 0;
                let mut ix = 0;
                for i in 0..n {
                    let new = T::dot(n - i, &a[ia..], 1, &x[ix..], incx);
                    x[ix] = new;
                    ia += lda + 1;
                    ix += incx;
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::Trans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ia = n * lda;
                let mut ix = n * incx;
                for i in 0..n {
                    ia -= lda;
                    ix -= incx;
                    let new = T::dot(n - i, &a[ia..], 1, x, incx);
                    x[ix] = new;
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::Trans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = 0;
                let mut ix = 0;
                for i in 0..n {
                    let new = T::dot(n - i, &a[ia..], lda, &x[ix..], incx);
                    x[ix] = new;
                    ia += lda + 1;
                    ix += incx;
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::Trans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ia = lda;
                let mut ix = n * incx;
                for i in 0..n {
                    ia -= 1;
                    ix -= incx;
                    let new = T::dot(n - i, &a[ia..], lda, x, incx);
                    x[ix] = new;
                }
            }
        },
    }
}
pub fn strmv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    a: &[f32],
    lda: usize,
    x: &mut [f32],
    incx: usize,
) {
    real_trmv(layout, uplo, trans, diag, n, a, lda, x, incx);
}
pub fn dtrmv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    a: &[f64],
    lda: usize,
    x: &mut [f64],
    incx: usize,
) {
    real_trmv(layout, uplo, trans, diag, n, a, lda, x, incx);
}
/// Complex triangular matrix . vector product
///  x := A * x         [NoTrans]
///  x := A' * x        [Trans]
///  x := conj(A') * x  [TransConj]
fn complex_trmv<T>(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    a: &[T],
    lda: usize,
    x: &mut [T],
    incx: usize,
) where
    T: ComplexFloat + lvl1::Dotc + AddAssign<T>,
{
    if n.is_zero() {
        return;
    }
    debug_assert!(incx >= 1);
    debug_assert!(x.len() >= 1 + (n - 1) * incx);
    debug_assert!(lda >= n.max(1));
    debug_assert!(a.len() >= n * lda);
    match diag {
        Diag::Unit => match (layout, uplo, trans) {
            (Layout::RowMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::Trans) => {
                let mut ia = 1;
                let mut ix = 0;
                for i in 0..n - 1 {
                    let new = T::dotu(n - i - 1, &a[ia..], 1, &x[ix + incx..], incx);
                    x[ix] += new;
                    ia += lda + 1;
                    ix += incx;
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::Trans) => {
                let mut ia = n * lda;
                let mut ix = n * incx;
                for i in 0..n - 1 {
                    ia -= lda;
                    ix -= incx;
                    let new = T::dotu(n - i - 1, &a[ia..], 1, &x[..], incx);
                    x[ix] += new;
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::Trans) => {
                let mut ia = lda;
                let mut ix = 0;
                for i in 0..n - 1 {
                    let new = T::dotu(n - i - 1, &a[ia..], lda, &x[ix + incx..], incx);
                    x[ix] += new;
                    ia += lda + 1;
                    ix += incx;
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::Trans) => {
                let mut ia = lda;
                let mut ix = n * incx;
                for i in 0..n {
                    ia -= 1;
                    ix -= incx;
                    let new = T::dotu(n - i - 1, &a[ia..], lda, x, incx);
                    x[ix] += new;
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = 1;
                let mut ix = 0;
                for i in 0..n - 1 {
                    let new = T::dotc(n - i - 1, &a[ia..], 1, &x[ix + incx..], incx);
                    x[ix] += new;
                    ia += lda + 1;
                    ix += incx;
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ia = n * lda;
                let mut ix = n * incx;
                for i in 0..n - 1 {
                    ia -= lda;
                    ix -= incx;
                    let new = T::dotc(n - i - 1, &a[ia..], 1, &x[..], incx);
                    x[ix] += new;
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = lda;
                let mut ix = 0;
                for i in 0..n - 1 {
                    let new = T::dotc(n - i - 1, &a[ia..], lda, &x[ix + incx..], incx);
                    x[ix] += new;
                    ia += lda + 1;
                    ix += incx;
                }
            }
            (Layout::RowMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ia = lda;
                let mut ix = n * incx;
                for i in 0..n {
                    ia -= 1;
                    ix -= incx;
                    let new = T::dotc(n - i - 1, &a[ia..], lda, x, incx);
                    x[ix] += new;
                }
            }
        },
        Diag::NonUnit => match (layout, uplo, trans) {
            (Layout::RowMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::Trans) => {
                let mut ia = 0;
                let mut ix = 0;
                for i in 0..n {
                    let new = T::dotu(n - i, &a[ia..], 1, &x[ix..], incx);
                    x[ix] = new;
                    ia += lda + 1;
                    ix += incx;
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::Trans) => {
                let mut ia = n * lda;
                let mut ix = n * incx;
                for i in 0..n {
                    ia -= lda;
                    ix -= incx;
                    let new = T::dotu(n - i, &a[ia..], 1, x, incx);
                    x[ix] = new;
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::Trans) => {
                let mut ia = 0;
                let mut ix = 0;
                for i in 0..n {
                    let new = T::dotu(n - i, &a[ia..], lda, &x[ix..], incx);
                    x[ix] = new;
                    ia += lda + 1;
                    ix += incx;
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::Trans) => {
                let mut ia = lda;
                let mut ix = n * incx;
                for i in 0..n {
                    ia -= 1;
                    ix -= incx;
                    let new = T::dotu(n - i, &a[ia..], lda, x, incx);
                    x[ix] = new;
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = 0;
                let mut ix = 0;
                for i in 0..n {
                    let new = T::dotc(n - i, &a[ia..], 1, &x[ix..], incx);
                    x[ix] = new;
                    ia += lda + 1;
                    ix += incx;
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ia = n * lda;
                let mut ix = n * incx;
                for i in 0..n {
                    ia -= lda;
                    ix -= incx;
                    let new = T::dotc(n - i, &a[ia..], 1, x, incx);
                    x[ix] = new;
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = 0;
                let mut ix = 0;
                for i in 0..n {
                    let new = T::dotc(n - i, &a[ia..], lda, &x[ix..], incx);
                    x[ix] = new;
                    ia += lda + 1;
                    ix += incx;
                }
            }
            (Layout::RowMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ia = lda;
                let mut ix = n * incx;
                for i in 0..n {
                    ia -= 1;
                    ix -= incx;
                    let new = T::dotc(n - i, &a[ia..], lda, x, incx);
                    x[ix] = new;
                }
            }
        },
    }
}
pub fn ctrmv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    a: &[Complex32],
    lda: usize,
    x: &mut [Complex32],
    incx: usize,
) {
    complex_trmv(layout, uplo, trans, diag, n, a, lda, x, incx);
}
pub fn ztrmv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    a: &[Complex64],
    lda: usize,
    x: &mut [Complex64],
    incx: usize,
) {
    complex_trmv(layout, uplo, trans, diag, n, a, lda, x, incx);
}
/// Resolution of a system from a real triangular matrix:
///   A . x = b   [NoTrans]
///   A' . x = b  [Trans | TransConj]
pub fn real_trsv<T>(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    a: &[T],
    lda: usize,
    x: &mut [T],
    incx: usize,
) where
    T: Float + lvl1::Dot + AddAssign<T> + SubAssign<T> + DivAssign<T>,
{
    if n.is_zero() {
        return;
    }
    debug_assert!(incx >= 1);
    debug_assert!(x.len() >= 1 + (n - 1) * incx);
    debug_assert!(lda >= n.max(1));
    debug_assert!(a.len() >= n * lda);
    match diag {
        Diag::Unit => match (layout, uplo, trans) {
            (Layout::RowMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::Trans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ix = (n - 1) * incx;
                let mut ia = n * n;
                for i in 1..n {
                    ix -= incx;
                    ia -= lda + 1;
                    x[ix] -= T::dot(i, &a[ia..], 1, &x[ix + incx..], incx);
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::Trans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ix = 0;
                let mut ia = 0;
                for i in 1..n {
                    ix += incx;
                    ia += lda;
                    x[ix] -= T::dot(i, &a[ia..], 1, x, incx);
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::Trans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ix = (n - 1) * incx;
                let mut ia = n * n + lda - 1;
                for i in 1..n {
                    ia -= lda + 1;
                    ix -= incx;
                    x[ix] -= T::dot(i, &a[ia..], lda, &x[ix + incx..], incx);
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::Trans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ix = 0;
                let mut ia = 0;
                for i in 1..n {
                    ix += incx;
                    ia += 1;
                    x[ix] -= T::dot(i, &a[ia..], lda, x, incx);
                }
            }
        },
        Diag::NonUnit => match (layout, uplo, trans) {
            (Layout::RowMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::Trans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ix = (n - 1) * incx;
                let mut ia = n * n - 1;
                x[ix] /= a[ia];
                for i in 1..n {
                    ix -= incx;
                    ia -= lda + 1;
                    x[ix] -= T::dot(i, &a[ia + 1..], 1, &x[ix + incx..], incx);
                    x[ix] /= a[ia];
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::Trans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ix = 0;
                let mut ia = 0;
                x[ix] /= a[ia];
                for i in 1..n {
                    ix += incx;
                    ia += lda;
                    x[ix] -= T::dot(i, &a[ia..], 1, x, incx);
                    x[ix] /= a[ia + i];
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::Trans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ix = (n - 1) * incx;
                let mut ia = n * n - 1;
                x[ix] /= a[ia];
                for i in 1..n {
                    ia -= lda + 1;
                    ix -= incx;
                    x[ix] -= T::dot(i, &a[ia + lda..], lda, &x[ix + incx..], incx);
                    x[ix] /= a[ia];
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::Trans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ix = 0;
                let mut ia = 0;
                x[0] /= a[0];
                for i in 1..n {
                    ia += 1;
                    ix += incx;
                    x[ix] -= T::dot(i, &a[ia..], lda, x, incx);
                    x[ix] /= a[ia + i * lda];
                }
            }
        },
    }
}
pub fn strsv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    a: &[f32],
    lda: usize,
    x: &mut [f32],
    incx: usize,
) {
    real_trsv(layout, uplo, trans, diag, n, a, lda, x, incx);
}
pub fn dtrsv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    a: &[f64],
    lda: usize,
    x: &mut [f64],
    incx: usize,
) {
    real_trsv(layout, uplo, trans, diag, n, a, lda, x, incx);
}
/// Resolution of a system from a complex triangular matrix:
///   A . x = b         [NoTrans]
///   A' . x = b        [Trans]
///   conj(A') . x = b  [ConjTrans]
fn complex_trsv<T>(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    a: &[T],
    lda: usize,
    x: &mut [T],
    incx: usize,
) where
    T: ComplexFloat + lvl1::Dotc + SubAssign<T> + DivAssign<T>,
{
    if n.is_zero() {
        return;
    }
    debug_assert!(incx >= 1);
    debug_assert!(x.len() >= 1 + (n - 1) * incx);
    debug_assert!(lda >= n.max(1));
    debug_assert!(a.len() >= n * lda);
    match diag {
        Diag::Unit => match (layout, uplo, trans) {
            (Layout::RowMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::Trans) => {
                let mut ix = (n - 1) * incx;
                let mut ia = n * n;
                for i in 1..n {
                    ix -= incx;
                    ia -= lda + 1;
                    x[ix] -= T::dotu(i, &a[ia..], 1, &x[ix + incx..], incx);
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ix = (n - 1) * incx;
                let mut ia = n * n;
                for i in 1..n {
                    ix -= incx;
                    ia -= lda + 1;
                    x[ix] -= T::dotc(i, &a[ia..], 1, &x[ix + incx..], incx);
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::Trans) => {
                let mut ix = 0;
                let mut ia = 0;
                for i in 1..n {
                    ix += incx;
                    ia += lda;
                    x[ix] -= T::dotu(i, &a[ia..], 1, x, incx);
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ix = 0;
                let mut ia = 0;
                for i in 1..n {
                    ix += incx;
                    ia += lda;
                    x[ix] -= T::dotc(i, &a[ia..], 1, x, incx);
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::Trans) => {
                let mut ix = (n - 1) * incx;
                let mut ia = n * n + lda - 1;
                for i in 1..n {
                    ia -= lda + 1;
                    ix -= incx;
                    x[ix] -= T::dotu(i, &a[ia..], lda, &x[ix + incx..], incx);
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ix = (n - 1) * incx;
                let mut ia = n * n + lda - 1;
                for i in 1..n {
                    ia -= lda + 1;
                    ix -= incx;
                    x[ix] -= T::dotc(i, &a[ia..], lda, &x[ix + incx..], incx);
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::Trans) => {
                let mut ix = 0;
                let mut ia = 0;
                for i in 1..n {
                    ix += incx;
                    ia += 1;
                    x[ix] -= T::dotu(i, &a[ia..], lda, x, incx);
                }
            }
            (Layout::RowMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ix = 0;
                let mut ia = 0;
                for i in 1..n {
                    ix += incx;
                    ia += 1;
                    x[ix] -= T::dotc(i, &a[ia..], lda, x, incx);
                }
            }
        },
        Diag::NonUnit => match (layout, uplo, trans) {
            (Layout::RowMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::Trans) => {
                let mut ix = (n - 1) * incx;
                let mut ia = n * n - 1;
                x[ix] /= a[ia];
                for i in 1..n {
                    ix -= incx;
                    ia -= lda + 1;
                    x[ix] -= T::dotu(i, &a[ia + 1..], 1, &x[ix + incx..], incx);
                    x[ix] /= a[ia];
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ix = (n - 1) * incx;
                let mut ia = n * n - 1;
                x[ix] /= a[ia].conj();
                for i in 1..n {
                    ix -= incx;
                    ia -= lda + 1;
                    x[ix] -= T::dotc(i, &a[ia + 1..], 1, &x[ix + incx..], incx);
                    x[ix] /= a[ia].conj();
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::Trans) => {
                let mut ix = 0;
                let mut ia = 0;
                x[ix] /= a[ia];
                for i in 1..n {
                    ix += incx;
                    ia += lda;
                    x[ix] -= T::dotu(i, &a[ia..], 1, x, incx);
                    x[ix] /= a[ia + i];
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ix = 0;
                let mut ia = 0;
                x[ix] /= a[ia].conj();
                for i in 1..n {
                    ix += incx;
                    ia += lda;
                    x[ix] -= T::dotc(i, &a[ia..], 1, x, incx);
                    x[ix] /= a[ia + i].conj();
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::Trans) => {
                let mut ix = (n - 1) * incx;
                let mut ia = n * n - 1;
                x[ix] /= a[ia];
                for i in 1..n {
                    ia -= lda + 1;
                    ix -= incx;
                    x[ix] -= T::dotu(i, &a[ia + lda..], lda, &x[ix + incx..], incx);
                    x[ix] /= a[ia];
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ix = (n - 1) * incx;
                let mut ia = n * n - 1;
                x[ix] /= a[ia].conj();
                for i in 1..n {
                    ia -= lda + 1;
                    ix -= incx;
                    x[ix] -= T::dotc(i, &a[ia + lda..], lda, &x[ix + incx..], incx);
                    x[ix] /= a[ia].conj();
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::Trans) => {
                let mut ix = 0;
                let mut ia = 0;
                x[0] /= a[0];
                for i in 1..n {
                    ia += 1;
                    ix += incx;
                    x[ix] -= T::dotu(i, &a[ia..], lda, x, incx);
                    x[ix] /= a[ia + i * lda];
                }
            }
            (Layout::RowMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ix = 0;
                let mut ia = 0;
                x[0] /= a[0].conj();
                for i in 1..n {
                    ia += 1;
                    ix += incx;
                    x[ix] -= T::dotc(i, &a[ia..], lda, x, incx);
                    x[ix] /= a[ia + i * lda].conj();
                }
            }
        },
    }
}
pub fn ctrsv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    a: &[Complex32],
    lda: usize,
    x: &mut [Complex32],
    incx: usize,
) {
    complex_trsv(layout, uplo, trans, diag, n, a, lda, x, incx);
}
pub fn ztrsv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    a: &[Complex64],
    lda: usize,
    x: &mut [Complex64],
    incx: usize,
) {
    complex_trsv(layout, uplo, trans, diag, n, a, lda, x, incx);
}
/// Real packed triangular matrix . vector product
///  x := A * x  [NoTrans]
///  x := A' * x  [Trans | TransConj]
pub fn real_tpmv<T>(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    ap: &[T],
    x: &mut [T],
    incx: usize,
) where
    T: Float + lvl1::Dot + lvl1::Axpy + AddAssign<T>,
{
    if n.is_zero() {
        return;
    }
    debug_assert!(incx >= 1);
    debug_assert!(x.len() >= 1 + (n - 1) * incx);
    debug_assert!(ap.len() >= n * (n + 1) / 2);
    match diag {
        Diag::Unit => match (layout, uplo, trans) {
            (Layout::RowMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::Trans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = 0;
                let mut ix = 0;
                for i in 0..n - 1 {
                    let new = T::dot(n - i - 1, &ap[ia + 1..], 1, &x[ix + incx..], incx);
                    x[ix] += new;
                    ia += n - i;
                    ix += incx;
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::Trans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ia = n * (n + 1) / 2;
                let mut ix = n * incx;
                for i in 0..n - 1 {
                    ia -= n - i;
                    ix -= incx;
                    let new = T::dot(n - i - 1, &ap[ia..], 1, &x[..], incx);
                    x[ix] += new;
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::Trans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = 0;
                let mut ix = 0;
                for i in 1..n {
                    ia += i;
                    ix += incx;
                    let u = x[ix];
                    T::axpy(i, u, &ap[ia..], 1, x, incx);
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::Trans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ia = n * (n + 1) / 2 - 1;
                let mut ix = (n - 1) * incx;
                for i in 1..n {
                    ia -= i + 1;
                    ix -= incx;
                    let u = x[ix];
                    T::axpy(i, u, &ap[ia + 1..], 1, &mut x[ix + incx..], incx);
                }
            }
        },
        Diag::NonUnit => match (layout, uplo, trans) {
            (Layout::RowMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::Trans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = 0;
                let mut ix = 0;
                for i in 0..n {
                    let new = T::dot(n - i, &ap[ia..], 1, &x[ix..], incx);
                    x[ix] = new;
                    ia += n - i;
                    ix += incx;
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::Trans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ia = n * (n + 1) / 2;
                let mut ix = n * incx;
                for i in 0..n {
                    ia -= n - i;
                    ix -= incx;
                    let new = T::dot(n - i, &ap[ia..], 1, &x[..], incx);
                    x[ix] = new;
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::Trans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ix = 0;
                let mut ia = 0;
                for i in 0..n {
                    let u = x[ix];
                    x[ix] = T::zero();
                    T::axpy(i + 1, u, &ap[ia..], 1, x, incx);
                    ia += i + 1;
                    ix += incx;
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::Trans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ia = n * (n + 1) / 2;
                let mut ix = n * incx;
                for i in 0..n {
                    ia -= i + 1;
                    ix -= incx;
                    let u = x[ix];
                    x[ix] = T::zero();
                    T::axpy(i + 1, u, &ap[ia..], 1, &mut x[ix..], incx);
                }
            }
        },
    }
}
pub fn stpmv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    ap: &[f32],
    x: &mut [f32],
    incx: usize,
) {
    real_tpmv(layout, uplo, trans, diag, n, ap, x, incx)
}
pub fn dtpmv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    ap: &[f64],
    x: &mut [f64],
    incx: usize,
) {
    real_tpmv(layout, uplo, trans, diag, n, ap, x, incx)
}
/// Real packed triangular matrix . vector product
///  x := A * x         [NoTrans]
///  x := A' * x        [Trans]
///  x := conj(A') * x  [TransConj]
pub fn complex_tpmv<T>(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    ap: &[T],
    x: &mut [T],
    incx: usize,
) where
    T: ComplexFloat + lvl1::Dotc + lvl1::Axpyc + AddAssign<T>,
{
    if n.is_zero() {
        return;
    }
    debug_assert!(incx >= 1);
    debug_assert!(x.len() >= 1 + (n - 1) * incx);
    debug_assert!(ap.len() >= n * (n + 1) / 2);
    match diag {
        Diag::Unit => match (layout, uplo, trans) {
            (Layout::RowMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::Trans) => {
                let mut ia = 0;
                let mut ix = 0;
                for i in 0..n - 1 {
                    let new = T::dotu(n - i - 1, &ap[ia + 1..], 1, &x[ix + incx..], incx);
                    x[ix] += new;
                    ia += n - i;
                    ix += incx;
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = 0;
                let mut ix = 0;
                for i in 0..n - 1 {
                    let new = T::dotc(n - i - 1, &ap[ia + 1..], 1, &x[ix + incx..], incx);
                    x[ix] += new;
                    ia += n - i;
                    ix += incx;
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::Trans) => {
                let mut ia = n * (n + 1) / 2;
                let mut ix = n * incx;
                for i in 0..n - 1 {
                    ia -= n - i;
                    ix -= incx;
                    let new = T::dotu(n - i - 1, &ap[ia..], 1, &x[..], incx);
                    x[ix] += new;
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ia = n * (n + 1) / 2;
                let mut ix = n * incx;
                for i in 0..n - 1 {
                    ia -= n - i;
                    ix -= incx;
                    let new = T::dotc(n - i - 1, &ap[ia..], 1, &x[..], incx);
                    x[ix] += new;
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::Trans) => {
                let mut ia = 0;
                let mut ix = 0;
                for i in 1..n {
                    ia += i;
                    ix += incx;
                    let u = x[ix];
                    T::axpyu(i, u, &ap[ia..], 1, x, incx);
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = 0;
                let mut ix = 0;
                for i in 1..n {
                    ia += i;
                    ix += incx;
                    let u = x[ix];
                    T::axpyc(i, u, &ap[ia..], 1, x, incx);
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::Trans) => {
                let mut ia = n * (n + 1) / 2 - 1;
                let mut ix = (n - 1) * incx;
                for i in 1..n {
                    ia -= i + 1;
                    ix -= incx;
                    let u = x[ix];
                    T::axpyu(i, u, &ap[ia + 1..], 1, &mut x[ix + incx..], incx);
                }
            }
            (Layout::RowMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ia = n * (n + 1) / 2 - 1;
                let mut ix = (n - 1) * incx;
                for i in 1..n {
                    ia -= i + 1;
                    ix -= incx;
                    let u = x[ix];
                    T::axpyc(i, u, &ap[ia + 1..], 1, &mut x[ix + incx..], incx);
                }
            }
        },
        Diag::NonUnit => match (layout, uplo, trans) {
            (Layout::RowMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::Trans) => {
                let mut ia = 0;
                let mut ix = 0;
                for i in 0..n {
                    let new = T::dotu(n - i, &ap[ia..], 1, &x[ix..], incx);
                    x[ix] += new;
                    ia += n - i;
                    ix += incx;
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = 0;
                let mut ix = 0;
                for i in 0..n {
                    let new = T::dotc(n - i, &ap[ia..], 1, &x[ix..], incx);
                    x[ix] += new;
                    ia += n - i;
                    ix += incx;
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::Trans) => {
                let mut ia = n * (n + 1) / 2;
                let mut ix = n * incx;
                for i in 0..n {
                    ia -= n - i;
                    ix -= incx;
                    let new = T::dotu(n - i, &ap[ia..], 1, &x[..], incx);
                    x[ix] += new;
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ia = n * (n + 1) / 2;
                let mut ix = n * incx;
                for i in 0..n {
                    ia -= n - i;
                    ix -= incx;
                    let new = T::dotc(n - i, &ap[ia..], 1, &x[..], incx);
                    x[ix] += new;
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::Trans) => {
                let mut ia = 0;
                for i in 0..n {
                    let u = x[i];
                    T::axpyu(i + 1, u, &ap[ia..], 1, x, incx);
                    ia += i + 1;
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = 0;
                for i in 0..n {
                    let u = x[i];
                    T::axpyc(i + 1, u, &ap[ia..], 1, x, incx);
                    ia += i + 1;
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::Trans) => {
                let mut ia = n * (n + 1) / 2;
                let mut ix = n * incx;
                for i in 0..n {
                    ia -= n - i;
                    ix -= incx;
                    let u = x[i];
                    T::axpyu(n - i, u, &ap[ia..], 1, &mut x[ix..], incx);
                }
            }
            (Layout::RowMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ia = n * (n + 1) / 2;
                let mut ix = n * incx;
                for i in 0..n {
                    ia -= n - i;
                    ix -= incx;
                    let u = x[i];
                    T::axpyc(n - i, u, &ap[ia..], 1, &mut x[ix..], incx);
                }
            }
        },
    }
}
pub fn ctpmv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    ap: &[Complex32],
    x: &mut [Complex32],
    incx: usize,
) {
    complex_tpmv(layout, uplo, trans, diag, n, ap, x, incx)
}
pub fn ztpmv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    ap: &[Complex64],
    x: &mut [Complex64],
    incx: usize,
) {
    complex_tpmv(layout, uplo, trans, diag, n, ap, x, incx)
}
/// Resolution of a system from a packed real triangular matrix:
///   A . x = b   [NoTrans]
///   A' . x = b  [Trans | TransConj]
pub fn real_tpsv<T>(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    ap: &[T],
    x: &mut [T],
    incx: usize,
) where
    T: Float + lvl1::Dot + lvl1::Axpy + AddAssign<T> + SubAssign<T> + DivAssign<T>,
{
    if n.is_zero() {
        return;
    }
    debug_assert!(incx >= 1);
    debug_assert!(x.len() >= 1 + (n - 1) * incx);
    debug_assert!(ap.len() >= n * (n + 1) / 2);
    match diag {
        Diag::Unit => match (layout, uplo, trans) {
            (Layout::RowMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::Trans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = n * (n + 1) / 2 - 1;
                let mut ix = (n - 1) * incx;
                for i in 1..n {
                    ix -= incx;
                    ia -= i + 1;
                    x[ix] -= T::dot(i, &ap[ia + 1..], 1, &x[ix + incx..], incx);
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::Trans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ia = 0;
                let mut ix = 0;
                for i in 1..n {
                    ix += incx;
                    ia += i;
                    x[ix] -= T::dot(i, &ap[ia..], 1, &x[..], incx);
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::Trans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = n * (n + 1) / 2;
                let mut ix = (n - 1) * incx;
                for i in 1..n {
                    ia -= n - i + 1;
                    let u = x[ix];
                    T::axpy(n - i, -u, &ap[ia..], 1, x, incx);
                    ix -= incx;
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::Trans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ix = 0;
                let mut ia = 0;
                for i in 1..n {
                    let u = x[ix];
                    ix += incx;
                    T::axpy(n - i, -u, &ap[ia + 1..], 1, &mut x[ix..], incx);
                    ia += n - i + 1;
                }
            }
        },
        Diag::NonUnit => match (layout, uplo, trans) {
            (Layout::RowMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::Trans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = n * (n + 1) / 2 - 1;
                let mut ix = (n - 1) * incx;
                x[ix] /= ap[ia];
                for i in 1..n {
                    ix -= incx;
                    ia -= i + 1;
                    x[ix] -= T::dot(i, &ap[ia + 1..], 1, &x[ix + incx..], incx);
                    x[ix] /= ap[ia];
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::Trans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ia = 0;
                let mut ix = 0;
                x[ix] /= ap[ia];
                for i in 1..n {
                    ix += incx;
                    ia += i;
                    x[ix] -= T::dot(i, &ap[ia..], 1, &x[..], incx);
                    x[ix] /= ap[ia + i];
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::Trans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = n * (n + 1) / 2;
                let mut ix = (n - 1) * incx;
                x[ix] /= ap[ia - 1];
                for i in 1..n {
                    ia -= n - i + 1;
                    let u = x[ix];
                    T::axpy(n - i, -u, &ap[ia..], 1, x, incx);
                    ix -= incx;
                    x[ix] /= ap[ia - 1];
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::Trans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ix = 0;
                let mut ia = 0;
                x[ix] /= ap[ia];
                for i in 1..n {
                    let u = x[ix];
                    ix += incx;
                    T::axpy(n - i, -u, &ap[ia + 1..], 1, &mut x[ix..], incx);
                    ia += n - i + 1;
                    x[ix] /= ap[ia];
                }
            }
        },
    }
}
pub fn stpsv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    ap: &[f32],
    x: &mut [f32],
    incx: usize,
) {
    real_tpsv(layout, uplo, trans, diag, n, ap, x, incx);
}
pub fn dtpsv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    ap: &[f64],
    x: &mut [f64],
    incx: usize,
) {
    real_tpsv(layout, uplo, trans, diag, n, ap, x, incx);
}
/// Resolution of a system from a packed complex triangular matrix:
///   A . x = b         [NoTrans]
///   A' . x = b        [Trans]
///   conj(A') . x = b  [ConjTrans]
fn complex_tpsv<T>(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    ap: &[T],
    x: &mut [T],
    incx: usize,
) where
    T: ComplexFloat + lvl1::Dotc + lvl1::Axpyc + SubAssign<T> + DivAssign<T>,
{
    if n.is_zero() {
        return;
    }
    debug_assert!(incx >= 1);
    debug_assert!(x.len() >= 1 + (n - 1) * incx);
    debug_assert!(ap.len() >= n * (n + 1) / 2);
    match diag {
        Diag::Unit => match (layout, uplo, trans) {
            (Layout::RowMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::Trans) => {
                let mut ia = n * (n + 1) / 2 - 1;
                let mut ix = (n - 1) * incx;
                for i in 1..n {
                    ix -= incx;
                    ia -= i + 1;
                    x[ix] -= T::dotu(i, &ap[ia + 1..], 1, &x[ix + incx..], incx);
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = n * (n + 1) / 2 - 1;
                let mut ix = (n - 1) * incx;
                for i in 1..n {
                    ix -= incx;
                    ia -= i + 1;
                    x[ix] -= T::dotc(i, &ap[ia + 1..], 1, &x[ix + incx..], incx);
                }

            }
            (Layout::RowMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::Trans) => {
                let mut ia = 0;
                let mut ix = 0;
                for i in 1..n {
                    ix += incx;
                    ia += i;
                    x[ix] -= T::dotu(i, &ap[ia..], 1, &x[..], incx);
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ia = 0;
                let mut ix = 0;
                for i in 1..n {
                    ix += incx;
                    ia += i;
                    x[ix] -= T::dotc(i, &ap[ia..], 1, &x[..], incx);
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::Trans) => {
                let mut ia = n * (n + 1) / 2;
                let mut ix = (n - 1) * incx;
                for i in 1..n {
                    ia -= n - i + 1;
                    let u = x[ix];
                    T::axpyu(n - i, -u, &ap[ia..], 1, x, incx);
                    ix -= incx;
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = n * (n + 1) / 2;
                let mut ix = (n - 1) * incx;
                for i in 1..n {
                    ia -= n - i + 1;
                    let u = x[ix];
                    T::axpyc(n - i, -u, &ap[ia..], 1, x, incx);
                    ix -= incx;
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::Trans) => {
                let mut ix = 0;
                let mut ia = 0;
                for i in 1..n {
                    let u = x[ix];
                    ix += incx;
                    T::axpyu(n - i, -u, &ap[ia + 1..], 1, &mut x[ix..], incx);
                    ia += n - i + 1;
                }
            }
            (Layout::RowMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ix = 0;
                let mut ia = 0;
                for i in 1..n {
                    let u = x[ix];
                    ix += incx;
                    T::axpyc(n - i, -u, &ap[ia + 1..], 1, &mut x[ix..], incx);
                    ia += n - i + 1;
                }
            }
        },
        Diag::NonUnit => match (layout, uplo, trans) {
            (Layout::RowMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Lower, Transpose::Trans) => {
                let mut ia = n * (n + 1) / 2 - 1;
                let mut ix = (n - 1) * incx;
                x[ix] /= ap[ia];
                for i in 1..n {
                    ix -= incx;
                    ia -= i + 1;
                    x[ix] -= T::dotu(i, &ap[ia + 1..], 1, &x[ix + incx..], incx);
                    x[ix] /= ap[ia];
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = n * (n + 1) / 2 - 1;
                let mut ix = (n - 1) * incx;
                x[ix] /= ap[ia].conj();
                for i in 1..n {
                    ix -= incx;
                    ia -= i + 1;
                    x[ix] -= T::dotc(i, &ap[ia + 1..], 1, &x[ix + incx..], incx);
                    x[ix] /= ap[ia].conj();
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::ColMajor, UpLo::Upper, Transpose::Trans) => {
                let mut ia = 0;
                let mut ix = 0;
                x[ix] /= ap[ia];
                for i in 1..n {
                    ix += incx;
                    ia += i;
                    x[ix] -= T::dotu(i, &ap[ia..], 1, &x[..], incx);
                    x[ix] /= ap[ia + i];
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ia = 0;
                let mut ix = 0;
                x[ix] /= ap[ia].conj();
                for i in 1..n {
                    ix += incx;
                    ia += i;
                    x[ix] -= T::dotc(i, &ap[ia..], 1, &x[..], incx);
                    x[ix] /= ap[ia + i].conj();
                }
            }
            (Layout::ColMajor, UpLo::Upper, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Lower, Transpose::Trans) => {
                let mut ia = n * (n + 1) / 2;
                let mut ix = (n - 1) * incx;
                x[ix] /= ap[ia - 1];
                for i in 1..n {
                    ia -= n - i + 1;
                    let u = x[ix];
                    T::axpyu(n - i, -u, &ap[ia..], 1, x, incx);
                    ix -= incx;
                    x[ix] /= ap[ia - 1];
                }
            }
            (Layout::RowMajor, UpLo::Lower, Transpose::ConjTrans) => {
                let mut ia = n * (n + 1) / 2;
                let mut ix = (n - 1) * incx;
                x[ix] /= ap[ia - 1].conj();
                for i in 1..n {
                    ia -= n - i + 1;
                    let u = x[ix];
                    T::axpyc(n - i, -u, &ap[ia..], 1, x, incx);
                    ix -= incx;
                    x[ix] /= ap[ia - 1].conj();
                }
            }
            (Layout::ColMajor, UpLo::Lower, Transpose::NoTrans)
            | (Layout::RowMajor, UpLo::Upper, Transpose::Trans) => {
                let mut ix = 0;
                let mut ia = 0;
                x[ix] /= ap[ia];
                for i in 1..n {
                    let u = x[ix];
                    ix += incx;
                    T::axpyu(n - i, -u, &ap[ia + 1..], 1, &mut x[ix..], incx);
                    ia += n - i + 1;
                    x[ix] /= ap[ia];
                }
            }
            (Layout::RowMajor, UpLo::Upper, Transpose::ConjTrans) => {
                let mut ix = 0;
                let mut ia = 0;
                x[ix] /= ap[ia].conj();
                for i in 1..n {
                    let u = x[ix];
                    ix += incx;
                    T::axpyc(n - i, -u, &ap[ia + 1..], 1, &mut x[ix..], incx);
                    ia += n - i + 1;
                    x[ix] /= ap[ia].conj();
                }
            }
        },
    }
}
pub fn ctpsv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    ap: &[Complex32],
    x: &mut [Complex32],
    incx: usize,
) {
    complex_tpsv(layout, uplo, trans, diag, n, ap, x, incx);
}
pub fn ztpsv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    ap: &[Complex64],
    x: &mut [Complex64],
    incx: usize,
) {
    complex_tpsv(layout, uplo, trans, diag, n, ap, x, incx);
}
/// Triangular band matrix . vector product
///   x := A . x  [NoTrans]
///   X := A' . x [Trans | ConjTrans]
pub fn real_tbmv<T>(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    k: usize,
    a: &[T],
    lda: usize,
    x: &mut [T],
    incx: usize,
) where
    T: Float + lvl1::Dot,
{
    if n.is_zero() {
        return;
    }
    debug_assert!(incx >= 1);
    debug_assert!(x.len() >= 1 + (n - 1) * incx);
    debug_assert!(lda >= k + 1);
    debug_assert!(a.len() >= n * lda);
    unimplemented!("TODO");
}
pub fn stbmv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    x: &mut [f32],
    incx: usize,
) {
    real_tbmv(layout, uplo, trans, diag, n, k, a, lda, x, incx)
}
pub fn dtbmv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    x: &mut [f64],
    incx: usize,
) {
    real_tbmv(layout, uplo, trans, diag, n, k, a, lda, x, incx)
}
/// Triangular band matrix . vector product
///   x := A . x         [NoTrans]
///   x := A' . x        [Trans]
///   x := conj(A') . x  [ConjTrans]
pub fn complex_tbmv<T>(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    k: usize,
    a: &[T],
    lda: usize,
    x: &mut [T],
    incx: usize,
) where
    T: ComplexFloat,
{
    unimplemented!("TODO");
}
pub fn ctbmv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    k: usize,
    a: &[Complex32],
    lda: usize,
    x: &mut [Complex32],
    incx: usize,
) {
    complex_tbmv(layout, uplo, trans, diag, n, k, a, lda, x, incx)
}
pub fn ztbmv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    k: usize,
    a: &[Complex64],
    lda: usize,
    x: &mut [Complex64],
    incx: usize,
) {
    complex_tbmv(layout, uplo, trans, diag, n, k, a, lda, x, incx)
}
/// Solves a system given by a triangular band matrix product
///   A . x = b  [NoTrans]
///   A' . x = b [Trans | ConjTrans]
pub fn real_tbsv<T>(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    k: usize,
    a: &[T],
    lda: usize,
    x: &mut [T],
    incx: usize,
) where
    T: Float,
{
    unimplemented!("TODO");
}
pub fn stbsv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    x: &mut [f32],
    incx: usize,
) {
    real_tbsv(layout, uplo, trans, diag, n, k, a, lda, x, incx)
}
pub fn dtbsv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    x: &mut [f64],
    incx: usize,
) {
    real_tbsv(layout, uplo, trans, diag, n, k, a, lda, x, incx)
}
/// Solves a system given by a triangular band matrix product
///   A . x = b          [NoTrans]
///   A' . x = b         [Trans]
///   conj(A') . x = b   [ConjTrans]
pub fn complex_tbsv<T>(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    k: usize,
    a: &[T],
    lda: usize,
    x: &mut [T],
    incx: usize,
) where
    T: ComplexFloat,
{
    unimplemented!("TODO");
}
pub fn ctbsv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    k: usize,
    a: &[Complex32],
    lda: usize,
    x: &mut [Complex32],
    incx: usize,
) {
    complex_tbsv(layout, uplo, trans, diag, n, k, a, lda, x, incx)
}
pub fn ztbsv(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    k: usize,
    a: &[Complex64],
    lda: usize,
    x: &mut [Complex64],
    incx: usize,
) {
    complex_tbsv(layout, uplo, trans, diag, n, k, a, lda, x, incx)
}
/// General matrix rank1 update
pub trait Ger: Sized {
    /// General update
    fn ger(
        layout: Layout,
        m: usize,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        a: &mut [Self],
        lda: usize,
    );
}
pub trait Gerc: Sized {
    /// General update without conjugate
    fn geru(
        layout: Layout,
        m: usize,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        a: &mut [Self],
        lda: usize,
    );
    /// General update with conjugate
    fn gerc(
        layout: Layout,
        m: usize,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        a: &mut [Self],
        lda: usize,
    );
}
/// General matrix-vector product
pub trait Gemv: Sized {
    /// General matrix-vector product
    fn gemv(
        layout: Layout,
        trans: Transpose,
        m: usize,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        x: &[Self],
        incx: usize,
        beta: Self,
        y: &mut [Self],
        incy: usize,
    );
}
impl Ger for f32 {
    fn ger(
        layout: Layout,
        m: usize,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        a: &mut [Self],
        lda: usize,
    ) {
        sger(layout, m, n, alpha, x, incx, y, incy, a, lda)
    }
}
impl Gemv for f32 {
    fn gemv(
        layout: Layout,
        trans: Transpose,
        m: usize,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        x: &[Self],
        incx: usize,
        beta: Self,
        y: &mut [Self],
        incy: usize,
    ) {
        sgemv(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    }
}
impl Ger for f64 {
    fn ger(
        layout: Layout,
        m: usize,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        a: &mut [Self],
        lda: usize,
    ) {
        dger(layout, m, n, alpha, x, incx, y, incy, a, lda)
    }
}
impl Gemv for f64 {
    fn gemv(
        layout: Layout,
        trans: Transpose,
        m: usize,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        x: &[Self],
        incx: usize,
        beta: Self,
        y: &mut [Self],
        incy: usize,
    ) {
        dgemv(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    }
}
impl Ger for Complex32 {
    fn ger(
        layout: Layout,
        m: usize,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        a: &mut [Self],
        lda: usize,
    ) {
        cgeru(layout, m, n, alpha, x, incx, y, incy, a, lda)
    }
}
impl Gemv for Complex32 {
    fn gemv(
        layout: Layout,
        trans: Transpose,
        m: usize,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        x: &[Self],
        incx: usize,
        beta: Self,
        y: &mut [Self],
        incy: usize,
    ) {
        cgemv(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    }
}
impl Ger for Complex64 {
    fn ger(
        layout: Layout,
        m: usize,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        a: &mut [Self],
        lda: usize,
    ) {
        zgeru(layout, m, n, alpha, x, incx, y, incy, a, lda)
    }
}
impl Gemv for Complex64 {
    fn gemv(
        layout: Layout,
        trans: Transpose,
        m: usize,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        x: &[Self],
        incx: usize,
        beta: Self,
        y: &mut [Self],
        incy: usize,
    ) {
        zgemv(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    }
}
impl Gerc for Complex32 {
    fn geru(
        layout: Layout,
        m: usize,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        a: &mut [Self],
        lda: usize,
    ) {
        cgeru(layout, m, n, alpha, x, incx, y, incy, a, lda)
    }
    fn gerc(
        layout: Layout,
        m: usize,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        a: &mut [Self],
        lda: usize,
    ) {
        cgerc(layout, m, n, alpha, x, incx, y, incy, a, lda)
    }
}
impl Gerc for Complex64 {
    fn geru(
        layout: Layout,
        m: usize,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        a: &mut [Self],
        lda: usize,
    ) {
        zgeru(layout, m, n, alpha, x, incx, y, incy, a, lda)
    }
    fn gerc(
        layout: Layout,
        m: usize,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        a: &mut [Self],
        lda: usize,
    ) {
        zgerc(layout, m, n, alpha, x, incx, y, incy, a, lda)
    }
}
/// Symmetric level2
pub trait Symmetric: Sized {
    /// Symmetric rank-1 update
    fn syr(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        a: &mut [Self],
        lda: usize,
    );
    /// Symmetric rank-2 update
    fn syr2(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        a: &mut [Self],
        lda: usize,
    );
    /// Symmetric matrix-vector product
    fn symv(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        x: &[Self],
        incx: usize,
        beta: Self,
        y: &mut [Self],
        incy: usize,
    );
    /// Packed symmetric rank-1 update
    fn spr(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        ap: &mut [Self],
    );
    /// Packed symmetric rank-2 update
    fn spr2(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        ap: &mut [Self],
    );
    /// Symmetric packed matrix-vector product
    fn spmv(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        ap: &[Self],
        x: &[Self],
        incx: usize,
        beta: Self,
        y: &mut [Self],
        incy: usize,
    );
}
impl Symmetric for f32 {
    fn syr(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        a: &mut [Self],
        lda: usize,
    ) {
        ssyr(layout, uplo, n, alpha, x, incx, a, lda)
    }
    fn syr2(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        a: &mut [Self],
        lda: usize,
    ) {
        ssyr2(layout, uplo, n, alpha, x, incx, y, incy, a, lda)
    }
    fn symv(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        x: &[Self],
        incx: usize,
        beta: Self,
        y: &mut [Self],
        incy: usize,
    ) {
        ssymv(layout, uplo, n, alpha, a, lda, x, incx, beta, y, incy)
    }
    /// Packed symmetric rank-1 update
    fn spr(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        ap: &mut [Self],
    ) {
        sspr(layout, uplo, n, alpha, x, incx, ap);
    }
    /// Packed symmetric rank-2 update
    fn spr2(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        ap: &mut [Self],
    ) {
        sspr2(layout, uplo, n, alpha, x, incx, y, incy, ap);
    }
    fn spmv(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        ap: &[Self],
        x: &[Self],
        incx: usize,
        beta: Self,
        y: &mut [Self],
        incy: usize,
    ) {
        sspmv(layout, uplo, n, alpha, ap, x, incx, beta, y, incy);
    }
}
impl Symmetric for f64 {
    fn syr(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        a: &mut [Self],
        lda: usize,
    ) {
        dsyr(layout, uplo, n, alpha, x, incx, a, lda)
    }
    fn syr2(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        a: &mut [Self],
        lda: usize,
    ) {
        dsyr2(layout, uplo, n, alpha, x, incx, y, incy, a, lda)
    }
    fn symv(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        x: &[Self],
        incx: usize,
        beta: Self,
        y: &mut [Self],
        incy: usize,
    ) {
        dsymv(layout, uplo, n, alpha, a, lda, x, incx, beta, y, incy)
    }
    /// Packed symmetric rank-1 update
    fn spr(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        ap: &mut [Self],
    ) {
        dspr(layout, uplo, n, alpha, x, incx, ap);
    }
    /// Packed symmetric rank-2 update
    fn spr2(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        ap: &mut [Self],
    ) {
        dspr2(layout, uplo, n, alpha, x, incx, y, incy, ap);
    }
    fn spmv(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        ap: &[Self],
        x: &[Self],
        incx: usize,
        beta: Self,
        y: &mut [Self],
        incy: usize,
    ) {
        dspmv(layout, uplo, n, alpha, ap, x, incx, beta, y, incy)
    }
}
/// Complex matrix level2
pub trait Hermitian: Sized {
    /// Hermitian rank-1 update
    fn her(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        a: &mut [Self],
        lda: usize,
    );
    /// Hermitian rank-2 update
    fn her2(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        a: &mut [Self],
        lda: usize,
    );
    /// Hermitian matrix-vector product
    fn hemv(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        x: &[Self],
        incx: usize,
        beta: Self,
        y: &mut [Self],
        incy: usize,
    );
    /// Packed hermitian rank-1 update
    fn hpr(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        a: &mut [Self],
    );
    /// Packed hermitian rank-2 update
    fn hpr2(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        a: &mut [Self],
    );
    /// Hermitian packed matrix-vector product
    fn hpmv(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        ap: &[Self],
        x: &[Self],
        incx: usize,
        beta: Self,
        y: &mut [Self],
        incy: usize,
    );
}
impl Hermitian for Complex32 {
    /// Symmetric rank-1 update
    fn her(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        a: &mut [Self],
        lda: usize,
    ) {
        cher(layout, uplo, n, alpha, x, incx, a, lda)
    }
    /// Symmetric rank-2 update
    fn her2(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        a: &mut [Self],
        lda: usize,
    ) {
        cher2(layout, uplo, n, alpha, x, incx, y, incy, a, lda)
    }
    /// Symmetric matrix-vector product
    fn hemv(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        x: &[Self],
        incx: usize,
        beta: Self,
        y: &mut [Self],
        incy: usize,
    ) {
        chemv(layout, uplo, n, alpha, a, lda, x, incx, beta, y, incy)
    }
    /// Packed hermitian rank-1 update
    fn hpr(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        a: &mut [Self],
    ) {
        chpr(layout, uplo, n, alpha, x, incx, a);
    }
    /// Packed hermitian rank-2 update
    fn hpr2(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        a: &mut [Self],
    ) {
        chpr2(layout, uplo, n, alpha, x, incx, y, incy, a);
    }
    /// Hermitian packed matrix-vector product
    fn hpmv(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        ap: &[Self],
        x: &[Self],
        incx: usize,
        beta: Self,
        y: &mut [Self],
        incy: usize,
    ) {
        chpmv(layout, uplo, n, alpha, ap, x, incx, beta, y, incy)
    }
}
impl Hermitian for Complex64 {
    /// Symmetric rank-1 update
    fn her(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        a: &mut [Self],
        lda: usize,
    ) {
        zher(layout, uplo, n, alpha, x, incx, a, lda)
    }
    /// Symmetric rank-2 update
    fn her2(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        a: &mut [Self],
        lda: usize,
    ) {
        zher2(layout, uplo, n, alpha, x, incx, y, incy, a, lda)
    }
    /// Symmetric matrix-vector product
    fn hemv(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        x: &[Self],
        incx: usize,
        beta: Self,
        y: &mut [Self],
        incy: usize,
    ) {
        zhemv(layout, uplo, n, alpha, a, lda, x, incx, beta, y, incy)
    }
    /// Packed hermitian rank-1 update
    fn hpr(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        a: &mut [Self],
    ) {
        zhpr(layout, uplo, n, alpha, x, incx, a);
    }
    /// Packed hermitian rank-2 update
    fn hpr2(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        x: &[Self],
        incx: usize,
        y: &[Self],
        incy: usize,
        a: &mut [Self],
    ) {
        zhpr2(layout, uplo, n, alpha, x, incx, y, incy, a);
    }
    /// Hermitian packed matrix-vector product
    fn hpmv(
        layout: Layout,
        uplo: UpLo,
        n: usize,
        alpha: Self,
        ap: &[Self],
        x: &[Self],
        incx: usize,
        beta: Self,
        y: &mut [Self],
        incy: usize,
    ) {
        zhpmv(layout, uplo, n, alpha, ap, x, incx, beta, y, incy)
    }
}
/// Triangular matrices
pub trait Triangular: Sized {
    /// Triangular matrix . vector product
    ///  x := A . x
    fn trmv(
        layout: Layout,
        uplo: UpLo,
        trans: Transpose,
        diag: Diag,
        n: usize,
        a: &[Self],
        lda: usize,
        x: &mut [Self],
        incx: usize,
    ) -> ();
    /// Triangular packed matrix . vector product
    ///  x := A . x
    fn tpmv(
        layout: Layout,
        uplo: UpLo,
        trans: Transpose,
        diag: Diag,
        n: usize,
        ap: &[Self],
        x: &mut [Self],
        incx: usize,
    ) -> ();
    /// Solves a triangular system:
    ///  A . x = b
    fn trsv(
        layout: Layout,
        uplo: UpLo,
        trans: Transpose,
        diag: Diag,
        n: usize,
        a: &[Self],
        lda: usize,
        x: &mut [Self],
        incx: usize,
    ) -> ();
    /// Solves a packed triangular system:
    ///  A . x = b
    fn tpsv(
        layout: Layout,
        uplo: UpLo,
        trans: Transpose,
        diag: Diag,
        n: usize,
        ap: &[Self],
        x: &mut [Self],
        incx: usize,
    ) -> ();
}
impl Triangular for f32 {
    fn trmv(
        layout: Layout,
        uplo: UpLo,
        trans: Transpose,
        diag: Diag,
        n: usize,
        a: &[Self],
        lda: usize,
        x: &mut [Self],
        incx: usize,
    ) -> () {
        strmv(layout, uplo, trans, diag, n, a, lda, x, incx);
    }
    fn tpmv(
        layout: Layout,
        uplo: UpLo,
        trans: Transpose,
        diag: Diag,
        n: usize,
        ap: &[Self],
        x: &mut [Self],
        incx: usize,
    ) -> () {
        stpmv(layout, uplo, trans, diag, n, ap, x, incx);
    }
    fn trsv(layout: Layout, uplo: UpLo, trans: Transpose, diag: Diag, n: usize, a: &[Self], lda: usize, x: &mut [Self], incx: usize) -> () {
        strsv(layout, uplo, trans, diag, n, a, lda, x, incx);
    }
    fn tpsv(layout: Layout, uplo: UpLo, trans: Transpose, diag: Diag, n: usize, ap: &[Self], x: &mut [Self], incx: usize) -> () {
        stpsv(layout, uplo, trans, diag, n, ap, x, incx);
    }
}
impl Triangular for f64 {
    fn trmv(
        layout: Layout,
        uplo: UpLo,
        trans: Transpose,
        diag: Diag,
        n: usize,
        a: &[Self],
        lda: usize,
        x: &mut [Self],
        incx: usize,
    ) -> () {
        dtrmv(layout, uplo, trans, diag, n, a, lda, x, incx);
    }
    fn tpmv(
        layout: Layout,
        uplo: UpLo,
        trans: Transpose,
        diag: Diag,
        n: usize,
        ap: &[Self],
        x: &mut [Self],
        incx: usize,
    ) -> () {
        dtpmv(layout, uplo, trans, diag, n, ap, x, incx);
    }
    fn trsv(layout: Layout, uplo: UpLo, trans: Transpose, diag: Diag, n: usize, a: &[Self], lda: usize, x: &mut [Self], incx: usize) -> () {
        dtrsv(layout, uplo, trans, diag, n, a, lda, x, incx);
    }
    fn tpsv(layout: Layout, uplo: UpLo, trans: Transpose, diag: Diag, n: usize, ap: &[Self], x: &mut [Self], incx: usize) -> () {
        dtpsv(layout, uplo, trans, diag, n, ap, x, incx);
    }
}
impl Triangular for Complex32 {
    fn trmv(
        layout: Layout,
        uplo: UpLo,
        trans: Transpose,
        diag: Diag,
        n: usize,
        a: &[Self],
        lda: usize,
        x: &mut [Self],
        incx: usize,
    ) -> () {
        ctrmv(layout, uplo, trans, diag, n, a, lda, x, incx);
    }
    fn tpmv(
        layout: Layout,
        uplo: UpLo,
        trans: Transpose,
        diag: Diag,
        n: usize,
        ap: &[Self],
        x: &mut [Self],
        incx: usize,
    ) -> () {
        ctpmv(layout, uplo, trans, diag, n, ap, x, incx);
    }
    fn trsv(layout: Layout, uplo: UpLo, trans: Transpose, diag: Diag, n: usize, a: &[Self], lda: usize, x: &mut [Self], incx: usize) -> () {
        ctrsv(layout, uplo, trans, diag, n, a, lda, x, incx);
    }
    fn tpsv(layout: Layout, uplo: UpLo, trans: Transpose, diag: Diag, n: usize, ap: &[Self], x: &mut [Self], incx: usize) -> () {
        ctpsv(layout, uplo, trans, diag, n, ap, x, incx);
    }
}
impl Triangular for Complex64 {
    fn trmv(
        layout: Layout,
        uplo: UpLo,
        trans: Transpose,
        diag: Diag,
        n: usize,
        a: &[Self],
        lda: usize,
        x: &mut [Self],
        incx: usize,
    ) -> () {
        ztrmv(layout, uplo, trans, diag, n, a, lda, x, incx);
    }
    fn tpmv(
        layout: Layout,
        uplo: UpLo,
        trans: Transpose,
        diag: Diag,
        n: usize,
        ap: &[Self],
        x: &mut [Self],
        incx: usize,
    ) -> () {
        ztpmv(layout, uplo, trans, diag, n, ap, x, incx);
    }
    fn trsv(layout: Layout, uplo: UpLo, trans: Transpose, diag: Diag, n: usize, a: &[Self], lda: usize, x: &mut [Self], incx: usize) -> () {
        ztrsv(layout, uplo, trans, diag, n, a, lda, x, incx);
    }
    fn tpsv(layout: Layout, uplo: UpLo, trans: Transpose, diag: Diag, n: usize, ap: &[Self], x: &mut [Self], incx: usize) -> () {
        ztpsv(layout, uplo, trans, diag, n, ap, x, incx);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use approx::AbsDiffEq;

    /// NA for non-referenced parts of test matrices
    const NA: f64 = f64::NAN;
    #[test]
    fn test_dgemv() {
        let a = vec![1.0, 2.0, -3.0, -1.0, 2.0, 3.0];
        let x1 = vec![1.0, 1.0, 1.0];
        let mut y1 = vec![0.0, 0.0];
        let x2 = vec![1.0, 1.0];
        let mut y2 = vec![0.0, 0.0, 0.0];
        dgemv(
            Layout::RowMajor,
            Transpose::NoTrans,
            2,
            3,
            1.0,
            &a,
            3,
            &x1,
            1,
            1.0,
            &mut y1,
            1,
        );
        assert!(y1.abs_diff_eq(&[0.0, 4.0], 1e-15), "{:?}", y1);
        dgemv(
            Layout::ColMajor,
            Transpose::NoTrans,
            3,
            2,
            1.0,
            &a,
            3,
            &x2,
            1,
            1.0,
            &mut y2,
            1,
        );
        assert!(y2.abs_diff_eq(&[0.0, 4.0, 0.0], 1e-15), "{:?}", y2);
        let mut y1t = vec![0.0, 0.0, 0.0];
        dgemv(
            Layout::RowMajor,
            Transpose::Trans,
            2,
            3,
            1.0,
            &a,
            3,
            &x2,
            1,
            1.0,
            &mut y1t,
            1,
        );
        assert!(y1t.abs_diff_eq(&y2, 1e-15), "{:?} vs {:?}", y1t, y2);
        let mut y2t = vec![0.0, 0.0];
        dgemv(
            Layout::ColMajor,
            Transpose::Trans,
            3,
            2,
            1.0,
            &a,
            3,
            &x1,
            1,
            1.0,
            &mut y2t,
            1,
        );
        assert!(y2t.abs_diff_eq(&y1, 1e-15), "{:?} vs {:?}", y2t, y1);
    }
    #[test]
    fn test_ssyr() {
        let x = vec![1.0, 2.0, 3.0];
        let mut v = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        ssyr(Layout::RowMajor, UpLo::Upper, 3, 10.0, &x, 1, &mut v, 3);
        assert!(
            v.abs_diff_eq(&[10.0, 20.0, 30.0, 0.0, 40.0, 60.0, 0.0, 0.0, 90.0], 1e-15),
            "{:?}",
            v
        );
        v.fill(0.0);
        ssyr(Layout::RowMajor, UpLo::Lower, 3, 10.0, &x, 1, &mut v, 3);
        assert!(
            v.abs_diff_eq(&[10.0, 0.0, 0.0, 20.0, 40.0, 0.0, 30.0, 60.0, 90.0], 1e-15),
            "{:?}",
            v
        );
        v.fill(0.0);
        ssyr(Layout::ColMajor, UpLo::Upper, 3, 10.0, &x, 1, &mut v, 3);
        assert!(
            v.abs_diff_eq(&[10.0, 0.0, 0.0, 20.0, 40.0, 0.0, 30.0, 60.0, 90.0], 1e-15),
            "{:?}",
            v
        );
        v.fill(0.0);
        ssyr(Layout::ColMajor, UpLo::Lower, 3, 10.0, &x, 1, &mut v, 3);
        assert!(
            v.abs_diff_eq(&[10.0, 20.0, 30.0, 0.0, 40.0, 60.0, 0.0, 0.0, 90.0], 1e-15),
            "{:?}",
            v
        );
    }
    #[test]
    fn test_dsyr() {
        let x = vec![1.0, 2.0, 3.0];
        let mut v = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        dsyr(Layout::RowMajor, UpLo::Upper, 3, 10.0, &x, 1, &mut v, 3);
        assert!(
            v.abs_diff_eq(&[10.0, 20.0, 30.0, 0.0, 40.0, 60.0, 0.0, 0.0, 90.0], 1e-15),
            "{:?}",
            v
        );
        v.fill(0.0);
        dsyr(Layout::RowMajor, UpLo::Lower, 3, 10.0, &x, 1, &mut v, 3);
        assert!(
            v.abs_diff_eq(&[10.0, 0.0, 0.0, 20.0, 40.0, 0.0, 30.0, 60.0, 90.0], 1e-15),
            "{:?}",
            v
        );
        v.fill(0.0);
        dsyr(Layout::ColMajor, UpLo::Upper, 3, 10.0, &x, 1, &mut v, 3);
        assert!(
            v.abs_diff_eq(&[10.0, 0.0, 0.0, 20.0, 40.0, 0.0, 30.0, 60.0, 90.0], 1e-15),
            "{:?}",
            v
        );
        v.fill(0.0);
        dsyr(Layout::ColMajor, UpLo::Lower, 3, 10.0, &x, 1, &mut v, 3);
        assert!(
            v.abs_diff_eq(&[10.0, 20.0, 30.0, 0.0, 40.0, 60.0, 0.0, 0.0, 90.0], 1e-15),
            "{:?}",
            v
        );
    }
    #[test]
    fn test_cher() {
        let x = vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, 1.0),
            Complex32::new(3.0, 1.0),
        ];
        let mut v = vec![
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
        ];
        cher(
            Layout::RowMajor,
            UpLo::Upper,
            3,
            Complex32::new(10.0, 0.0),
            &x,
            1,
            &mut v,
            3,
        );
        let expected = vec![
            Complex32::new(1.0, 1.0) * Complex32::new(1.0, -1.0),
            Complex32::new(1.0, 1.0) * Complex32::new(2.0, -1.0),
            Complex32::new(1.0, 1.0) * Complex32::new(3.0, -1.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(2.0, 1.0) * Complex32::new(2.0, -1.0),
            Complex32::new(2.0, 1.0) * Complex32::new(3.0, -1.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(3.0, 1.0) * Complex32::new(3.0, -1.0),
        ];
        for (i, (vi, xi)) in v.iter().zip(expected).enumerate() {
            assert_eq!(vi, &(10.0 * xi), "{}, {}, {}", vi, xi, i);
        }
        v.fill(Complex32::new(0.0, 0.0));
        cher(
            Layout::RowMajor,
            UpLo::Lower,
            3,
            Complex32::new(10.0, 0.0),
            &x,
            1,
            &mut v,
            3,
        );
        let expected = vec![
            Complex32::new(1.0, 1.0) * Complex32::new(1.0, -1.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(2.0, 1.0) * Complex32::new(1.0, -1.0),
            Complex32::new(2.0, 1.0) * Complex32::new(2.0, -1.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(3.0, 1.0) * Complex32::new(1.0, -1.0),
            Complex32::new(3.0, 1.0) * Complex32::new(2.0, -1.0),
            Complex32::new(3.0, 1.0) * Complex32::new(3.0, -1.0),
        ];
        for (vi, xi) in v.iter().zip(expected) {
            assert_eq!(vi, &(10.0 * xi));
        }
        v.fill(Complex32::new(0.0, 0.0));
        cher(
            Layout::ColMajor,
            UpLo::Upper,
            3,
            Complex32::new(10.0, 0.0),
            &x,
            1,
            &mut v,
            3,
        );
        let expected = vec![
            Complex32::new(1.0, 1.0) * Complex32::new(1.0, -1.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 1.0) * Complex32::new(2.0, -1.0),
            Complex32::new(2.0, 1.0) * Complex32::new(2.0, -1.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 1.0) * Complex32::new(3.0, -1.0),
            Complex32::new(2.0, 1.0) * Complex32::new(3.0, -1.0),
            Complex32::new(3.0, 1.0) * Complex32::new(3.0, -1.0),
        ];
        for (vi, xi) in v.iter().zip(expected) {
            assert_eq!(vi, &(10.0 * xi));
        }
        v.fill(Complex32::new(0.0, 0.0));
        cher(
            Layout::ColMajor,
            UpLo::Lower,
            3,
            Complex32::new(10.0, 0.0),
            &x,
            1,
            &mut v,
            3,
        );
        let expected = vec![
            Complex32::new(1.0, 1.0) * Complex32::new(1.0, -1.0),
            Complex32::new(2.0, 1.0) * Complex32::new(1.0, -1.0),
            Complex32::new(3.0, 1.0) * Complex32::new(1.0, -1.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(2.0, 1.0) * Complex32::new(2.0, -1.0),
            Complex32::new(3.0, 1.0) * Complex32::new(2.0, -1.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(3.0, 1.0) * Complex32::new(3.0, -1.0),
        ];
        for (vi, xi) in v.iter().zip(expected) {
            assert_eq!(vi, &(10.0 * xi));
        }
    }
    #[test]
    fn test_zher() {
        let x = vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(3.0, 1.0),
        ];
        let mut v = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        zher(
            Layout::RowMajor,
            UpLo::Upper,
            3,
            Complex64::new(10.0, 0.0),
            &x,
            1,
            &mut v,
            3,
        );
        let expected = vec![
            Complex64::new(1.0, 1.0) * Complex64::new(1.0, -1.0),
            Complex64::new(1.0, 1.0) * Complex64::new(2.0, -1.0),
            Complex64::new(1.0, 1.0) * Complex64::new(3.0, -1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 1.0) * Complex64::new(2.0, -1.0),
            Complex64::new(2.0, 1.0) * Complex64::new(3.0, -1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(3.0, 1.0) * Complex64::new(3.0, -1.0),
        ];
        for (i, (vi, xi)) in v.iter().zip(expected).enumerate() {
            assert_eq!(vi, &(10.0 * xi), "{}, {}, {}", vi, xi, i);
        }
        v.fill(Complex64::new(0.0, 0.0));
        zher(
            Layout::RowMajor,
            UpLo::Lower,
            3,
            Complex64::new(10.0, 0.0),
            &x,
            1,
            &mut v,
            3,
        );
        let expected = vec![
            Complex64::new(1.0, 1.0) * Complex64::new(1.0, -1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 1.0) * Complex64::new(1.0, -1.0),
            Complex64::new(2.0, 1.0) * Complex64::new(2.0, -1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(3.0, 1.0) * Complex64::new(1.0, -1.0),
            Complex64::new(3.0, 1.0) * Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 1.0) * Complex64::new(3.0, -1.0),
        ];
        for (vi, xi) in v.iter().zip(expected) {
            assert_eq!(vi, &(10.0 * xi));
        }
        v.fill(Complex64::new(0.0, 0.0));
        zher(
            Layout::ColMajor,
            UpLo::Upper,
            3,
            Complex64::new(10.0, 0.0),
            &x,
            1,
            &mut v,
            3,
        );
        let expected = vec![
            Complex64::new(1.0, 1.0) * Complex64::new(1.0, -1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 1.0) * Complex64::new(2.0, -1.0),
            Complex64::new(2.0, 1.0) * Complex64::new(2.0, -1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 1.0) * Complex64::new(3.0, -1.0),
            Complex64::new(2.0, 1.0) * Complex64::new(3.0, -1.0),
            Complex64::new(3.0, 1.0) * Complex64::new(3.0, -1.0),
        ];
        for (vi, xi) in v.iter().zip(expected) {
            assert_eq!(vi, &(10.0 * xi));
        }
        v.fill(Complex64::new(0.0, 0.0));
        zher(
            Layout::ColMajor,
            UpLo::Lower,
            3,
            Complex64::new(10.0, 0.0),
            &x,
            1,
            &mut v,
            3,
        );
        let expected = vec![
            Complex64::new(1.0, 1.0) * Complex64::new(1.0, -1.0),
            Complex64::new(2.0, 1.0) * Complex64::new(1.0, -1.0),
            Complex64::new(3.0, 1.0) * Complex64::new(1.0, -1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 1.0) * Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 1.0) * Complex64::new(2.0, -1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(3.0, 1.0) * Complex64::new(3.0, -1.0),
        ];
        for (vi, xi) in v.iter().zip(expected) {
            assert_eq!(vi, &(10.0 * xi));
        }
    }
    #[test]
    fn test_dspmv() {
        // 3x3 packed symmetric
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![-1.0, 2.0, -3.0];
        let mut y = vec![0.0, 0.0, 0.0];
        dspmv(
            Layout::RowMajor,
            UpLo::Upper,
            3,
            1.0,
            &a,
            &x,
            1,
            1.0,
            &mut y,
            1,
        );
        assert!(y.abs_diff_eq(&vec![-6.0, -9.0, -11.0], 1e-15), "{:?}", y);
        y.fill(0.0);
        dspmv(
            Layout::RowMajor,
            UpLo::Lower,
            3,
            1.0,
            &a,
            &x,
            1,
            1.0,
            &mut y,
            1,
        );
        assert!(y.abs_diff_eq(&vec![-9.0, -11.0, -12.0], 1e-15), "{:?}", y);
        y.fill(0.0);
        dspmv(
            Layout::ColMajor,
            UpLo::Upper,
            3,
            1.0,
            &a,
            &x,
            1,
            1.0,
            &mut y,
            1,
        );
        assert!(y.abs_diff_eq(&vec![-9.0, -11.0, -12.0], 1e-15), "{:?}", y);
        y.fill(0.0);
        dspmv(
            Layout::ColMajor,
            UpLo::Lower,
            3,
            1.0,
            &a,
            &x,
            1,
            1.0,
            &mut y,
            1,
        );
        assert!(y.abs_diff_eq(&vec![-6.0, -9.0, -11.0], 1e-15), "{:?}", y);
    }
    #[test]
    fn test_zhpmv() {
        // 3x3 packed symmetric
        let x = vec![
            Complex64::new(-1.0, 1.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(-3.0, 1.0),
        ];
        let mut y = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        zhpmv(
            Layout::RowMajor,
            UpLo::Upper,
            3,
            Complex64::new(1.0, 0.0),
            &vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(3.0, 1.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 1.0),
                Complex64::new(6.0, 0.0),
            ],
            &x,
            1,
            Complex64::new(1.0, 0.0),
            &mut y,
            1,
        );
        for (i, (yi, ei)) in y
            .iter()
            .zip(vec![
                Complex64::new(1.0, 0.0) * x[0]
                    + Complex64::new(2.0, 1.0) * x[1]
                    + Complex64::new(3.0, 1.0) * x[2],
                Complex64::new(2.0, -1.0) * x[0]
                    + Complex64::new(4.0, 0.0) * x[1]
                    + Complex64::new(5.0, 1.0) * x[2],
                Complex64::new(3.0, -1.0) * x[0]
                    + Complex64::new(5.0, -1.0) * x[1]
                    + Complex64::new(6.0, 0.0) * x[2],
            ])
            .enumerate()
        {
            assert_eq!(yi, &ei, "{} - {} != {}", i, yi, &ei);
        }
        y.fill(Complex64::new(0.0, 0.0));
        zhpmv(
            Layout::RowMajor,
            UpLo::Lower,
            3,
            Complex64::new(1.0, 0.0),
            &vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 1.0),
                Complex64::new(5.0, 1.0),
                Complex64::new(6.0, 0.0),
            ],
            &x,
            1,
            Complex64::new(1.0, 0.0),
            &mut y,
            1,
        );
        for (yi, ei) in y.iter().zip(vec![
            Complex64::new(1.0, 0.0) * x[0]
                + Complex64::new(2.0, -1.0) * x[1]
                + Complex64::new(4.0, -1.0) * x[2],
            Complex64::new(2.0, 1.0) * x[0]
                + Complex64::new(3.0, 0.0) * x[1]
                + Complex64::new(5.0, -1.0) * x[2],
            Complex64::new(4.0, 1.0) * x[0]
                + Complex64::new(5.0, 1.0) * x[1]
                + Complex64::new(6.0, 0.0) * x[2],
        ]) {
            assert_eq!(yi, &ei);
        }
        y.fill(Complex64::new(0.0, 0.0));
        zhpmv(
            Layout::ColMajor,
            UpLo::Upper,
            3,
            Complex64::new(1.0, 0.0),
            &vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 1.0),
                Complex64::new(5.0, 1.0),
                Complex64::new(6.0, 0.0),
            ],
            &x,
            1,
            Complex64::new(1.0, 0.0),
            &mut y,
            1,
        );
        for (yi, ei) in y.iter().zip(vec![
            Complex64::new(1.0, 0.0) * x[0]
                + Complex64::new(2.0, 1.0) * x[1]
                + Complex64::new(4.0, 1.0) * x[2],
            Complex64::new(2.0, -1.0) * x[0]
                + Complex64::new(3.0, 0.0) * x[1]
                + Complex64::new(5.0, 1.0) * x[2],
            Complex64::new(4.0, -1.0) * x[0]
                + Complex64::new(5.0, -1.0) * x[1]
                + Complex64::new(6.0, 0.0) * x[2],
        ]) {
            assert_eq!(yi, &ei);
        }
        y.fill(Complex64::new(0.0, 0.0));
        zhpmv(
            Layout::ColMajor,
            UpLo::Lower,
            3,
            Complex64::new(1.0, 0.0),
            &vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(3.0, 1.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 1.0),
                Complex64::new(6.0, 0.0),
            ],
            &x,
            1,
            Complex64::new(1.0, 0.0),
            &mut y,
            1,
        );
        for (yi, ei) in y.iter().zip(vec![
            Complex64::new(1.0, 0.0) * x[0]
                + Complex64::new(2.0, -1.0) * x[1]
                + Complex64::new(3.0, -1.0) * x[2],
            Complex64::new(2.0, 1.0) * x[0]
                + Complex64::new(4.0, 0.0) * x[1]
                + Complex64::new(5.0, -1.0) * x[2],
            Complex64::new(3.0, 1.0) * x[0]
                + Complex64::new(5.0, 1.0) * x[1]
                + Complex64::new(6.0, 0.0) * x[2],
        ]) {
            assert_eq!(yi, &ei);
        }
    }
    #[test]
    fn test_dspr() {
        let mut a = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let x = vec![1.0, 2.0, 3.0];
        dspr(Layout::RowMajor, UpLo::Upper, 3, 1.0, &x, 1, &mut a[..]);
        assert!(
            a.abs_diff_eq(
                &vec![
                    1.0 * 1.0,
                    1.0 * 2.0,
                    1.0 * 3.0,
                    2.0 * 2.0,
                    2.0 * 3.0,
                    3.0 * 3.0,
                ],
                1e-15
            ),
            "{:?}",
            a
        );
        a.fill(0.0);
        dspr(Layout::RowMajor, UpLo::Lower, 3, 1.0, &x, 1, &mut a[..]);
        assert!(
            a.abs_diff_eq(
                &vec![
                    1.0 * 1.0,
                    2.0 * 1.0,
                    2.0 * 2.0,
                    3.0 * 1.0,
                    3.0 * 2.0,
                    3.0 * 3.0,
                ],
                1e-15
            ),
            "{:?}",
            a
        );
        a.fill(0.0);
        dspr(Layout::ColMajor, UpLo::Upper, 3, 1.0, &x, 1, &mut a[..]);
        assert!(
            a.abs_diff_eq(
                &vec![
                    1.0 * 1.0,
                    2.0 * 1.0,
                    2.0 * 2.0,
                    3.0 * 1.0,
                    3.0 * 2.0,
                    3.0 * 3.0,
                ],
                1e-15
            ),
            "{:?}",
            a
        );
        a.fill(0.0);
        dspr(Layout::ColMajor, UpLo::Lower, 3, 1.0, &x, 1, &mut a[..]);
        assert!(
            a.abs_diff_eq(
                &vec![
                    1.0 * 1.0,
                    1.0 * 2.0,
                    1.0 * 3.0,
                    2.0 * 2.0,
                    2.0 * 3.0,
                    3.0 * 3.0,
                ],
                1e-15
            ),
            "{:?}",
            a
        );
    }
    #[test]
    fn test_dtrmv() {
        let mut x = vec![10.0, 2.0, 1.0];
        dtrmv(
            Layout::RowMajor,
            UpLo::Upper,
            Transpose::NoTrans,
            Diag::NonUnit,
            3,
            &vec![5.0, 2.0, 3.0, 0.0, 2.0, 4.0, 0.0, 0.0, 3.0],
            3,
            &mut x,
            1,
        );
        assert!(x.abs_diff_eq(&vec![57.0, 8.0, 3.0], 1e-15), "{:?}", &x,);
        x = vec![10.0, 2.0, 1.0];
        dtrmv(
            Layout::RowMajor,
            UpLo::Lower,
            Transpose::NoTrans,
            Diag::NonUnit,
            3,
            &vec![5.0, 0.0, 0.0, 2.0, 3.0, 0.0, 2.0, 4.0, 3.0],
            3,
            &mut x,
            1,
        );
        assert!(x.abs_diff_eq(&vec![50.0, 26.0, 31.0], 1e-15), "{:?}", &x,);
        x = vec![10.0, 2.0, 1.0];
        dtrmv(
            Layout::ColMajor,
            UpLo::Upper,
            Transpose::NoTrans,
            Diag::NonUnit,
            3,
            &vec![5.0, 0.0, 0.0, 2.0, 3.0, 0.0, 2.0, 4.0, 3.0],
            3,
            &mut x,
            1,
        );
        assert!(x.abs_diff_eq(&vec![56.0, 10.0, 3.0], 1e-15), "{:?}", &x,);
        x = vec![10.0, 2.0, 1.0];
        dtrmv(
            Layout::ColMajor,
            UpLo::Lower,
            Transpose::NoTrans,
            Diag::NonUnit,
            3,
            &vec![5.0, 2.0, 3.0, 0.0, 3.0, 1.0, 0.0, 0.0, 4.0],
            3,
            &mut x,
            1,
        );
        assert!(x.abs_diff_eq(&vec![50.0, 26.0, 36.0], 1e-15), "{:?}", &x,);
    }
    #[test]
    fn test_dtrmv_unit() {
        let mut x: Vec<_>;
        let mut xexp: Vec<_>;
        x = vec![10.0, 2.0, 1.0];
        dtrmv(
            Layout::RowMajor,
            UpLo::Upper,
            Transpose::NoTrans,
            Diag::Unit,
            3,
            &vec![5.0, 2.0, 3.0, 0.0, 2.0, 4.0, 0.0, 0.0, 3.0],
            3,
            &mut x,
            1,
        );
        xexp = vec![10.0, 2.0, 1.0];
        dtrmv(
            Layout::RowMajor,
            UpLo::Upper,
            Transpose::NoTrans,
            Diag::NonUnit,
            3,
            &vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 0.0, 0.0, 1.0],
            3,
            &mut xexp,
            1,
        );
        assert!(x.abs_diff_eq(&xexp, 1e-15), "{:?} vs {:?}", &x, &xexp);
        x = vec![10.0, 2.0, 1.0];
        dtrmv(
            Layout::RowMajor,
            UpLo::Lower,
            Transpose::NoTrans,
            Diag::Unit,
            3,
            &vec![5.0, 0.0, 0.0, 2.0, 3.0, 0.0, 2.0, 4.0, 3.0],
            3,
            &mut x,
            1,
        );
        xexp = vec![10.0, 2.0, 1.0];
        dtrmv(
            Layout::RowMajor,
            UpLo::Lower,
            Transpose::NoTrans,
            Diag::NonUnit,
            3,
            &vec![1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 4.0, 1.0],
            3,
            &mut xexp,
            1,
        );
        assert!(x.abs_diff_eq(&xexp, 1e-15), "{:?} vs {:?}", &x, &xexp);
        x = vec![10.0, 2.0, 1.0];
        dtrmv(
            Layout::ColMajor,
            UpLo::Upper,
            Transpose::NoTrans,
            Diag::Unit,
            3,
            &vec![5.0, 0.0, 0.0, 2.0, 3.0, 0.0, 2.0, 4.0, 3.0],
            3,
            &mut x,
            1,
        );
        xexp = vec![10.0, 2.0, 1.0];
        dtrmv(
            Layout::ColMajor,
            UpLo::Upper,
            Transpose::NoTrans,
            Diag::NonUnit,
            3,
            &vec![1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 4.0, 1.0],
            3,
            &mut xexp,
            1,
        );
        assert!(x.abs_diff_eq(&xexp, 1e-15), "{:?} vs {:?}", &x, &xexp);
        x = vec![10.0, 2.0, 1.0];
        dtrmv(
            Layout::ColMajor,
            UpLo::Lower,
            Transpose::NoTrans,
            Diag::Unit,
            3,
            &vec![5.0, 2.0, 3.0, 0.0, 3.0, 1.0, 0.0, 0.0, 4.0],
            3,
            &mut x,
            1,
        );
        xexp = vec![10.0, 2.0, 1.0];
        dtrmv(
            Layout::ColMajor,
            UpLo::Lower,
            Transpose::NoTrans,
            Diag::NonUnit,
            3,
            &vec![1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            3,
            &mut xexp,
            1,
        );
        assert!(x.abs_diff_eq(&xexp, 1e-15), "{:?} vs {:?}", &x, &xexp);
    }
    #[test]
    fn test_dtpmv() {
        let mut x = vec![1.0, 10.0, 100.0];
        dtpmv(
            Layout::RowMajor,
            UpLo::Upper,
            Transpose::NoTrans,
            Diag::NonUnit,
            3,
            &vec![8.0, 2.0, 3.0, 7.0, 4.0, -1.0],
            &mut x,
            1,
        );
        assert!(x.abs_diff_eq(&vec![328.0, 470., -100.], 1e-21), "{:?}", x);
        x = vec![1.0, 10.0, 100.0];
        dtpmv(
            Layout::RowMajor,
            UpLo::Upper,
            Transpose::Trans,
            Diag::NonUnit,
            3,
            &vec![8.0, 2.0, 3.0, 7.0, 4.0, 9.0],
            &mut x,
            1,
        );
        assert!(x.abs_diff_eq(&vec![8.0, 72., 943.], 1e-21), "{:?}", x);
        x = vec![1.0, 10.0, 100.0];
        dtpmv(
            Layout::RowMajor,
            UpLo::Lower,
            Transpose::NoTrans,
            Diag::NonUnit,
            3,
            &vec![8.0, 2.0, 7.0, 3.0, 4.0, 9.0],
            &mut x,
            1,
        );
        assert!(x.abs_diff_eq(&vec![8.0, 72., 943.], 1e-21), "{:?}", x);
        x = vec![1.0, 10.0, 100.0];
        dtpmv(
            Layout::RowMajor,
            UpLo::Lower,
            Transpose::Trans,
            Diag::NonUnit,
            3,
            &vec![8.0, 2.0, 7.0, 3.0, 4.0, -1.0],
            &mut x,
            1,
        );
        assert!(x.abs_diff_eq(&vec![328.0, 470., -100.], 1e-21), "{:?}", x);
        x = vec![1.0, 10.0, 100.0];
        dtpmv(
            Layout::ColMajor,
            UpLo::Upper,
            Transpose::NoTrans,
            Diag::NonUnit,
            3,
            &vec![8.0, 2.0, 7.0, 3.0, 4.0, -1.0],
            &mut x,
            1,
        );
        assert!(x.abs_diff_eq(&vec![328.0, 470., -100.], 1e-21), "{:?}", x);
        x = vec![1.0, 10.0, 100.0];
        dtpmv(
            Layout::ColMajor,
            UpLo::Upper,
            Transpose::Trans,
            Diag::NonUnit,
            3,
            &vec![8.0, 2.0, 7.0, 3.0, 4.0, 9.0],
            &mut x,
            1,
        );
        assert!(x.abs_diff_eq(&vec![8.0, 72., 943.], 1e-21), "{:?}", x);
        x = vec![1.0, 10.0, 100.0];
        dtpmv(
            Layout::ColMajor,
            UpLo::Lower,
            Transpose::NoTrans,
            Diag::NonUnit,
            3,
            &vec![8.0, 2.0, 3.0, 7.0, 4.0, 9.0],
            &mut x,
            1,
        );
        assert!(x.abs_diff_eq(&vec![8.0, 72., 943.], 1e-21), "{:?}", x);
        x = vec![1.0, 10.0, 100.0];
        dtpmv(
            Layout::ColMajor,
            UpLo::Lower,
            Transpose::Trans,
            Diag::NonUnit,
            3,
            &vec![8.0, 2.0, 3.0, 7.0, 4.0, -1.0],
            &mut x,
            1,
        );
        assert!(x.abs_diff_eq(&vec![328.0, 470., -100.], 1e-21), "{:?}", x);
    }
    #[test]
    fn test_dtpmv_unit() {
        let mut x = vec![1.0, 10.0, 100.0];
        dtpmv(
            Layout::RowMajor,
            UpLo::Upper,
            Transpose::NoTrans,
            Diag::Unit,
            3,
            &vec![NA, 2.0, 3.0, NA, 4.0, NA],
            &mut x,
            1,
        );
        assert!(x.abs_diff_eq(&vec![321.0, 410., 100.], 1e-21), "{:?}", x);
        x = vec![1.0, 10.0, 100.0];
        dtpmv(
            Layout::RowMajor,
            UpLo::Upper,
            Transpose::Trans,
            Diag::Unit,
            3,
            &vec![NA, 2.0, 3.0, NA, 4.0, NA],
            &mut x,
            1,
        );
        assert!(x.abs_diff_eq(&vec![1.0, 12., 143.], 1e-21), "{:?}", x);
        x = vec![1.0, 10.0, 100.0];
        dtpmv(
            Layout::RowMajor,
            UpLo::Lower,
            Transpose::NoTrans,
            Diag::Unit,
            3,
            &vec![NA, 2.0, NA, 3.0, 4.0, NA],
            &mut x,
            1,
        );
        assert!(x.abs_diff_eq(&vec![1.0, 12., 143.], 1e-21), "{:?}", x);
        x = vec![1.0, 10.0, 100.0];
        dtpmv(
            Layout::RowMajor,
            UpLo::Lower,
            Transpose::Trans,
            Diag::Unit,
            3,
            &vec![NA, 2.0, NA, 3.0, 4.0, NA],
            &mut x,
            1,
        );
        assert!(x.abs_diff_eq(&vec![321.0, 410., 100.], 1e-21), "{:?}", x);
        x = vec![1.0, 10.0, 100.0];
        dtpmv(
            Layout::ColMajor,
            UpLo::Upper,
            Transpose::NoTrans,
            Diag::Unit,
            3,
            &vec![NA, 2.0, NA, 3.0, 4.0, NA],
            &mut x,
            1,
        );
        assert!(x.abs_diff_eq(&vec![321.0, 410., 100.], 1e-21), "{:?}", x);
        x = vec![1.0, 10.0, 100.0];
        dtpmv(
            Layout::ColMajor,
            UpLo::Upper,
            Transpose::Trans,
            Diag::Unit,
            3,
            &vec![NA, 2.0, NA, 3.0, 4.0, NA],
            &mut x,
            1,
        );
        assert!(x.abs_diff_eq(&vec![1.0, 12., 143.], 1e-21), "{:?}", x);
        x = vec![1.0, 10.0, 100.0];
        dtpmv(
            Layout::ColMajor,
            UpLo::Lower,
            Transpose::NoTrans,
            Diag::Unit,
            3,
            &vec![NA, 2.0, 3.0, NA, 4.0, NA],
            &mut x,
            1,
        );
        assert!(x.abs_diff_eq(&vec![1.0, 12., 143.], 1e-21), "{:?}", x);
        x = vec![1.0, 10.0, 100.0];
        dtpmv(
            Layout::ColMajor,
            UpLo::Lower,
            Transpose::Trans,
            Diag::Unit,
            3,
            &vec![NA, 2.0, 3.0, NA, 4.0, NA],
            &mut x,
            1,
        );
        assert!(x.abs_diff_eq(&vec![321.0, 410., 100.], 1e-21), "{:?}", x);
    }
    #[test]
    fn test_gbmv_row_notrans() {
        let a = vec![
            NA, NA, 1.0, 2.0, NA, 2.0, 1.0, 3.0, 3.0, 1.0, 2.0, 5.0, 1.0, 2.0, 3.0, NA, 3.0, 1.0,
            NA, NA,
        ];
        let afull = vec![
            1.0, 2.0, 0.0, 0.0, 2.0, 1.0, 3.0, 0.0, 3.0, 1.0, 2.0, 5.0, 0.0, 1.0, 2.0, 3.0, 0.0,
            0.0, 3.0, 1.0,
        ];
        let x = vec![1.0, 10.0, 100.0, 1000.0];
        let mut yb = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let mut yf = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        dgemv(
            Layout::RowMajor,
            Transpose::NoTrans,
            5,
            4,
            1.0,
            &afull,
            4,
            &x,
            1,
            1.0,
            &mut yf,
            1,
        );
        dgbmv(
            Layout::RowMajor,
            Transpose::NoTrans,
            5,
            4,
            2,
            1,
            1.0,
            &a,
            4,
            &x,
            1,
            1.0,
            &mut yb,
            1,
        );
        assert!(yb.abs_diff_eq(&yf, 1e-21), "{:?} != {:?}", &yb, &yf)
    }
    #[test]
    fn test_gbmv_col_notrans() {
        let a = vec![
            NA, NA, 1.0, 2.0, NA, 2.0, 1.0, 3.0, 3.0, 1.0, 2.0, 5.0, 1.0, 2.0, 3.0, NA, 3.0, 1.0,
            NA, NA,
        ];
        let afull = vec![
            1.0, 2.0, 0.0, 0.0, 2.0, 1.0, 3.0, 0.0, 3.0, 1.0, 2.0, 5.0, 0.0, 1.0, 2.0, 3.0, 0.0,
            0.0, 3.0, 1.0,
        ];
        let x = vec![1.0, 10.0, 100.0, 1000.0, -1.0];
        let mut yb = vec![0.0, 0.0, 0.0, 0.0];
        let mut yf = vec![0.0, 0.0, 0.0, 0.0];
        dgemv(
            Layout::ColMajor,
            Transpose::NoTrans,
            4,
            5,
            1.0,
            &afull,
            4,
            &x,
            1,
            1.0,
            &mut yf,
            1,
        );
        dgbmv(
            Layout::ColMajor,
            Transpose::NoTrans,
            4,
            5,
            1,
            2,
            1.0,
            &a,
            4,
            &x,
            1,
            1.0,
            &mut yb,
            1,
        );
        assert!(yb.abs_diff_eq(&yf, 1e-21), "{:?} != {:?}", &yb, &yf)
    }
    #[test]
    pub fn test_real_trsv() {
        let a = vec![-1.0, 1.0, 1.0, 12.5, 9.0, 2.0, -7.3, 4.9, -50.0];
        let mut x = vec![1.0, 2.0, 3.0];
        let mut xcopy = x.clone();
        for layout in vec![Layout::RowMajor, Layout::ColMajor] {
            for uplo in vec![UpLo::Upper, UpLo::Lower] {
                for trans in vec![Transpose::Trans, Transpose::NoTrans, Transpose::ConjTrans] {
                    for unit in vec![Diag::Unit, Diag::NonUnit] {
                        dtrsv(layout, uplo, trans, unit, 3, &a, 3, &mut x, 1);
                        dtrmv(layout, uplo, trans, unit, 3, &a, 3, &mut x, 1);
                        assert!(
                            x.abs_diff_eq(&xcopy, 1e-15),
                            "({:?}, {:?}, {:?}, {:?}) {:?} != {:?}",
                            layout,
                            uplo,
                            trans,
                            unit,
                            &x,
                            &xcopy
                        )
                    }
                }
            }
        }
    }
    #[test]
    pub fn test_complex_trsv() {
        let a = vec![
            Complex64::new(-1.0, 1.0), Complex64::new(1.0, -1.0), Complex64::new(1.0, 5.0),
            Complex64::new(12.5, -1.0), Complex64::new(5.0, 2.0), Complex64::new(-3.0, 1.0),
            Complex64::new(-7.3, 5.0), Complex64::new(4.9, -1.0), Complex64::new(-50.0, 5.0),
        ];
        let mut x = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 1.0), Complex64::new(3.0, -1.0)];
        let mut xcopy = x.clone();
        for layout in vec![Layout::RowMajor, Layout::ColMajor] {
            for uplo in vec![UpLo::Upper, UpLo::Lower] {
                for trans in vec![Transpose::Trans, Transpose::NoTrans, Transpose::ConjTrans] {
                    for unit in vec![Diag::Unit, Diag::NonUnit] {
                        ztrsv(layout, uplo, trans, unit, 3, &a, 3, &mut x, 1);
                        ztrmv(layout, uplo, trans, unit, 3, &a, 3, &mut x, 1);
                        for i in 0..3 {
                            assert!(
                                (x[i] - xcopy[i]).norm() < 1e-12,
                                "({:?}, {:?}, {:?}, {:?}) {:?} != {:?}",
                                layout,
                                uplo,
                                trans,
                                unit,
                                &x,
                                &xcopy
                            );

                        }
                    }
                }
            }
        }
    }
    #[test]
    pub fn test_real_tpsv() {
        let a = vec![-1.0, 1.0, 12.5, 9.0, -7.3, 5.0];
        let mut x = vec![1.0, 2.0, 3.0];
        let mut xcopy = x.clone();
        for unit in vec![Diag::Unit, Diag::NonUnit] {
            for layout in vec![Layout::RowMajor, Layout::ColMajor] {
                for trans in vec![Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
                    for uplo in vec![UpLo::Upper, UpLo::Lower] {
                        dtpsv(layout, uplo, trans, unit, 3, &a, &mut x, 1);
                        dtpmv(layout, uplo, trans, unit, 3, &a, &mut x, 1);
                        assert!(
                            x.abs_diff_eq(&xcopy, 1e-14),
                            "({:?}, {:?}, {:?}, {:?}) {:?} != {:?}",
                            layout,
                            uplo,
                            trans,
                            unit,
                            &x,
                            &xcopy
                        )
                    }
                }
            }
        }
    }
    #[test]
    fn test_lda() {
        let a = vec![
             1.0,  2.0,  3.0,  4.0,
            -1.0, -5.0,  1.0, -3.0,
             2.0,  1.0, -1.0,  7.0,
            -4.0,  4.0,  6.0, -9.0,
        ];
        let b = vec![
             1.0,  2.0,  3.0,
            -1.0, -5.0,  1.0,
             2.0,  1.0, -1.0,
        ];
        let x = vec![1.0, 0.0, 0.0, 10.0, 0.0, 0.0, 100.0];
        for layout in vec![Layout::RowMajor, Layout::ColMajor] {
            for trans in vec![Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
                let mut ya = vec![0.0, 0.0, 0.0];
                let mut yb = vec![0.0, 0.0, 0.0];
                dgemv(
                    layout,
                    trans,
                    3,
                    3,
                    1.0,
                    &a,
                    4,
                    &x,
                    3,
                    1.0,
                    &mut ya,
                    1,
                );
                dgemv(
                    layout,
                    trans,
                    3,
                    3,
                    1.0,
                    &b,
                    3,
                    &x,
                    3,
                    1.0,
                    &mut yb,
                    1,
                );
                assert!(
                    ya.abs_diff_eq(&yb, 1e-12),
                    "{:?} != {:?}", &ya, &yb,
                );
            }
        }
        for layout in vec![Layout::RowMajor, Layout::ColMajor] {
            for uplo in vec![UpLo::Upper, UpLo::Lower] {
                let mut ya = vec![1.0, 10.0, 100.0];
                let mut yb = vec![1.0, 10.0, 100.0];
                dsymv(
                    layout,
                    uplo,
                    3,
                    1.0,
                    &a,
                    4,
                    &x,
                    3,
                    1.0,
                    &mut ya,
                    1,
                );
                dsymv(
                    layout,
                    uplo,
                    3,
                    1.0,
                    &b,
                    3,
                    &x,
                    3,
                    1.0,
                    &mut yb,
                    1,
                );
                assert!(
                    ya.abs_diff_eq(&yb, 1e-12),
                    "{:?} != {:?}", &ya, &yb,
                );
            }
        }
    }
    #[test]
    fn test_dsyr2() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![10.0, 100.0, 1_000.0];
        let mut v = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        dsyr2(Layout::RowMajor, UpLo::Upper, 3, 1.0, &x, 1, &y, 1, &mut v, 3);
        assert!(
            v.abs_diff_eq(&[
                20.0, 120.0, 1_030.0,
                0.0,  400.0, 2_300.0,
                0.0,    0.0, 6_000.0
            ], 1e-15),
            "{:?}",
            v
        );
        v.fill(0.0);
        dsyr2(Layout::RowMajor, UpLo::Lower, 3, 1.0, &x, 1, &y, 1, &mut v, 3);
        assert!(
            v.abs_diff_eq(&[
                20.0,        0.0,     0.0,
                120.0,     400.0,     0.0,
                1_030.0, 2_300.0, 6_000.0
            ], 1e-15),
            "{:?}",
            v
        );
        v.fill(0.0);
        dsyr2(Layout::ColMajor, UpLo::Lower, 3, 1.0, &x, 1, &y, 1, &mut v, 3);
        assert!(
            v.abs_diff_eq(&[
                20.0, 120.0, 1_030.0,
                0.0,  400.0, 2_300.0,
                0.0,    0.0, 6_000.0
            ], 1e-15),
            "{:?}",
            v
        );
        v.fill(0.0);
        dsyr2(Layout::ColMajor, UpLo::Upper, 3, 1.0, &x, 1, &y, 1, &mut v, 3);
        assert!(
            v.abs_diff_eq(&[
                20.0,        0.0,     0.0,
                120.0,     400.0,     0.0,
                1_030.0, 2_300.0, 6_000.0
            ], 1e-15),
            "{:?}",
            v
        );
        v.fill(0.0);
    }
}
