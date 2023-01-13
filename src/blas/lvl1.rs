//! Level 1 blas: vector & vector-vector operations
use num::complex::{Complex32, Complex64};
use num::{Complex, Float, One, Zero};
use std::iter::{once};
use std::ops::{Add, AddAssign, Deref, DerefMut, Mul, MulAssign, Sub};
use crate::sum::{Kahan, Summation};

pub fn map_sum<I, T, F, S>(mut out: S, itr: I, f: F) -> T
where
    I: Iterator,
    F: Fn(I::Item) -> T,
    S: Summation<T>,
{
    for elt in itr {
        out += f(elt)
    }
    out.into()
}
pub fn asum<S, T, F, U>(n: usize, arr: &[T], inc: usize, f: F) -> U
where
    F: Fn(&T) -> U,
    U: Zero + Copy,
    S: Summation<U>,
{
    if n.is_zero() {
        U::zero()
    } else if inc.is_zero() {
        let u = f(&arr[0]);
        map_sum(S::zero(), once(u).cycle().take(n), |x| x)
    } else {
        map_sum(S::zero(), arr.into_iter().step_by(inc).take(n), f)
    }
}
pub fn sasum(n: usize, arr: &[f32], inc: usize) -> f32 {
    asum::<f32, _, _, _>(n, arr, inc, |s| s.abs())
}
pub fn dasum(n: usize, arr: &[f64], inc: usize) -> f64 {
    asum::<f64, _, _, _>(n, arr, inc, |d| d.abs())
}
pub fn casum(n: usize, arr: &[Complex32], inc: usize) -> f32 {
    asum::<f32, _, _, _>(n, arr, inc, |c| c.norm())
}
pub fn zasum(n: usize, arr: &[Complex64], inc: usize) -> f64 {
    asum::<f64, _, _, _>(n, arr, inc, |z| z.norm())
}

pub fn axpy<T, I1, I2>(a: T, x: I1, y: I2)
where
    I1: Iterator,
    I2: Iterator,
    T: Copy + Mul<I1::Item>,
    I2::Item: DerefMut,
    <I2::Item as Deref>::Target: AddAssign<<T as Mul<I1::Item>>::Output>,
{
    for (xi, mut yi) in x.zip(y) {
        *yi += a * xi;
    }
}
pub fn saxpy(n: usize, a: f32, x: &[f32], incx: usize, y: &mut [f32], incy: usize) {
    if n.is_zero() {
        return;
    }
    if incx.is_zero() && incy.is_zero() {
        for _ in 0..n {
            y[0] += a * x[0];
        }
    } else if incx.is_zero() {
        axpy(
            a,
            once(x[0]).cycle().take(n),
            y.into_iter().step_by(incy).take(n),
        )
    } else if incy.is_zero() {
        for xi in x.into_iter().step_by(incx).take(n) {
            y[0] += a * xi;
        }
    } else {
        axpy(
            a,
            x.into_iter().step_by(incx).take(n),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn daxpy(n: usize, a: f64, x: &[f64], incx: usize, y: &mut [f64], incy: usize) {
    if n.is_zero() {
        return;
    }
    if incx.is_zero() && incy.is_zero() {
        y[0] += (n as f64) * a * x[0];
    } else if incx.is_zero() {
        axpy(
            a,
            once(x[0]).cycle().take(n),
            y.into_iter().step_by(incy).take(n),
        )
    } else if incy.is_zero() {
        for xi in x.into_iter().step_by(incx).take(n) {
            y[0] += a * xi;
        }
    } else {
        axpy(
            a,
            x.into_iter().step_by(incx).take(n),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn caxpy(
    n: usize,
    a: Complex32,
    x: &[Complex32],
    incx: usize,
    y: &mut [Complex32],
    incy: usize,
) {
    if n.is_zero() {
        return;
    }
    if incx.is_zero() && incy.is_zero() {
        y[0] += a * (n as f32) * x[0];
    } else if incx.is_zero() {
        axpy(
            a,
            once(x[0]).cycle().take(n),
            y.into_iter().step_by(incy).take(n),
        )
    } else if incy.is_zero() {
        for xi in x.into_iter().step_by(incx).take(n) {
            y[0] += a * xi;
        }
    } else {
        axpy(
            a,
            x.into_iter().step_by(incx).take(n),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn zaxpy(
    n: usize,
    a: Complex64,
    x: &[Complex64],
    incx: usize,
    y: &mut [Complex64],
    incy: usize,
) {
    if n.is_zero() {
        return;
    }
    if incx.is_zero() && incy.is_zero() {
        y[0] += a * (n as f64) * x[0];
    } else if incx.is_zero() {
        axpy(
            a,
            once(x[0]).cycle().take(n),
            y.into_iter().step_by(incy).take(n),
        )
    } else if incy.is_zero() {
        for xi in x.into_iter().step_by(incx).take(n) {
            y[0] += a * xi;
        }
    } else {
        axpy(
            a,
            x.into_iter().step_by(incx).take(n),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn caxpyc(
    n: usize,
    a: Complex32,
    x: &[Complex32],
    incx: usize,
    y: &mut [Complex32],
    incy: usize,
) {
    if n.is_zero() {
        return;
    }
    if incx.is_zero() && incy.is_zero() {
        y[0] += a * (n as f32) * x[0].conj();
    } else if incx.is_zero() {
        axpy(
            a,
            once(x[0].conj()).cycle().take(n),
            y.into_iter().step_by(incy).take(n),
        )
    } else if incy.is_zero() {
        for xi in x.into_iter().step_by(incx).take(n) {
            y[0] += a * xi.conj();
        }
    } else {
        axpy(
            a,
            x.into_iter().step_by(incx).take(n).map(|x| x.conj()),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn zaxpyc(
    n: usize,
    a: Complex64,
    x: &[Complex64],
    incx: usize,
    y: &mut [Complex64],
    incy: usize,
) {
    if n.is_zero() {
        return;
    }
    if incx.is_zero() && incy.is_zero() {
        y[0] += a * (n as f64) * x[0].conj();
    } else if incx.is_zero() {
        axpy(
            a,
            once(x[0].conj()).cycle().take(n),
            y.into_iter().step_by(incy).take(n),
        )
    } else if incy.is_zero() {
        for xi in x.into_iter().step_by(incx).take(n) {
            y[0] += a * xi.conj();
        }
    } else {
        axpy(
            a,
            x.into_iter().step_by(incx).take(n).map(|x| x.conj()),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn icopy<T, I1, I2>(src: I1, dst: I2)
where
    I1: Iterator,
    I2: Iterator,
    I1::Item: Deref<Target = T>,
    I2::Item: Deref<Target = T> + DerefMut,
    T: Copy,
{
    for (xi, mut yi) in src.zip(dst) {
        *yi = *xi;
    }
}
pub fn copy<T: Copy>(n: usize, x: &[T], incx: usize, y: &mut [T], incy: usize) {
    if n.is_zero() {
        return;
    }
    if incx.is_zero() && incy.is_zero() {
        y[0] = x[0];
    } else if incy.is_zero() {
        y[0] = x[(n - 1) * incx];
    } else if incx.is_zero() {
        y.into_iter().step_by(incy).take(n).for_each(|yi| {
            *yi = x[0];
        });
    } else {
        icopy(
            x.into_iter().step_by(incx).take(n),
            y.into_iter().step_by(incy).take(n),
        );
    }
}
pub fn scopy(n: usize, x: &[f32], incx: usize, y: &mut [f32], incy: usize) {
    if n.is_zero() {
        return;
    }
    if incy.is_zero() {
        y[0] = x[n * incx];
    } else if incx.is_zero() {
        icopy(
            once(&x[0]).cycle().take(n),
            y.into_iter().step_by(incy).take(n),
        )
    } else {
        icopy(
            x.into_iter().step_by(incx).take(n),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn dcopy(n: usize, x: &[f64], incx: usize, y: &mut [f64], incy: usize) {
    if n.is_zero() {
        return;
    }
    if incy.is_zero() {
        y[0] = x[n * incx];
    } else if incx.is_zero() {
        icopy(
            once(&x[0]).cycle().take(n),
            y.into_iter().step_by(incy).take(n),
        )
    } else {
        icopy(
            x.into_iter().step_by(incx).take(n),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn ccopy(n: usize, x: &[Complex32], incx: usize, y: &mut [Complex32], incy: usize) {
    if n.is_zero() {
        return;
    }
    if incy.is_zero() {
        y[0] = x[n * incx];
    } else if incx.is_zero() {
        icopy(
            once(&x[0]).cycle().take(n),
            y.into_iter().step_by(incy).take(n),
        )
    } else {
        icopy(
            x.into_iter().step_by(incx).take(n),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn zcopy(n: usize, x: &[Complex64], incx: usize, y: &mut [Complex64], incy: usize) {
    if n.is_zero() {
        return;
    }
    if incy.is_zero() {
        y[0] = x[n * incx];
    } else if incx.is_zero() {
        icopy(
            once(&x[0]).cycle().take(n),
            y.into_iter().step_by(incy).take(n),
        )
    } else {
        icopy(
            x.into_iter().step_by(incx).take(n),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn idot<I1, I2, T, S>(mut out: S, x: I1, y: I2) -> T
where
    S: AddAssign<<I1::Item as Mul<I2::Item>>::Output>,
    S: Into<T>,
    I1: Iterator,
    I2: Iterator,
    I1::Item: Mul<I2::Item>,
{
    for (xi, yi) in x.zip(y) {
        out += xi * yi;
    }
    out.into()
}
pub fn dot<'a, T, U>(n: usize, x: &'a [T], incx: usize, y: &'a [T], incy: usize) -> U
where
    U: Zero + Add<T, Output = U> + AddAssign<T>,
    &'a T: Mul<&'a T, Output = T>,
{
    if n.is_zero() {
        U::zero()
    } else if incx.is_zero() && incy.is_zero() {
        idot(
            U::zero(),
            once(&x[0]).cycle().take(n),
            once(&y[0]).cycle().take(n),
        )
    } else if incx.is_zero() {
        idot(
            U::zero(),
            once(&x[0]).cycle().take(n),
            y.into_iter().step_by(incy).take(n),
        )
    } else if incy.is_zero() {
        idot(
            U::zero(),
            x.into_iter().step_by(incx).take(n),
            once(&y[0]).cycle().take(n),
        )
    } else {
        idot(
            U::zero(),
            x.into_iter().step_by(incx).take(n),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn sdot(n: usize, x: &[f32], incx: usize, y: &[f32], incy: usize) -> f32 {
    if n.is_zero() {
        0.0f32
    } else if incx.is_zero() && incy.is_zero() {
        (n as f32) * x[0] * y[0]
    } else if incx.is_zero() {
        x[0] * y.into_iter().step_by(incy).take(n).sum::<f32>()
    } else if incy.is_zero() {
        y[0] * x.into_iter().step_by(incx).take(n).sum::<f32>()
    } else {
        idot(
            0.0f32,
            x.into_iter().step_by(incx).take(n),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn ddot(n: usize, x: &[f64], incx: usize, y: &[f64], incy: usize) -> f64 {
    if n.is_zero() {
        0.0f64
    } else if incx.is_zero() && incy.is_zero() {
        (n as f64) * x[0] * y[0]
    } else if incx.is_zero() {
        x[0] * y.into_iter().step_by(incy).take(n).sum::<f64>()
    } else if incy.is_zero() {
        y[0] * x.into_iter().step_by(incx).take(n).sum::<f64>()
    } else {
        idot(
            0.0f64,
            x.into_iter().step_by(incx).take(n),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn dsdot(n: usize, x: &[f32], incx: usize, y: &[f32], incy: usize) -> f64 {
    if n.is_zero() {
        0.0f64
    } else if incx.is_zero() && incy.is_zero() {
        (n as f64) * x[0] as f64 * y[0] as f64
    } else if incx.is_zero() {
        x[0] as f64
            * y.into_iter()
                .step_by(incy)
                .take(n)
                .map(|&f| f as f64)
                .sum::<f64>()
    } else if incy.is_zero() {
        y[0] as f64
            * x.into_iter()
                .step_by(incx)
                .take(n)
                .map(|&f| f as f64)
                .sum::<f64>()
    } else {
        idot(
            0.0f64,
            x.into_iter().step_by(incx).take(n).map(|&f| f as f64),
            y.into_iter().step_by(incy).take(n).map(|&f| f as f64),
        )
    }
}
pub fn sdsdot(n: usize, sb: f32, x: &[f32], incx: usize, y: &[f32], incy: usize) -> f32 {
    if n.is_zero() {
        sb
    } else {
        sb + dsdot(n, x, incx, y, incy) as f32
    }
}
pub fn cdotc(n: usize, x: &[Complex32], incx: usize, y: &[Complex32], incy: usize) -> Complex32 {
    if n.is_zero() {
        Complex::zero()
    } else if incx.is_zero() && incy.is_zero() {
        x[0].conj() * y[0] * (n as f32)
    } else if incx.is_zero() {
        x[0].conj() * y.into_iter().step_by(incy).take(n).sum::<Complex32>()
    } else if incy.is_zero() {
        y[0] * x
            .into_iter()
            .step_by(incx)
            .take(n)
            .sum::<Complex32>()
            .conj()
    } else {
        idot(
            Complex::zero(),
            x.into_iter().step_by(incx).take(n).map(|c| c.conj()),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn zdotc(n: usize, x: &[Complex64], incx: usize, y: &[Complex64], incy: usize) -> Complex64 {
    if n.is_zero() {
        Complex::zero()
    } else if incx.is_zero() && incy.is_zero() {
        x[0].conj() * y[0] * (n as f64)
    } else if incx.is_zero() {
        x[0].conj() * y.into_iter().step_by(incy).take(n).sum::<Complex64>()
    } else if incy.is_zero() {
        y[0] * x
            .into_iter()
            .step_by(incx)
            .take(n)
            .sum::<Complex64>()
            .conj()
    } else {
        idot(
            Complex::zero(),
            x.into_iter().step_by(incx).take(n).map(|c| c.conj()),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn cdotu(n: usize, x: &[Complex32], incx: usize, y: &[Complex32], incy: usize) -> Complex32 {
    if n.is_zero() {
        Complex::zero()
    } else if incx.is_zero() && incy.is_zero() {
        x[0] * y[0] * (n as f32)
    } else if incx.is_zero() {
        x[0] * y.into_iter().step_by(incy).take(n).sum::<Complex32>()
    } else if incy.is_zero() {
        y[0] * x.into_iter().step_by(incx).take(n).sum::<Complex32>()
    } else {
        idot(
            Complex::zero(),
            x.into_iter().step_by(incx).take(n),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn zdotu(n: usize, x: &[Complex64], incx: usize, y: &[Complex64], incy: usize) -> Complex64 {
    if n.is_zero() {
        Complex::zero()
    } else if incx.is_zero() && incy.is_zero() {
        x[0] * y[0] * (n as f64)
    } else if incx.is_zero() {
        x[0] * y.into_iter().step_by(incy).take(n).sum::<Complex64>()
    } else if incy.is_zero() {
        y[0] * x.into_iter().step_by(incx).take(n).sum::<Complex64>()
    } else {
        idot(
            Complex::zero(),
            x.into_iter().step_by(incx).take(n),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn inrm2<I, T, F, G, U, V, S>(mut out: S, x: I, sqr: F, sqrt: G) -> V
where
    S: AddAssign<U> + Into<T>,
    I: Iterator,
    F: Fn(I::Item) -> U,
    G: Fn(T) -> V,
{
    for xi in x {
        out += sqr(xi)
    }
    sqrt(out.into())
}
pub fn snrm2(n: usize, x: &[f32], incx: usize) -> f32 {
    if n.is_zero() {
        0.0f32
    } else if incx.is_zero() {
        x[0].abs()
    } else {
        inrm2(
            0.0f32,
            x.into_iter().step_by(incx).take(n),
            |x| x * x,
            |f: f32| f.sqrt(),
        )
    }
}
pub fn dnrm2(n: usize, x: &[f64], incx: usize) -> f64 {
    if n.is_zero() {
        0.0f64
    } else if incx.is_zero() {
        x[0].abs()
    } else {
        inrm2(
            0.0f64,
            x.into_iter().step_by(incx).take(n),
            |x| x * x,
            |f: f64| f.sqrt(),
        )
    }
}
pub fn scnrm2(n: usize, x: &[Complex32], incx: usize) -> f32 {
    if n.is_zero() {
        0.0f32
    } else if incx.is_zero() {
        x[0].norm()
    } else {
        inrm2(
            0.0f32,
            x.into_iter().step_by(incx).take(n),
            |x| x.norm_sqr(),
            |f: f32| f.sqrt(),
        )
    }
}
pub fn dznrm2(n: usize, x: &[Complex64], incx: usize) -> f64 {
    if n.is_zero() {
        0.0f64
    } else if incx.is_zero() {
        x[0].norm()
    } else {
        inrm2(
            0.0f64,
            x.into_iter().step_by(incx).take(n),
            |x| x.norm_sqr(),
            |f: f64| f.sqrt(),
        )
    }
}
fn real_rot<T: Float, U>(s: T, c: T, x: &mut U, y: &mut U)
where
    U: Copy + Mul<T, Output = U> + Add<U, Output = U> + Sub<U, Output = U>,
{
    let newx = *x * c + *y * s;
    let newy = *y * c - *x * s;
    *x = newx;
    *y = newy;
}
fn complex_rot<T: Float>(s: Complex<T>, c: T, x: &mut Complex<T>, y: &mut Complex<T>) {
    let newx = *x * c + s * *y;
    let newy = *y * c - s.conj() * *x;
    *x = newx;
    *y = newy;
}
pub fn srot(n: usize, x: &mut [f32], incx: usize, y: &mut [f32], incy: usize, c: f32, s: f32) {
    if n.is_zero() {
        return;
    } else if incx.is_zero() && incy.is_zero() {
        for _ in 0..n {
            real_rot(s, c, &mut x[0], &mut y[0]);
        }
    } else if incx.is_zero() {
        y.into_iter()
            .step_by(incy)
            .take(n)
            .for_each(move |yi| real_rot(s, c, &mut x[0], yi));
    } else if incy.is_zero() {
        x.into_iter()
            .step_by(incx)
            .take(n)
            .for_each(move |xi| real_rot(s, c, xi, &mut y[0]));
    } else {
        x.into_iter()
            .step_by(incx)
            .take(n)
            .zip(y.into_iter().step_by(incy).take(n))
            .for_each(move |(xi, yi)| real_rot(s, c, xi, yi));
    }
}
pub fn drot(n: usize, x: &mut [f64], incx: usize, y: &mut [f64], incy: usize, c: f64, s: f64) {
    if n.is_zero() {
        return;
    } else if incx.is_zero() && incy.is_zero() {
        for _ in 0..n {
            real_rot(s, c, &mut x[0], &mut y[0]);
        }
    } else if incx.is_zero() {
        y.into_iter()
            .step_by(incy)
            .take(n)
            .for_each(move |yi| real_rot(s, c, &mut x[0], yi));
    } else if incy.is_zero() {
        x.into_iter()
            .step_by(incx)
            .take(n)
            .for_each(move |xi| real_rot(s, c, xi, &mut y[0]));
    } else {
        x.into_iter()
            .step_by(incx)
            .take(n)
            .zip(y.into_iter().step_by(incy).take(n))
            .for_each(move |(xi, yi)| real_rot(s, c, xi, yi));
    }
}
pub fn crot(
    n: usize,
    x: &mut [Complex32],
    incx: usize,
    y: &mut [Complex32],
    incy: usize,
    c: f32,
    s: Complex32,
) {
    if n.is_zero() {
        return;
    } else if incx.is_zero() && incy.is_zero() {
        for _ in 0..n {
            complex_rot(s, c, &mut x[0], &mut y[0]);
        }
    } else if incx.is_zero() {
        y.into_iter()
            .step_by(incy)
            .take(n)
            .for_each(move |yi| complex_rot(s, c, &mut x[0], yi));
    } else if incy.is_zero() {
        x.into_iter()
            .step_by(incx)
            .take(n)
            .for_each(move |xi| complex_rot(s, c, xi, &mut y[0]));
    } else {
        x.into_iter()
            .step_by(incx)
            .take(n)
            .zip(y.into_iter().step_by(incy).take(n))
            .for_each(move |(xi, yi)| complex_rot(s, c, xi, yi));
    }
}
pub fn zrot(
    n: usize,
    x: &mut [Complex64],
    incx: usize,
    y: &mut [Complex64],
    incy: usize,
    c: f64,
    s: Complex64,
) {
    if n.is_zero() {
        return;
    } else if incx.is_zero() && incy.is_zero() {
        for _ in 0..n {
            complex_rot(s, c, &mut x[0], &mut y[0]);
        }
    } else if incx.is_zero() {
        y.into_iter()
            .step_by(incy)
            .take(n)
            .for_each(move |yi| complex_rot(s, c, &mut x[0], yi));
    } else if incy.is_zero() {
        x.into_iter()
            .step_by(incx)
            .take(n)
            .for_each(move |xi| complex_rot(s, c, xi, &mut y[0]));
    } else {
        x.into_iter()
            .step_by(incx)
            .take(n)
            .zip(y.into_iter().step_by(incy).take(n))
            .for_each(move |(xi, yi)| complex_rot(s, c, xi, yi));
    }
}
pub fn csrot(
    n: usize,
    x: &mut [Complex32],
    incx: usize,
    y: &mut [Complex32],
    incy: usize,
    c: f32,
    s: f32,
) {
    if n.is_zero() {
        return;
    } else if incx.is_zero() && incy.is_zero() {
        for _ in 0..n {
            real_rot(s, c, &mut x[0], &mut y[0]);
        }
    } else if incx.is_zero() {
        y.into_iter()
            .step_by(incy)
            .take(n)
            .for_each(move |yi| real_rot(s, c, &mut x[0], yi));
    } else if incy.is_zero() {
        x.into_iter()
            .step_by(incx)
            .take(n)
            .for_each(move |xi| real_rot(s, c, xi, &mut y[0]));
    } else {
        x.into_iter()
            .step_by(incx)
            .take(n)
            .zip(y.into_iter().step_by(incy).take(n))
            .for_each(move |(xi, yi)| real_rot(s, c, xi, yi));
    }
}
pub fn zdrot(
    n: usize,
    x: &mut [Complex64],
    incx: usize,
    y: &mut [Complex64],
    incy: usize,
    c: f64,
    s: f64,
) {
    if n.is_zero() {
        return;
    } else if incx.is_zero() && incy.is_zero() {
        for _ in 0..n {
            real_rot(s, c, &mut x[0], &mut y[0]);
        }
    } else if incx.is_zero() {
        y.into_iter()
            .step_by(incy)
            .take(n)
            .for_each(move |yi| real_rot(s, c, &mut x[0], yi));
    } else if incy.is_zero() {
        x.into_iter()
            .step_by(incx)
            .take(n)
            .for_each(move |xi| real_rot(s, c, xi, &mut y[0]));
    } else {
        x.into_iter()
            .step_by(incx)
            .take(n)
            .zip(y.into_iter().step_by(incy).take(n))
            .for_each(move |(xi, yi)| real_rot(s, c, xi, yi));
    }
}
/// Returns the square of its argument
fn square<T: Copy + Mul<T, Output = T>>(x: T) -> T {
    x * x
}
/// Givens rotation
pub struct Givens<T, U = T> {
    c: U,
    s: T,
    r: T,
}
impl<T: Float> Givens<T, T> {
    fn real(a: T, b: T) -> (Self, T) {
        let anorm = a.abs();
        let bnorm = b.abs();
        if bnorm.is_zero() {
            (
                Givens {
                    c: T::one(),
                    s: T::zero(),
                    r: a,
                },
                T::zero(),
            )
        } else if anorm.is_zero() {
            (
                Givens {
                    c: T::zero(),
                    s: T::one(),
                    r: b,
                },
                T::one(),
            )
        } else {
            let sigma = if anorm > bnorm {
                a.signum()
            } else {
                b.signum()
            };
            let r = sigma * (square(a) + square(b)).sqrt();
            let c = a / r;
            let s = b / r;
            let z = if anorm > bnorm {
                s
            } else if c.is_zero() {
                T::one()
            } else {
                c.recip()
            };
            (Givens { c, s, r }, z)
        }
    }
}
impl<T: Float> Givens<Complex<T>, T> {
    fn complex(a: Complex<T>, b: Complex<T>) -> Self {
        let safmax: T = T::max_value();
        let safmin: T = T::min_positive_value();
        let rtmin: T = (safmin / (T::one() + T::one())).sqrt();
        if b.is_zero() {
            Givens {
                c: T::one(),
                s: Complex::zero(),
                r: a.clone(),
            }
        } else if a.is_zero() {
            let c = T::zero();
            if b.re.is_zero() {
                let r = b.im.abs();
                Givens {
                    c,
                    s: b.conj() / r,
                    r: Complex::new(r, T::zero()),
                }
            } else if b.im.is_zero() {
                let r = b.re.abs();
                Givens {
                    c,
                    s: b.conj() / r,
                    r: Complex::new(r, T::zero()),
                }
            } else {
                let rtmax: T = (safmax / (T::one() + T::one())).sqrt();
                let abs_max = b.re.abs().max(b.im.abs());
                if abs_max > rtmin && abs_max < rtmax {
                    // Unscaled algorithm
                    let r = b.norm();
                    Givens {
                        c,
                        s: b.conj() / r,
                        r: Complex::new(r, T::zero()),
                    }
                } else {
                    // Scaling everything to ensure numerical stability
                    let scale = safmax.min(safmin.max(abs_max));
                    let scaled = b / scale;
                    let norm = scaled.norm();
                    Givens {
                        c,
                        s: scaled.conj() / norm,
                        r: Complex::new(norm * scale, T::zero()),
                    }
                }
            }
        } else {
            let amax = a.re.abs().max(a.im.abs());
            let bmax = b.re.abs().max(b.im.abs());
            let mut rtmax: T = (safmax / (T::one() + T::one() + T::one() + T::one())).sqrt();
            if amax > rtmin && amax < rtmax && bmax > rtmin && bmax < rtmax {
                // Using unscaled algorithm
                let anorm2 = a.norm_sqr();
                let bnorm2 = b.norm_sqr();
                let h2 = anorm2 + bnorm2;
                if anorm2 > h2 * safmin {
                    let c = (anorm2 / h2).sqrt();
                    let r = a / c;
                    rtmax = rtmax + rtmax;
                    let s = if anorm2 > rtmin && h2 < rtmax {
                        b.conj() * (a / (anorm2 * h2).sqrt())
                    } else {
                        b.conj() * (r / h2)
                    };
                    Givens { c, r, s }
                } else {
                    // anorm / h may be subnormal and h / f may overflow
                    let d = (anorm2 * h2).sqrt();
                    let c = anorm2 / d;
                    let r = if c > safmin { a / c } else { a * (h2 / d) };
                    Givens {
                        c,
                        r,
                        s: b.conj() * (a / d),
                    }
                }
            } else {
                // Scaling everything
                let scale = safmax.min(safmin.max(amax.max(bmax)));
                let bscaled = b / scale;
                let bnorm2 = bscaled.norm_sqr();
                let (ascaled, anorm2, h2, w) = if amax / scale < rtmin {
                    let v = safmax.min(safmin.max(amax));
                    let w = v / scale;
                    let ascaled = a / v;
                    let anorm2 = ascaled.norm_sqr();
                    (ascaled, anorm2, anorm2 * w * w + bnorm2, w)
                } else {
                    let w = T::one();
                    let ascaled = a / scale;
                    let anorm2 = ascaled.norm_sqr();
                    (ascaled, anorm2, anorm2 + bnorm2, w)
                };
                if anorm2 > h2 * safmin {
                    let c = (anorm2 / h2).sqrt();
                    let r = ascaled / c;
                    rtmax = rtmax + rtmax;
                    let s = if anorm2 > rtmin && h2 < rtmax {
                        bscaled.conj() * (ascaled / (anorm2 * h2).sqrt())
                    } else {
                        bscaled.conj() * (r / h2)
                    };
                    Givens { c, s, r }
                } else {
                    let d = (anorm2 * h2).sqrt();
                    let c = anorm2 / d;
                    let r = if c > safmin {
                        ascaled / c
                    } else {
                        ascaled * (h2 / d)
                    };
                    let s = bscaled.conj() * (ascaled / d);
                    Givens {
                        c: c * w,
                        s,
                        r: r * scale,
                    }
                }
            }
        }
    }
}
/// Computes the parameters for a Givens rotation
pub fn srotg(a: f32, b: f32) -> (Givens<f32>, f32) {
    Givens::real(a, b)
}
/// Computes the parameters for a Givens rotation
pub fn drotg(a: f64, b: f64) -> (Givens<f64>, f64) {
    Givens::real(a, b)
}
/// Computes the parameters for a Given rotation
pub fn crotg(a: Complex32, b: Complex32) -> Givens<Complex32, f32> {
    Givens::complex(a, b)
}
/// Computes the parameters for a Given rotation
pub fn zrotg(a: Complex64, b: Complex64) -> Givens<Complex64, f64> {
    Givens::complex(a, b)
}
// TODO
// #[derive(Clone, Copy ,Debug)]
// pub enum MGivensFlag {
//     Raw,
//     NDiag,
//     Diag,
//     Id,
// }
// pub struct MGivens<T> {
//     flag: MGivensFlag,
//     h11: T,
//     h12: T,
//     h21: T,
//     h22: T,
// }
// impl<T: Float> MGivens<T> {
//     pub fn generate(d1: &mut T, d2: &mut T, x1: &mut T, y1: &T, flag: MGivensFlag) -> MGivensFlag<T> {
//
//         MGivens {h11, h12, h21, h22, flag}
//     }
//     /// Applies the modified givens rotation
//     pub fn apply<I1, I2>(&self, x: I1, y: I2)
//     where I1: Iterator,
//           I2: Iterator,
//           I1::Item: Copy + Deref<Target=T> + DerefMut,
//           I2::Item: Copy + Deref<Target=T> + DerefMut,
//     {
//         let (h11, h12, h21, h22) = match self.flag {
//             MGivensFlag::Raw => (self.h11, self.h12, self.h21, self.h22),
//             MGivensFlag::NDiag => (T::one(), self.h12, self.h21, T::one()),
//             MGivensFlag::Diag => (self.h11, T::one(), -T::one(), self.h22),
//             MGivensFlag::Id => { return; },
//         };
//         for (mut xi, mut yi) in x.zip(y) {
//             let new_x = *xi * h11 + *yi * h12;
//             let new_y = *xi * h21 + *yi * h22;
//             *xi = new_x;
//             *yi = new_y;
//         }
//     }
// }
// pub fn srotm(n: usize, x: &mut[f32], incx: usize, y: &mut[f32], incy: usize, givens: &MGivens<f32>) {
//     givens.apply(
//         x.into_iter().step_by(incx).take(n),
//         y.into_iter().step_by(incy).take(n),
//     )
// }
// pub fn drotm(n: usize, x: &mut[f64], incx: usize, y: &mut[f64], incy: usize, givens: &MGivens<f64>) {
//     givens.apply(
//         x.into_iter().step_by(incx).take(n),
//         y.into_iter().step_by(incy).take(n),
//     )
// }
pub fn iscal<T, I>(a: T, x: I)
where
    I: Iterator,
    I::Item: DerefMut,
    <I::Item as Deref>::Target: MulAssign<T>,
    T: Copy,
{
    x.for_each(move |mut xi| {
        *xi *= a;
    })
}
pub fn scal<T>(n: usize, a: T, x: &mut [T], inc: usize)
where
    T: Zero + One + Copy + MulAssign<T> + PartialEq,
{
    if n.is_zero() {
        return;
    }
    if a.is_zero() {
        fill(n, &a, x, inc);
        return;
    } else if a.is_one() {
        return;
    }
    if inc.is_zero() {
        x[0] *= a;
    } else {
        iscal(a, x.into_iter().step_by(inc).take(n))
    }
}
pub fn sscal(n: usize, a: f32, x: &mut [f32], inc: usize) {
    if n.is_zero() {
        return;
    }
    if a.is_zero() {
        sfill(n, a, x, inc);
        return;
    } else if a.is_one() {
        return;
    }
    if inc.is_zero() {
        x[0] *= a;
    } else {
        iscal(a, x.into_iter().step_by(inc).take(n))
    }
}
pub fn dscal(n: usize, a: f64, x: &mut [f64], inc: usize) {
    if n.is_zero() {
        return;
    }
    if a.is_zero() {
        dfill(n, a, x, inc);
        return;
    } else if a.is_one() {
        return;
    }
    if inc.is_zero() {
        x[0] *= a;
    } else {
        iscal(a, x.into_iter().step_by(inc).take(n))
    }
}
pub fn cscal(n: usize, a: Complex32, x: &mut [Complex32], inc: usize) {
    if n.is_zero() {
        return;
    }
    if a.is_zero() {
        cfill(n, a, x, inc);
        return;
    } else if a.is_one() {
        return;
    }
    if inc.is_zero() {
        x[0] *= a;
    } else {
        iscal(a, x.into_iter().step_by(inc).take(n))
    }
}
pub fn zscal(n: usize, a: Complex64, x: &mut [Complex64], inc: usize) {
    if n.is_zero() {
        return;
    }
    if a.is_zero() {
        zfill(n, a, x, inc);
        return;
    } else if a.is_one() {
        return;
    }
    if inc.is_zero() {
        x[0] *= a;
    } else {
        iscal(a, x.into_iter().step_by(inc).take(n))
    }
}
pub fn csscal(n: usize, a: f32, x: &mut [Complex32], inc: usize) {
    if n.is_zero() {
        return;
    }
    if a.is_zero() {
        cfill(n, Complex32::zero(), x, inc);
        return;
    }
    if inc.is_zero() {
        x[0] *= a;
    } else {
        iscal(a, x.into_iter().step_by(inc).take(n))
    }
}
pub fn zdscal(n: usize, a: f64, x: &mut [Complex64], inc: usize) {
    if n.is_zero() {
        return;
    }
    if a.is_zero() {
        zfill(n, Complex64::zero(), x, inc);
        return;
    }
    if inc.is_zero() {
        x[0] *= a;
    } else {
        iscal(a, x.into_iter().step_by(inc).take(n))
    }
}
pub fn ishift<T, I>(a: T, x: I)
where
    I: Iterator,
    I::Item: DerefMut,
    <I::Item as Deref>::Target: AddAssign<T>,
    T: Copy,
{
    x.for_each(move |mut xi| {
        *xi += a;
    })
}
pub fn shift<T>(n: usize, a: T, x: &mut [T], inc: usize)
where
    T: Copy + AddAssign<T>,
{
    if n.is_zero() {
        return;
    }
    if inc.is_zero() {
        x[0] += a;
    } else {
        ishift(a, x.into_iter().step_by(inc).take(n))
    }
}
pub fn sshift(n: usize, a: f32, x: &mut [f32], inc: usize) {
    if n.is_zero() || a.is_zero() {
        return;
    }
    if inc.is_zero() {
        x[0] += a;
    } else {
        ishift(a, x.into_iter().step_by(inc).take(n))
    }
}
pub fn dshift(n: usize, a: f64, x: &mut [f64], inc: usize) {
    if n.is_zero() || a.is_zero() {
        return;
    }
    if inc.is_zero() {
        x[0] += a;
    } else {
        ishift(a, x.into_iter().step_by(inc).take(n))
    }
}
pub fn cshift(n: usize, a: Complex32, x: &mut [Complex32], inc: usize) {
    if n.is_zero() || a.is_zero() {
        return;
    }
    if inc.is_zero() {
        x[0] += a;
    } else {
        ishift(a, x.into_iter().step_by(inc).take(n))
    }
}
pub fn zshift(n: usize, a: Complex64, x: &mut [Complex64], inc: usize) {
    if n.is_zero() || a.is_zero() {
        return;
    }
    if inc.is_zero() {
        x[0] += a;
    } else {
        ishift(a, x.into_iter().step_by(inc).take(n))
    }
}
pub fn fill<T: Clone>(n: usize, value: &T, x: &mut [T], incx: usize) {
    if n.is_zero() {
        return;
    } else if incx.is_zero() {
        x[0] = value.clone();
    } else {
        x.into_iter().step_by(incx).take(n).for_each(move |xi| {
            *xi = value.clone();
        })
    }
}
pub fn afill<T: Copy>(n: usize, value: T, x: &mut [T], incx: usize) {
    if n.is_zero() {
        return;
    } else if incx.is_zero() {
        x[0] = value;
    } else {
        x.into_iter().step_by(incx).take(n).for_each(move |xi| {
            *xi = value;
        })
    }
}
pub fn sfill(n: usize, value: f32, x: &mut [f32], incx: usize) {
    if n.is_zero() {
        return;
    } else if incx.is_zero() {
        x[0] = value;
    } else {
        x.into_iter().step_by(incx).take(n).for_each(move |xi| {
            *xi = value;
        })
    }
}
pub fn dfill(n: usize, value: f64, x: &mut [f64], incx: usize) {
    if n.is_zero() {
        return;
    } else if incx.is_zero() {
        x[0] = value;
    } else {
        x.into_iter().step_by(incx).take(n).for_each(move |xi| {
            *xi = value;
        })
    }
}

pub fn cfill(n: usize, value: Complex32, x: &mut [Complex32], incx: usize) {
    if n.is_zero() {
        return;
    } else if incx.is_zero() {
        x[0] = value;
    } else {
        x.into_iter().step_by(incx).take(n).for_each(move |xi| {
            *xi = value;
        })
    }
}

pub fn zfill(n: usize, value: Complex64, x: &mut [Complex64], incx: usize) {
    if n.is_zero() {
        return;
    } else if incx.is_zero() {
        x[0] = value;
    } else {
        x.into_iter().step_by(incx).take(n).for_each(move |xi| {
            *xi = value;
        })
    }
}
pub fn swap<'a, I1, I2, T>(x: I1, y: I2)
where
    T: 'a,
    I1: Iterator<Item = &'a mut T>,
    I2: Iterator<Item = &'a mut T>,
{
    x.zip(y).for_each(|(xi, yi)| std::mem::swap(xi, yi))
}
pub fn aswap<T: Clone>(n: usize, x: &mut [T], incx: usize, y: &mut [T], incy: usize) {
    if n.is_zero() {
        return;
    }
    if incx.is_zero() && incy.is_zero() {
        std::mem::swap(&mut x[0], &mut y[0]);
    } else if incx.is_zero() {
        let imx = (n - 1) * incy;
        std::mem::swap(&mut x[0], &mut y[imx]);
        if imx > 0 {
            let (start, stop) = y.split_at_mut(imx - 1);
            fill(n - 1, &stop[0], start, incy);
        }
    } else if incy.is_zero() {
        let imx = (n - 1) * incx;
        std::mem::swap(&mut x[imx], &mut y[0]);
        if imx > 0 {
            let (start, stop) = x.split_at_mut(imx - 1);
            fill(n - 1, &stop[0], start, incy);
        }
    } else {
        swap(
            x.into_iter().step_by(incx).take(n),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn sswap(n: usize, x: &mut [f32], incx: usize, y: &mut [f32], incy: usize) {
    if n.is_zero() {
        return;
    }
    if incx.is_zero() && incy.is_zero() {
        std::mem::swap(&mut x[0], &mut y[0]);
    } else if incx.is_zero() {
        let imx = (n - 1) * incy;
        std::mem::swap(&mut x[0], &mut y[imx]);
        if imx > 0 {
            let (start, stop) = y.split_at_mut(imx - 1);
            afill(n - 1, stop[0], start, incy);
        }
    } else if incy.is_zero() {
        let imx = (n - 1) * incx;
        std::mem::swap(&mut x[imx], &mut y[0]);
        if imx > 0 {
            let (start, stop) = x.split_at_mut(imx - 1);
            afill(n - 1, stop[0], start, incy);
        }
    } else {
        swap(
            x.into_iter().step_by(incx).take(n),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn dswap(n: usize, x: &mut [f64], incx: usize, y: &mut [f64], incy: usize) {
    if n.is_zero() {
        return;
    }
    if incx.is_zero() && incy.is_zero() {
        std::mem::swap(&mut x[0], &mut y[0]);
    } else if incx.is_zero() {
        let imx = (n - 1) * incy;
        std::mem::swap(&mut x[0], &mut y[imx]);
        if imx > 0 {
            let (start, stop) = y.split_at_mut(imx - 1);
            afill(n - 1, stop[0], start, incy);
        }
    } else if incy.is_zero() {
        let imx = (n - 1) * incx;
        std::mem::swap(&mut x[imx], &mut y[0]);
        if imx > 0 {
            let (start, stop) = x.split_at_mut(imx - 1);
            afill(n - 1, stop[0], start, incy);
        }
    } else {
        swap(
            x.into_iter().step_by(incx).take(n),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn cswap(n: usize, x: &mut [Complex32], incx: usize, y: &mut [Complex32], incy: usize) {
    if n.is_zero() {
        return;
    }
    if incx.is_zero() && incy.is_zero() {
        std::mem::swap(&mut x[0], &mut y[0]);
    } else if incx.is_zero() {
        let imx = (n - 1) * incy;
        std::mem::swap(&mut x[0], &mut y[imx]);
        if imx > 0 {
            let (start, stop) = y.split_at_mut(imx - 1);
            afill(n - 1, stop[0], start, incy);
        }
    } else if incy.is_zero() {
        let imx = (n - 1) * incx;
        std::mem::swap(&mut x[imx], &mut y[0]);
        if imx > 0 {
            let (start, stop) = x.split_at_mut(imx - 1);
            afill(n - 1, stop[0], start, incy);
        }
    } else {
        swap(
            x.into_iter().step_by(incx).take(n),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
pub fn zswap(n: usize, x: &mut [Complex64], incx: usize, y: &mut [Complex64], incy: usize) {
    if n.is_zero() {
        return;
    }
    if incx.is_zero() && incy.is_zero() {
        std::mem::swap(&mut x[0], &mut y[0]);
    } else if incx.is_zero() {
        let imx = (n - 1) * incy;
        std::mem::swap(&mut x[0], &mut y[imx]);
        if imx > 0 {
            let (start, stop) = y.split_at_mut(imx - 1);
            afill(n - 1, stop[0], start, incy);
        }
    } else if incy.is_zero() {
        let imx = (n - 1) * incx;
        std::mem::swap(&mut x[imx], &mut y[0]);
        if imx > 0 {
            let (start, stop) = x.split_at_mut(imx - 1);
            afill(n - 1, stop[0], start, incy);
        }
    } else {
        swap(
            x.into_iter().step_by(incx).take(n),
            y.into_iter().step_by(incy).take(n),
        )
    }
}
/// Finds the index of the extremum element using the provided comparison
///
/// # Arguments
///  * `x` - the iterator for which the index is required
///  * `cmp` - compare the current item to the running extremum. If this comparison returns `true`,
///            the running extremum and its index are updated
///  * `early_exit` - a condition on the items. If this condition is verified, the function returns
///                   immediately with the current index
///
/// # Returns
/// Either the index of the extremum (according to the provided comparison) within the iterator or
/// the index of the first item matching the `early_exit`.
/// If the iterator is empty, this returns `0`
fn iextr<I, P, Q>(mut x: I, cmp: P, early_exit: Q) -> usize
where
    I: Iterator,
    P: Fn(&I::Item, &I::Item) -> bool,
    Q: Fn(&I::Item) -> bool,
{
    let mut out = 0;
    let mut xextr = match x.next() {
        Some(xi) if early_exit(&xi) => {
            return 0;
        }
        None => return 0,
        Some(xi) => xi,
    };
    for (i, xi) in x.enumerate() {
        if early_exit(&xi) {
            return i + 1;
        } else if cmp(&xi, &xextr) {
            xextr = xi;
            out = i + 1;
        }
    }
    out
}
/// Finds the index of the first maximum element
pub fn imax<I>(x: I) -> usize
where
    I: Iterator,
    I::Item: Float,
{
    iextr(
        x,
        |x, mx| *x > *mx,
        |x| x.is_nan() || (x.is_infinite() && x.is_sign_positive()),
    )
}
/// Finds the index of the first minimum element
pub fn imin<I>(x: I) -> usize
where
    I: Iterator,
    I::Item: Float,
{
    iextr(
        x,
        |x, mx| x < mx,
        |x| x.is_nan() || (x.is_infinite() && x.is_sign_positive()),
    )
}
/// Returns the index of the maximum element
pub fn isamax(n: usize, x: &[f32], incx: usize) -> usize {
    imax(x.into_iter().step_by(incx).take(n).map(|x| x.abs()))
}
/// Returns the index of the maximum element
pub fn idamax(n: usize, x: &[f64], incx: usize) -> usize {
    imax(x.into_iter().step_by(incx).take(n).map(|x| x.abs()))
}
/// Returns the index of the maximum element
pub fn icamax(n: usize, x: &[Complex32], incx: usize) -> usize {
    imax(x.into_iter().step_by(incx).take(n).map(|x| x.norm_sqr()))
}
/// Returns the index of the maximum element
pub fn izamax(n: usize, x: &[Complex64], incx: usize) -> usize {
    imax(x.into_iter().step_by(incx).take(n).map(|x| x.norm_sqr()))
}
/// Returns the index of the maximum element
pub fn isamin(n: usize, x: &[f32], incx: usize) -> usize {
    imin(x.into_iter().step_by(incx).take(n).map(|x| x.abs()))
}
/// Returns the index of the maximum element
pub fn idamin(n: usize, x: &[f64], incx: usize) -> usize {
    imin(x.into_iter().step_by(incx).take(n).map(|x| x.abs()))
}
/// Returns the index of the maximum element
pub fn icamin(n: usize, x: &[Complex32], incx: usize) -> usize {
    imin(x.into_iter().step_by(incx).take(n).map(|x| x.norm_sqr()))
}
/// Returns the index of the maximum element
pub fn izamin(n: usize, x: &[Complex64], incx: usize) -> usize {
    imin(x.into_iter().step_by(incx).take(n).map(|x| x.norm_sqr()))
}
pub trait ASum<S: Summation<Self::Output>>: Sized {
    type Output;
    fn asum(n: usize, arr: &[Self], inc: usize) -> Self::Output;
}
impl ASum<f32> for f32 {
    type Output = f32;
    fn asum(n: usize, arr: &[Self], inc: usize) -> f32 {
        sasum(n, arr, inc)
    }
}
impl ASum<f64> for f64 {
    type Output = f64;
    fn asum(n: usize, arr: &[Self], inc: usize) -> f64 {
        dasum(n, arr, inc)
    }
}
impl ASum<f32> for Complex32 {
    type Output = f32;
    fn asum(n: usize, arr: &[Self], inc: usize) -> f32 {
        casum(n, arr, inc)
    }
}
impl ASum<f64> for Complex64 {
    type Output = f64;
    fn asum(n: usize, arr: &[Self], inc: usize) -> f64 {
        zasum(n, arr, inc)
    }
}
pub trait Axpy: Sized {
    /// y := a . x + y
    fn axpy(n: usize, a: Self, x: &[Self], incx: usize, y: &mut [Self], incy: usize);
}
impl Axpy for f32 {
    fn axpy(n: usize, a: Self, x: &[Self], incx: usize, y: &mut [Self], incy: usize) {
        saxpy(n, a, x, incx, y, incy)
    }
}
impl Axpy for f64 {
    fn axpy(n: usize, a: Self, x: &[Self], incx: usize, y: &mut [Self], incy: usize) {
        daxpy(n, a, x, incx, y, incy)
    }
}
impl Axpy for Complex32 {
    fn axpy(n: usize, a: Self, x: &[Self], incx: usize, y: &mut [Self], incy: usize) {
        caxpy(n, a, x, incx, y, incy)
    }
}
impl Axpy for Complex64 {
    fn axpy(n: usize, a: Self, x: &[Self], incx: usize, y: &mut [Self], incy: usize) {
        zaxpy(n, a, x, incx, y, incy)
    }
}
pub trait Axpyc: Sized {
    /// y := a . conj(x) + y
    fn axpyc(n: usize, a: Self, x: &[Self], incx: usize, y: &mut [Self], incy: usize);
    /// y := a . x + y
    fn axpyu(n: usize, a: Self, x: &[Self], incx: usize, y: &mut [Self], incy: usize);
}
impl Axpyc for Complex32 {
    fn axpyc(n: usize, a: Self, x: &[Self], incx: usize, y: &mut [Self], incy: usize) {
        caxpyc(n, a, x, incx, y, incy)
    }
    fn axpyu(n: usize, a: Self, x: &[Self], incx: usize, y: &mut [Self], incy: usize) {
        caxpy(n, a, x, incx, y, incy)
    }
}
impl Axpyc for Complex64 {
    fn axpyc(n: usize, a: Self, x: &[Self], incx: usize, y: &mut [Self], incy: usize) {
        zaxpyc(n, a, x, incx, y, incy)
    }
    fn axpyu(n: usize, a: Self, x: &[Self], incx: usize, y: &mut [Self], incy: usize) {
        zaxpy(n, a, x, incx, y, incy)
    }
}
pub trait Copyv: Sized {
    fn copy(n: usize, src: &[Self], incx: usize, dst: &mut [Self], incy: usize);
}
impl Copyv for f32 {
    fn copy(n: usize, src: &[Self], incx: usize, dst: &mut [Self], incy: usize) {
        scopy(n, src, incx, dst, incy)
    }
}
impl Copyv for f64 {
    fn copy(n: usize, src: &[Self], incx: usize, dst: &mut [Self], incy: usize) {
        dcopy(n, src, incx, dst, incy)
    }
}
impl Copyv for Complex32 {
    fn copy(n: usize, src: &[Self], incx: usize, dst: &mut [Self], incy: usize) {
        ccopy(n, src, incx, dst, incy)
    }
}
impl Copyv for Complex64 {
    fn copy(n: usize, src: &[Self], incx: usize, dst: &mut [Self], incy: usize) {
        zcopy(n, src, incx, dst, incy)
    }
}
pub trait Dot<S = Self>: Sized + Summation<Self> {
    fn dot(n: usize, x: &[Self], incx: usize, y: &[Self], incy: usize) -> Self;
}
pub trait Dotc<S = Self>: Sized + Summation<Self> {
    /// Dot product unconjugated `x' . y`
    fn dotu(n: usize, x: &[Self], incx: usize, y: &[Self], incy: usize) -> Self;
    /// Dot product conjugated `conj(x') . y`
    fn dotc(n: usize, x: &[Self], incx: usize, y: &[Self], incy: usize) -> Self;
}
impl Dot for f32 {
    fn dot(n: usize, x: &[Self], incx: usize, y: &[Self], incy: usize) -> f32 {
        sdot(n, x, incx, y, incy)
    }
}
impl Dot for f64 {
    fn dot(n: usize, x: &[Self], incx: usize, y: &[Self], incy: usize) -> f64 {
        ddot(n, x, incx, y, incy)
    }
}
impl Dot<Kahan<f32>> for f32 {
    fn dot(n: usize, x: &[f32], incx: usize, y: &[f32], incy: usize) -> f32 {
        if n.is_zero() {
            0.0f32
        } else if incx.is_zero() && incy.is_zero() {
            (n as f32) * x[0] * y[0]
        } else if incx.is_zero() {
            x[0] * Kahan::<f32>::sum(y.into_iter().step_by(incy).take(n))
        } else if incy.is_zero() {
            y[0] * Kahan::<f32>::sum(x.into_iter().step_by(incx).take(n))
        } else {
            idot(
                Kahan::<f32>::zero(),
                x.into_iter().step_by(incx).take(n),
                y.into_iter().step_by(incy).take(n),
            )
        }
    }
}
impl Dot<f64> for f32 {
    fn dot(n: usize, x: &[Self], incx: usize, y: &[Self], incy: usize) -> f32 {
        sdsdot(n, 0.0, x, incx, y, incy)
    }
}
impl Dot<Kahan<f64>> for f64 {
    fn dot(n: usize, x: &[f64], incx: usize, y: &[f64], incy: usize) -> f64 {
        if n.is_zero() {
            0.0f64
        } else if incx.is_zero() && incy.is_zero() {
            (n as f64) * x[0] * y[0]
        } else if incx.is_zero() {
            x[0] * Kahan::<f64>::sum(y.into_iter().step_by(incy).take(n))
        } else if incy.is_zero() {
            y[0] * Kahan::<f64>::sum(x.into_iter().step_by(incx).take(n))
        } else {
            idot(
                Kahan::<f64>::zero(),
                x.into_iter().step_by(incx).take(n),
                y.into_iter().step_by(incy).take(n),
            )
        }
    }
}
impl Dotc for Complex32 {
    /// Dot product unconjugated `x' . y`
    fn dotu(n: usize, x: &[Self], incx: usize, y: &[Self], incy: usize) -> Complex32 {
        cdotu(n, x, incx, y, incy)
    }
    /// Dot product conjugated `conj(x') . y`
    fn dotc(n: usize, x: &[Self], incx: usize, y: &[Self], incy: usize) -> Complex32 {
        cdotc(n, x, incx, y, incy)
    }
}
impl Dotc for Complex64 {
    fn dotu(n: usize, x: &[Self], incx: usize, y: &[Self], incy: usize) -> Complex64 {
        zdotu(n, x, incx, y, incy)
    }
    fn dotc(n: usize, x: &[Self], incx: usize, y: &[Self], incy: usize) -> Complex64 {
        zdotc(n, x, incx, y, incy)
    }
}
impl Dotc<Kahan<Complex32>> for Complex32 {
    /// Dot product unconjugated `x' . y`
    fn dotu(n: usize, x: &[Self], incx: usize, y: &[Self], incy: usize) -> Complex32 {
        if n.is_zero() {
            Complex::zero()
        } else if incx.is_zero() && incy.is_zero() {
            x[0] * y[0] * (n as f32)
        } else if incx.is_zero() {
            x[0] * Kahan::<Complex32>::sum(y.into_iter().step_by(incy).take(n))
        } else if incy.is_zero() {
            y[0] * Kahan::<Complex32>::sum(x.into_iter().step_by(incx).take(n))
        } else {
            idot(
                Kahan::<Complex32>::zero(),
                x.into_iter().step_by(incx).take(n),
                y.into_iter().step_by(incy).take(n),
            )
        }
    }
    /// Dot product conjugated `conj(x') . y`
    fn dotc(n: usize, x: &[Self], incx: usize, y: &[Self], incy: usize) -> Complex32 {
        if n.is_zero() {
            Complex::zero()
        } else if incx.is_zero() && incy.is_zero() {
            x[0].conj() * y[0] * (n as f32)
        } else if incx.is_zero() {
            x[0].conj() * Kahan::<Complex32>::sum(y.into_iter().step_by(incy).take(n))
        } else if incy.is_zero() {
            y[0] * Kahan::<Complex32>::sum(x
                .into_iter()
                .step_by(incx)
                .take(n)
            ).conj()
        } else {
            idot(
                Kahan::<Complex32>::zero(),
                x.into_iter().step_by(incx).take(n).map(|c| c.conj()),
                y.into_iter().step_by(incy).take(n),
            )
        }
    }
}
impl Dotc<Kahan<Complex64>> for Complex64 {
    fn dotu(n: usize, x: &[Self], incx: usize, y: &[Self], incy: usize) -> Complex64 {
        if n.is_zero() {
            Complex::zero()
        } else if incx.is_zero() && incy.is_zero() {
            x[0] * y[0] * (n as f64)
        } else if incx.is_zero() {
            x[0] * Kahan::<Complex64>::sum(y.into_iter().step_by(incy).take(n))
        } else if incy.is_zero() {
            y[0] * Kahan::<Complex64>::sum(x.into_iter().step_by(incx).take(n))
        } else {
            idot(
                Kahan::<Complex64>::zero(),
                x.into_iter().step_by(incx).take(n),
                y.into_iter().step_by(incy).take(n),
            )
        }
    }
    fn dotc(n: usize, x: &[Self], incx: usize, y: &[Self], incy: usize) -> Complex64 {
        if n.is_zero() {
            Complex::zero()
        } else if incx.is_zero() && incy.is_zero() {
            x[0].conj() * y[0] * (n as f64)
        } else if incx.is_zero() {
            x[0].conj() * Kahan::<Complex64>::sum(y.into_iter().step_by(incy).take(n))
        } else if incy.is_zero() {
            y[0] * Kahan::<Complex64>::sum(x
                .into_iter()
                .step_by(incx)
                .take(n)
            ).conj()
        } else {
            idot(
                Kahan::<Complex64>::zero(),
                x.into_iter().step_by(incx).take(n).map(|c| c.conj()),
                y.into_iter().step_by(incy).take(n),
            )
        }
    }
}
pub trait Nrm2<S: Summation<Self::Output>>: Sized {
    type Output;
    fn nrm2(n: usize, vec: &[Self], inc: usize) -> Self::Output;
}
impl Nrm2<f32> for f32 {
    type Output = f32;
    fn nrm2(n: usize, vec: &[Self], inc: usize) -> Self::Output {
        snrm2(n, vec, inc)
    }
}
impl Nrm2<f64> for f64 {
    type Output = f64;
    fn nrm2(n: usize, vec: &[Self], inc: usize) -> Self::Output {
        dnrm2(n, vec, inc)
    }
}
impl Nrm2<f32> for Complex32 {
    type Output = f32;
    fn nrm2(n: usize, vec: &[Self], inc: usize) -> Self::Output {
        scnrm2(n, vec, inc)
    }
}
impl Nrm2<f64> for Complex64 {
    type Output = f64;
    fn nrm2(n: usize, vec: &[Self], inc: usize) -> Self::Output {
        dznrm2(n, vec, inc)
    }
}
pub trait Rot<T, U = T>: Sized {
    fn rot(n: usize, x: &mut [Self], incx: usize, y: &mut [Self], incy: usize, c: T, s: U);
}
impl Rot<f32> for f32 {
    fn rot(n: usize, x: &mut [Self], incx: usize, y: &mut [Self], incy: usize, c: f32, s: f32) {
        srot(n, x, incx, y, incy, c, s)
    }
}
impl Rot<f64> for f64 {
    fn rot(n: usize, x: &mut [Self], incx: usize, y: &mut [Self], incy: usize, c: f64, s: f64) {
        drot(n, x, incx, y, incy, c, s)
    }
}
impl Rot<f32, Complex32> for Complex32 {
    fn rot(
        n: usize,
        x: &mut [Self],
        incx: usize,
        y: &mut [Self],
        incy: usize,
        c: f32,
        s: Complex32,
    ) {
        crot(n, x, incx, y, incy, c, s)
    }
}
impl Rot<f32, f32> for Complex32 {
    fn rot(n: usize, x: &mut [Self], incx: usize, y: &mut [Self], incy: usize, c: f32, s: f32) {
        csrot(n, x, incx, y, incy, c, s)
    }
}
impl Rot<f64, Complex64> for Complex64 {
    fn rot(
        n: usize,
        x: &mut [Self],
        incx: usize,
        y: &mut [Self],
        incy: usize,
        c: f64,
        s: Complex64,
    ) {
        zrot(n, x, incx, y, incy, c, s)
    }
}
impl Rot<f64, f64> for Complex64 {
    fn rot(n: usize, x: &mut [Self], incx: usize, y: &mut [Self], incy: usize, c: f64, s: f64) {
        zdrot(n, x, incx, y, incy, c, s)
    }
}
pub trait Scal: Sized {
    fn scal(n: usize, scale: Self, vec: &mut [Self], inc: usize);
}
impl Scal for f32 {
    fn scal(n: usize, scale: f32, vec: &mut [f32], inc: usize) {
        sscal(n, scale, vec, inc)
    }
}
impl Scal for f64 {
    fn scal(n: usize, scale: f64, vec: &mut [f64], inc: usize) {
        dscal(n, scale, vec, inc)
    }
}
impl Scal for Complex32 {
    fn scal(n: usize, scale: Complex32, vec: &mut [Complex32], inc: usize) {
        cscal(n, scale, vec, inc)
    }
}
impl Scal for Complex64 {
    fn scal(n: usize, scale: Complex64, vec: &mut [Complex64], inc: usize) {
        zscal(n, scale, vec, inc)
    }
}
pub trait Shift: Sized {
    fn shift(n: usize, value: Self, vec: &mut [Self], inc: usize);
}
impl Shift for f32 {
    fn shift(n: usize, value: f32, vec: &mut [f32], inc: usize) {
        sshift(n, value, vec, inc)
    }
}
impl Shift for f64 {
    fn shift(n: usize, value: f64, vec: &mut [f64], inc: usize) {
        dshift(n, value, vec, inc)
    }
}
impl Shift for Complex32 {
    fn shift(n: usize, value: Complex32, vec: &mut [Complex32], inc: usize) {
        cshift(n, value, vec, inc)
    }
}
impl Shift for Complex64 {
    fn shift(n: usize, value: Complex64, vec: &mut [Complex64], inc: usize) {
        zshift(n, value, vec, inc)
    }
}
pub trait Fill: Sized {
    fn fill(n: usize, value: Self, vec: &mut [Self], inc: usize);
}
impl Fill for f32 {
    fn fill(n: usize, value: f32, vec: &mut [f32], inc: usize) {
        sfill(n, value, vec, inc)
    }
}
impl Fill for f64 {
    fn fill(n: usize, value: f64, vec: &mut [f64], inc: usize) {
        dfill(n, value, vec, inc)
    }
}
impl Fill for Complex32 {
    fn fill(n: usize, value: Complex32, vec: &mut [Complex32], inc: usize) {
        cfill(n, value, vec, inc)
    }
}
impl Fill for Complex64 {
    fn fill(n: usize, value: Complex64, vec: &mut [Complex64], inc: usize) {
        zfill(n, value, vec, inc)
    }
}
pub trait Swap: Sized {
    fn swap(n: usize, x: &mut [Self], incx: usize, y: &mut [Self], incy: usize);
}
impl Swap for f32 {
    fn swap(n: usize, x: &mut [Self], incx: usize, y: &mut [Self], incy: usize) {
        sswap(n, x, incx, y, incy)
    }
}
impl Swap for f64 {
    fn swap(n: usize, x: &mut [Self], incx: usize, y: &mut [Self], incy: usize) {
        dswap(n, x, incx, y, incy)
    }
}
impl Swap for Complex32 {
    fn swap(n: usize, x: &mut [Self], incx: usize, y: &mut [Self], incy: usize) {
        cswap(n, x, incx, y, incy)
    }
}
impl Swap for Complex64 {
    fn swap(n: usize, x: &mut [Self], incx: usize, y: &mut [Self], incy: usize) {
        zswap(n, x, incx, y, incy)
    }
}
pub trait IAopt : Sized {
    fn iamin(n: usize, x: &[Self], incx: usize) -> usize;
    fn iamax(n: usize, x: &[Self], incx: usize) -> usize;
}
impl IAopt for f32 {
    fn iamin(n: usize, x: &[f32], incx: usize) -> usize {
        isamin(n, x, incx)
    }
    fn iamax(n: usize, x: &[f32], incx: usize) -> usize {
        isamax(n, x, incx)
    }
}
impl IAopt for f64 {
    fn iamin(n: usize, x: &[f64], incx: usize) -> usize {
        idamin(n, x, incx)
    }
    fn iamax(n: usize, x: &[f64], incx: usize) -> usize {
        idamax(n, x, incx)
    }
}
impl IAopt for Complex32 {
    fn iamin(n: usize, x: &[Complex32], incx: usize) -> usize {
        icamin(n, x, incx)
    }
    fn iamax(n: usize, x: &[Complex32], incx: usize) -> usize {
        icamax(n, x, incx)
    }
}
impl IAopt for Complex64 {
    fn iamin(n: usize, x: &[Complex64], incx: usize) -> usize {
        izamin(n, x, incx)
    }
    fn iamax(n: usize, x: &[Complex64], incx: usize) -> usize {
        izamax(n, x, incx)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use approx::AbsDiffEq;
    #[test]
    fn test_anysum() {
        let sum = asum::<i32, _, _, _>(
            10,
            &vec![
                true, false, false, true, true, false, false, true, false, true,
            ],
            1,
            |&b| if b { 1 } else { 0 },
        );
        assert_eq!(sum, 5);
    }
    #[test]
    fn test_asum32() {
        let sum = sasum(
            10,
            &vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            1,
        );
        assert_eq!(sum, 10.0);
    }
    #[test]
    fn test_asum64() {
        let sum = dasum(
            10,
            &vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            1,
        );
        assert_eq!(sum, 10.0);
    }
    #[test]
    fn test_casum32() {
        let sum = casum(
            5,
            &vec![
                Complex32::new(1.0, 0.0),
                Complex32::new(0.0, 1.0),
                Complex32::new(-1.0, 0.0),
                Complex32::new(0.0, -1.0),
                Complex32::new(1.0, 0.0),
            ],
            1,
        );
        assert_eq!(sum, 5.0);
    }
    #[test]
    fn test_casum64() {
        let sum = zasum(
            5,
            &vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(1.0, 0.0),
            ],
            1,
        );
        assert_eq!(sum, 5.0);
    }
    #[test]
    fn test_saxpy() {
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![1.0, 1.0, 1.0];
        saxpy(3, 10.0, &x, 1, &mut y, 1);
        let expected = vec![11.0, 21.0, 31.0f32];
        for (yi, ei) in y.into_iter().zip(expected) {
            assert_eq!(yi, ei);
        }
    }
    #[test]
    fn test_daxpy() {
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![1.0, 1.0, 1.0];
        daxpy(3, 10.0, &x, 1, &mut y, 1);
        let expected = vec![11.0, 21.0, 31.0];
        for (yi, ei) in y.into_iter().zip(expected) {
            assert_eq!(yi, ei);
        }
    }
    #[test]
    fn test_caxpy() {
        let x = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
        ];
        let mut y = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(1.0, 0.0),
        ];
        caxpy(3, Complex32::i(), &x, 1, &mut y, 1);
        let expected = vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(1.0, 2.0),
            Complex32::new(1.0, 3.0),
        ];
        for (yi, ei) in y.into_iter().zip(expected) {
            assert_eq!(yi, ei);
        }
    }
    #[test]
    fn test_zaxpy() {
        let x = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ];
        let mut y = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        zaxpy(3, Complex64::i(), &x, 1, &mut y, 1);
        let expected = vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(1.0, 2.0),
            Complex64::new(1.0, 3.0),
        ];
        for (yi, ei) in y.into_iter().zip(expected) {
            assert_eq!(yi, ei);
        }
    }
    #[test]
    fn test_acopy() {
        let x = vec![true, false, true, false, true, false];
        let mut y = vec![false, false, false];
        copy(3, &x, 2, &mut y, 1);
        let expected = vec![true, true, true];
        for (yi, ei) in y.into_iter().zip(expected) {
            assert_eq!(yi, ei);
        }
    }
    #[test]
    fn test_scopy() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut y = vec![0.0, 0.0, 0.0];
        scopy(3, &x, 2, &mut y, 1);
        let expected = vec![1.0, 3.0, 5.0f32];
        for (yi, ei) in y.into_iter().zip(expected) {
            assert_eq!(yi, ei);
        }
    }
    #[test]
    fn test_dcopy() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut y = vec![0.0, 0.0, 0.0];
        dcopy(3, &x, 2, &mut y, 1);
        let expected = vec![1.0, 3.0, 5.0];
        for (yi, ei) in y.into_iter().zip(expected) {
            assert_eq!(yi, ei);
        }
    }
    #[test]
    fn test_ccopy() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(|x| Complex32::new(x, 0.0))
            .collect::<Vec<_>>();
        let mut y = vec![0.0, 0.0, 0.0]
            .into_iter()
            .map(|x| Complex32::new(x, 0.0))
            .collect::<Vec<_>>();
        ccopy(3, &x, 2, &mut y, 1);
        let expected = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(5.0, 0.0),
        ];
        for (yi, ei) in y.into_iter().zip(expected) {
            assert_eq!(yi, ei);
        }
    }
    #[test]
    fn test_zcopy() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(|x| Complex64::new(x, 0.0))
            .collect::<Vec<_>>();
        let mut y = vec![0.0, 0.0, 0.0]
            .into_iter()
            .map(|x| Complex64::new(x, 0.0))
            .collect::<Vec<_>>();
        zcopy(3, &x, 2, &mut y, 1);
        let expected = vec![1.0, 3.0, 5.0]
            .into_iter()
            .map(|x| Complex64::new(x, 0.0))
            .collect::<Vec<_>>();
        for (yi, ei) in y.into_iter().zip(expected) {
            assert_eq!(yi, ei);
        }
    }
    #[test]
    fn test_srotg() {
        const TOL: f32 = 1e-6;
        let a = 1.0;
        let b = -4.0;
        let (Givens { r, s, c }, z) = srotg(a, b);
        println!("{}, {}, {}, {}, {}, {}", a, b, r, z, c, s);
        assert!((c * a + s * b).abs_diff_eq(&r, TOL));
        assert!((c * b).abs_diff_eq(&(s * a), TOL));
    }
}
