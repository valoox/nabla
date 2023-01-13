//! Summation algorithms
use num::{Zero};
use std::ops::{Add, AddAssign, Sub};
use num::complex::{Complex32, Complex64};

/// Represents a current summation using the Kahan algorithm
///
/// # See Also
/// https://en.wikipedia.org/wiki/Kahan_summation_algorithm
#[derive(Clone, Copy, Debug)]
pub struct Kahan<T> {
    sum: T,
    comp: T,
}
impl<T> Kahan<T> {
    pub fn new(sum: T, comp: T) -> Self {
        Kahan { sum, comp }
    }
    pub fn total(self) -> T {
        self.sum
    }
}
impl<T> Add<Kahan<T>> for Kahan<T>
    where T: Copy + Add<T, Output=T> + Sub<T, Output=T>
{
    type Output = Kahan<T>;
    fn add(self, rhs: Kahan<T>) -> Self::Output {
        let mut out = self;
        out += rhs.sum;
        out += rhs.comp;
        out
    }
}
impl<T> Add<T> for Kahan<T>
    where T: Copy + Add<T, Output=T> + Sub<T, Output=T>
{
    type Output = Kahan<T>;
    fn add(self, rhs: T) -> Self::Output {
        let mut out = self;
        out += rhs;
        out
    }
}
impl<'a, T> Add<Kahan<T>> for &'a Kahan<T>
    where T: Copy + Add<T, Output=T> + Sub<T, Output=T>
{
    type Output = Kahan<T>;
    fn add(self, rhs: Kahan<T>) -> Self::Output {
        self + &rhs
    }
}
impl<'a, T> Add<T> for &'a Kahan<T>
    where T: Copy + Add<T, Output=T> + Sub<T, Output=T>
{
    type Output = Kahan<T>;
    fn add(self, rhs: T) -> Self::Output {
        self + &rhs
    }
}
impl<'a, T> Add<&'a Kahan<T>> for &'a Kahan<T>
    where T: Copy + Add<T, Output=T> + Sub<T, Output=T>{
    type Output = Kahan<T>;
    fn add(self, rhs: &'a Kahan<T>) -> Self::Output {
        let mut out = self.clone();
        out += rhs.sum;
        out += rhs.comp;
        out
    }
}
impl<'a, T> Add<&'a T> for &'a Kahan<T>
    where T: Copy + Add<T, Output=T> + Sub<T, Output=T>
{
    type Output = Kahan<T>;
    fn add(self, rhs: &'a T) -> Self::Output {
        let mut out = self.clone();
        out += rhs;
        out
    }
}
impl<T> Zero for Kahan<T>
where T: Zero + Copy + Sub<T, Output=T>
{
    fn zero() -> Self {
        Kahan::new(T::zero(), T::zero())
    }
    fn is_zero(&self) -> bool {
        self.sum.is_zero() && self.comp.is_zero()
    }
}
impl Default for Kahan<f32> {
    fn default() -> Self {
        Kahan::new(0.0, 0.0)
    }
}
impl Default for Kahan<f64> {
    fn default() -> Self {
        Kahan::new(0.0, 0.0)
    }
}
impl Kahan<f32> {
    pub fn sum<'a, I>(itr: I) -> f32
    where I: IntoIterator<Item=&'a f32>
    {
        let mut out = Self::zero();
        for x in itr {
            out += x;
        }
        out.sum
    }
}
impl Kahan<f64> {
    pub fn sum<'a, I>(itr: I) -> f64
        where I: IntoIterator<Item=&'a f64>
    {
        let mut out = Self::zero();
        for x in itr {
            out += x;
        }
        out.sum
    }
}
impl<T> AddAssign<T> for Kahan<T>
    where T: Copy + Add<T, Output=T> + Sub<T, Output=T>
{
    fn add_assign(&mut self, rhs: T) {
        let y = rhs - self.comp;
        let t = self.sum + y;
        self.comp = (t - self.sum) - y;
        self.sum = t;
    }
}
impl<'a, T> AddAssign<&'a T> for Kahan<T>
    where T: Copy + Add<T, Output=T> + Sub<T, Output=T>
{
    fn add_assign(&mut self, rhs: &'a T) {
        *self += *rhs;
    }
}
macro_rules! kahan_into {
    ($($T:ty),*) => {
        $(
            impl Into<$T> for Kahan<$T> {
                fn into(self) -> $T {
                    self.sum
                }
            }
        )*
    }
}
kahan_into!(f32, f64, Complex32, Complex64);

pub trait Summation<T>: Zero + AddAssign<T> + for<'a> AddAssign<&'a T> + Into<T> {
    fn sum<I: IntoIterator>(itr: I) -> T
    where Self: AddAssign<I::Item>,
    {
        let mut out = Self::zero();
        for x in itr {
            out += x;
        }
        out.into()
    }
}
impl Summation<u8> for u8 {}
impl Summation<u16> for u16 {}
impl Summation<u32> for u32 {}
impl Summation<u64> for u64 {}
impl Summation<u128> for u128 {}
impl Summation<i8> for i8 {}
impl Summation<i16> for i16 {}
impl Summation<i32> for i32 {}
impl Summation<i64> for i64 {}
impl Summation<i128> for i128 {}
impl Summation<f32> for f32 {}
impl Summation<f64> for f64 {}
impl Summation<f32> for Kahan<f32> {}
impl Summation<f64> for Kahan<f64> {}
impl Summation<Complex32> for Complex32 {}
impl Summation<Complex64> for Complex64 {}
impl Summation<Complex32> for Kahan<Complex32> {}
impl Summation<Complex64> for Kahan<Complex64> {}
