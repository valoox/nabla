//! Core algorithms for operations
use std::arch::x86_64;

#[cfg()]
pub fn fadd(x: &[f32], y: &[f32], n: usize) -> f32 {
    x86_64::_mm256_andnot_pd()
}