mod lvl1;
mod lvl2;
mod lvl3;

/// The layout of matrices
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    RowMajor,
    ColMajor,
}
/// The transpositions to apply (or not)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transpose {
    NoTrans,
    Trans,
    ConjTrans,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpLo {
    Upper,
    Lower,
}
impl UpLo {
    pub fn flip(&self) -> UpLo {
        match self {
            UpLo::Upper => UpLo::Lower,
            UpLo::Lower => UpLo::Upper,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Diag {
    /// If the matrix is unit triangular
    Unit,
    /// If the matrix is not unit triangular
    NonUnit,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// If the left matrix has a particular structure
    Left,
    /// If the right matrix has a particular structure
    Right,
}

pub use self::lvl1::{
    sasum, dasum, casum, zasum,
    saxpy, daxpy, caxpy, zaxpy,
    scopy, dcopy, ccopy, zcopy,
    sdot, sdsdot, dsdot, ddot,
    cdotc, cdotu, zdotc, zdotu,
    snrm2, dnrm2, scnrm2, dznrm2,
    srot, drot, crot, zrot,
    srotg, drotg, crotg, zrotg,
    sscal, dscal, cscal, zscal,
    sswap, dswap, cswap, zswap,
    isamax, idamax, icamax, izamax,
    isamin, idamin, icamin, izamin,
};
pub use self::lvl2::{
    sgbmv, dgbmv, cgbmv, zgbmv,
    sgemv, dgemv, cgemv, zgemv,
    sger, dger,
    cgerc, zgerc,
    cgeru, zgeru,
    chbmv, zhbmv,
    chemv, zhemv,
    cher, zher,
    cher2, zher2,
    chpmv, zhpmv,
    chpr, zhpr,
    chpr2, zhpr2,
    ssbmv, dsbmv,
    sspmv, dspmv,
    sspr, dspr,
    sspr2, dspr2,
    ssymv, dsymv,
    ssyr, dsyr,
    ssyr2, dsyr2,
    stbmv, dtbmv, ctbmv, ztbmv,
    stbsv, dtbsv, ctbsv, ztbsv,
    stpmv, dtpmv, ctpmv, ztpmv,
    stpsv, dtpsv, ctpsv, ztpsv,
    strmv, dtrmv, ctrmv, ztrmv,
    strsv, dtrsv, ctrsv, ztrsv,
};
pub use self::lvl3::{
    sgemm, dgemm, cgemm, zgemm,
    chemm, zhemm,
    cherk, zherk,
    cher2k, zher2k,
    ssymm, dsymm,
    ssyrk, dsyrk,
    ssyr2k, dsyr2k,
    strmm, dtrmm, ctrmm, ztrmm,
    strsm, dtrsm, ctrsm, ztrsm,
};
pub mod traits {
    pub use super::lvl1::{
        ASum, Axpy, Axpyc, Copyv, Dot, Dotc, Nrm2, Rot, Swap, IAopt
    };
}