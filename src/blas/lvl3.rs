//! Level 3 operations: Matrix/Matrix
use super::{lvl1, lvl2};
use super::{Diag, Layout, Side, Transpose, UpLo};
use num::complex::{Complex32, Complex64, ComplexFloat};
use num::{Float, Zero};

/// C := alpha * op(A) * op(B) + beta * C
/// Where `C` is an `m` x `n` matrix, op(A) is an `m` x `k`, op(B) as a `k` x `n` matrix
fn real_gemm<T>(
    layout: Layout,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) where
    T: Float + lvl1::Scal + lvl2::Gemv,
{
    if m.is_zero() || n.is_zero() {
        return;
    }
    match (layout, transa, transb) {
        (Layout::ColMajor, Transpose::NoTrans, Transpose::NoTrans) => {
            debug_assert!(lda >= m.max(1));
            debug_assert!(a.len() >= lda * k);
            debug_assert!(ldb >= k.max(1));
            debug_assert!(b.len() >= ldb * n);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(n).for_each(|chunk| {
                        T::scal(m, beta, chunk, 1);
                    });
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .take(n)
                .enumerate()
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        Transpose::NoTrans,
                        m,
                        n,
                        alpha,
                        a,
                        lda,
                        &b[i * ldb..],
                        1,
                        beta,
                        chunk,
                        1,
                    );
                })
        },
        (Layout::ColMajor, Transpose::NoTrans, Transpose::Trans | Transpose::ConjTrans) => {
            debug_assert!(lda >= m.max(1));
            debug_assert!(a.len() >= lda * k);
            debug_assert!(ldb >= n.max(1));
            debug_assert!(b.len() >= ldb * k);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(n).for_each(|chunk| {
                        T::scal(m, beta, chunk, 1);
                    });
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .take(n)
                .enumerate()
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        Transpose::NoTrans,
                        m,
                        k,
                        alpha,
                        a,
                        lda,
                        &b[i..],
                        ldb,
                        beta,
                        chunk,
                        1,
                    );
                });
        },
        (Layout::ColMajor, Transpose::Trans | Transpose::ConjTrans, Transpose::NoTrans) => {
            debug_assert!(ldb >= k.max(1));
            debug_assert!(b.len() >= ldb * n);
            debug_assert!(lda >= k.max(1));
            debug_assert!(a.len() >= lda * m);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(n).for_each(|chunk| {
                        T::scal(m, beta, chunk, 1);
                    });
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .enumerate()
                .take(n)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        Transpose::Trans,
                        n,
                        m,
                        alpha,
                        a,
                        lda,
                        &b[i * ldb..],
                        1,
                        beta,
                        chunk,
                        1,
                    );
                })
        },
        (Layout::ColMajor, Transpose::Trans | Transpose::ConjTrans, Transpose::Trans | Transpose::ConjTrans) => {
            debug_assert!(lda >= k.max(1));
            debug_assert!(a.len() >= lda * m);
            debug_assert!(ldb >= n.max(1));
            debug_assert!(b.len() >= ldb * k);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(n).for_each(|chunk| {
                        T::scal(m, beta, chunk, 1);
                    });
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .enumerate()
                .take(n)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        Transpose::Trans,
                        n,
                        m,
                        alpha,
                        a,
                        lda,
                        &b[i..],
                        ldb,
                        beta,
                        chunk,
                        1,
                    );
                });
        },
        (Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans) => {
            debug_assert!(lda >= k.max(1));
            debug_assert!(a.len() >= lda * m);
            debug_assert!(ldb >= n.max(1));
            debug_assert!(b.len() >= ldb * k);
            debug_assert!(ldc >= n.max(1));
            debug_assert!(c.len() >= ldc * m);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(m).for_each(|chunk| {
                        T::scal(n, beta, chunk, 1);
                    });
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .take(m)
                .enumerate()
                .for_each(|(i, chunk)| {
                    T::gemv(
                        Layout::ColMajor,
                        Transpose::NoTrans,
                        n,
                        k,
                        alpha,
                        b,
                        ldb,
                        &a[i * lda..],
                        1,
                        beta,
                        chunk,
                        1,
                    );
                })
        },
        (Layout::RowMajor, Transpose::NoTrans, Transpose::Trans | Transpose::ConjTrans) => {
            debug_assert!(lda >= k.max(1));
            debug_assert!(a.len() >= lda * m);
            debug_assert!(ldb >= k.max(1));
            debug_assert!(b.len() >= ldb * n);
            debug_assert!(ldc >= n.max(1));
            debug_assert!(c.len() >= ldc * m);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(m).for_each(|chunk| {
                        T::scal(n, beta, chunk, 1);
                    });
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .enumerate()
                .take(m)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        Layout::ColMajor,
                        Transpose::Trans,
                        k,
                        n,
                        alpha,
                        b,
                        ldb,
                        &a[i * lda..],
                        1,
                        beta,
                        chunk,
                        1,
                    );
                })
        },
        (Layout::RowMajor, Transpose::Trans | Transpose::ConjTrans, Transpose::NoTrans) => {
            debug_assert!(lda >= m.max(1));
            debug_assert!(a.len() >= lda * k);
            debug_assert!(ldb >= n.max(1));
            debug_assert!(b.len() >= ldb * k);
            debug_assert!(ldc >= n.max(1));
            debug_assert!(c.len() >= ldc * m);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(m).for_each(|chunk| {
                        T::scal(n, beta, chunk, 1);
                    });
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .enumerate()
                .take(m)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        Layout::ColMajor,
                        transb,
                        n,
                        k,
                        alpha,
                        b,
                        ldb,
                        &a[i..],
                        lda,
                        beta,
                        chunk,
                        1,
                    );
                })
        },
        (Layout::RowMajor, Transpose::Trans | Transpose::ConjTrans, Transpose::Trans | Transpose::ConjTrans) => {
            debug_assert!(lda >= m.max(1));
            debug_assert!(a.len() >= lda * k);
            debug_assert!(ldb >= k.max(1));
            debug_assert!(b.len() >= ldb * n);
            debug_assert!(ldc >= n.max(1));
            debug_assert!(c.len() >= ldc * m);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(m).for_each(|chunk| {
                        T::scal(n, beta, chunk, 1);
                    });
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .enumerate()
                .take(m)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        Layout::ColMajor,
                        transb,
                        k,
                        n,
                        alpha,
                        b,
                        ldb,
                        &a[i..],
                        lda,
                        beta,
                        chunk,
                        1,
                    );
                })
        },
    }
}

pub fn sgemm(
    layout: Layout,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    real_gemm(
        layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    )
}
pub fn dgemm(
    layout: Layout,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    real_gemm(
        layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    )
}
fn complex_gemm<T>(
    layout: Layout,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) where
    T: ComplexFloat + lvl1::Scal + lvl2::Gemv,
{
    if m.is_zero() || n.is_zero() {
        return;
    }
    match (layout, transa, transb) {
        (Layout::ColMajor, Transpose::NoTrans, Transpose::NoTrans) => {
            debug_assert!(lda >= m.max(1));
            debug_assert!(a.len() >= lda * k);
            debug_assert!(ldb >= k.max(1));
            debug_assert!(b.len() >= ldb * n);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(n).for_each(|chunk| {
                        T::scal(m, beta, chunk, 1);
                    });
                }
                return;
            }
            c.chunks_mut(ldc)
                .enumerate()
                .take(n)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        transa,
                        m,
                        n,
                        alpha,
                        a,
                        lda,
                        &b[i * ldb..],
                        1,
                        beta,
                        chunk,
                        1,
                    );
                })
        },
        (Layout::ColMajor, Transpose::NoTrans, Transpose::Trans) => {
            debug_assert!(lda >= m.max(1));
            debug_assert!(a.len() >= lda * k);
            debug_assert!(ldb >= n.max(1));
            debug_assert!(b.len() >= ldb * k);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(n).for_each(|chunk| {
                        T::scal(m, beta, chunk, 1);
                    });
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .enumerate()
                .take(n)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        transa,
                        m,
                        n,
                        alpha,
                        a,
                        lda,
                        &b[i..],
                        ldb,
                        beta,
                        chunk,
                        1,
                    );
                });
        },
        (Layout::ColMajor, Transpose::NoTrans, Transpose::ConjTrans) => {
            unimplemented!("TODO!");
            debug_assert!(lda >= m.max(1));
            debug_assert!(a.len() >= lda * k);
            debug_assert!(ldb >= n.max(1));
            debug_assert!(b.len() >= ldb * k);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(n).for_each(|chunk| {
                        T::scal(m, beta, chunk, 1);
                    });
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .enumerate()
                .take(n)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        transa,
                        m,
                        n,
                        alpha,
                        a,
                        lda,
                        &b[i..],
                        ldb,
                        beta,
                        chunk,
                        1,
                    );
                });
        },
        (Layout::ColMajor, Transpose::Trans, Transpose::NoTrans) => {
            debug_assert!(ldb >= k.max(1));
            debug_assert!(b.len() >= ldb * n);
            debug_assert!(lda >= k.max(1));
            debug_assert!(a.len() >= lda * m);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(n).for_each(|chunk| {
                        T::scal(m, beta, chunk, 1);
                    });
                }
                return;
            }
            c.chunks_mut(ldc)
                .enumerate()
                .take(n)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        transa,
                        n,
                        m,
                        alpha,
                        a,
                        lda,
                        &b[i * ldb..],
                        1,
                        beta,
                        chunk,
                        1,
                    );
                })
        },
        (Layout::ColMajor, Transpose::ConjTrans, Transpose::NoTrans) => {
            unimplemented!("TODO");
            debug_assert!(ldb >= k.max(1));
            debug_assert!(b.len() >= ldb * n);
            debug_assert!(lda >= k.max(1));
            debug_assert!(a.len() >= lda * m);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(n).for_each(|chunk| {
                        T::scal(m, beta, chunk, 1);
                    });
                }
                return;
            }
            c.chunks_mut(ldc)
                .enumerate()
                .take(n)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        transa,
                        n,
                        m,
                        alpha,
                        a,
                        lda,
                        &b[i * ldb..],
                        1,
                        beta,
                        chunk,
                        1,
                    );
                })
        },
        (Layout::ColMajor, Transpose::Trans, Transpose::Trans) => {
            debug_assert!(lda >= k.max(1));
            debug_assert!(a.len() >= lda * m);
            debug_assert!(ldb >= n.max(1));
            debug_assert!(b.len() >= ldb * k);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(n).for_each(|chunk| {
                        T::scal(m, beta, chunk, 1);
                    });
                }
                return;
            }c
                .chunks_mut(ldc)
                .enumerate()
                .take(n)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        transa,
                        n,
                        m,
                        alpha,
                        a,
                        lda,
                        &b[i..],
                        ldb,
                        beta,
                        chunk,
                        1,
                    );
                });
        },
        (Layout::ColMajor, Transpose::ConjTrans, Transpose::Trans) => {
            unimplemented!("TODO");
            debug_assert!(lda >= k.max(1));
            debug_assert!(a.len() >= lda * m);
            debug_assert!(ldb >= n.max(1));
            debug_assert!(b.len() >= ldb * k);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(n).for_each(|chunk| {
                        T::scal(m, beta, chunk, 1);
                    });
                }
                return;
            }c
                .chunks_mut(ldc)
                .enumerate()
                .take(n)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        transa,
                        n,
                        m,
                        alpha,
                        a,
                        lda,
                        &b[i..],
                        ldb,
                        beta,
                        chunk,
                        1,
                    );
                });
        },
        (Layout::ColMajor, Transpose::Trans, Transpose::ConjTrans) => {
            unimplemented!("TODO");
            debug_assert!(lda >= k.max(1));
            debug_assert!(a.len() >= lda * m);
            debug_assert!(ldb >= n.max(1));
            debug_assert!(b.len() >= ldb * k);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(n).for_each(|chunk| {
                        T::scal(m, beta, chunk, 1);
                    });
                }
                return;
            }c
                .chunks_mut(ldc)
                .enumerate()
                .take(n)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        transa,
                        n,
                        m,
                        alpha,
                        a,
                        lda,
                        &b[i..],
                        ldb,
                        beta,
                        chunk,
                        1,
                    );
                });
        },
        (Layout::ColMajor, Transpose::ConjTrans, Transpose::ConjTrans) => {
            unimplemented!("TODO");
            debug_assert!(lda >= k.max(1));
            debug_assert!(a.len() >= lda * m);
            debug_assert!(ldb >= n.max(1));
            debug_assert!(b.len() >= ldb * k);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(n).for_each(|chunk| {
                        T::scal(m, beta, chunk, 1);
                    });
                }
                return;
            }c
                .chunks_mut(ldc)
                .enumerate()
                .take(n)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        transa,
                        n,
                        m,
                        alpha,
                        a,
                        lda,
                        &b[i..],
                        ldb,
                        beta,
                        chunk,
                        1,
                    );
                });
        },
        (Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans) => {
            debug_assert!(lda >= k.max(1));
            debug_assert!(a.len() >= lda * m);
            debug_assert!(ldb >= n.max(1));
            debug_assert!(b.len() >= ldb * k);
            debug_assert!(ldc >= n.max(1));
            debug_assert!(c.len() >= ldc * m);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(m).for_each(|chunk| {
                        T::scal(n, beta, chunk, 1);
                    });
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .enumerate()
                .take(m)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        Transpose::Trans,
                        k,
                        n,
                        alpha,
                        b,
                        ldb,
                        &a[i * lda..],
                        1,
                        beta,
                        chunk,
                        1,
                    );
                })
        },
        (Layout::RowMajor, Transpose::NoTrans, Transpose::Trans) => {
            debug_assert!(lda >= k.max(1));
            debug_assert!(a.len() >= lda * m);
            debug_assert!(ldb >= k.max(1));
            debug_assert!(b.len() >= ldb * n);
            debug_assert!(ldc >= n.max(1));
            debug_assert!(c.len() >= ldc * m);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(m).for_each(|chunk| {
                        T::scal(n, beta, chunk, 1);
                    });
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .enumerate()
                .take(m)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        Transpose::NoTrans,
                        n,
                        k,
                        alpha,
                        b,
                        ldb,
                        &a[i * lda..],
                        1,
                        beta,
                        chunk,
                        1,
                    );
                })
        },
        (Layout::RowMajor, Transpose::NoTrans, Transpose::ConjTrans) => {
            unimplemented!("TODO");
            debug_assert!(lda >= k.max(1));
            debug_assert!(a.len() >= lda * m);
            debug_assert!(ldb >= k.max(1));
            debug_assert!(b.len() >= ldb * n);
            debug_assert!(ldc >= n.max(1));
            debug_assert!(c.len() >= ldc * m);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(m).for_each(|chunk| {
                        T::scal(n, beta, chunk, 1);
                    });
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .enumerate()
                .take(m)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        Transpose::NoTrans,
                        n,
                        k,
                        alpha,
                        b,
                        ldb,
                        &a[i * lda..],
                        1,
                        beta,
                        chunk,
                        1,
                    );
                })
        },
        (Layout::RowMajor, Transpose::Trans, Transpose::NoTrans) => {
            debug_assert!(lda >= m.max(1));
            debug_assert!(a.len() >= lda * k);
            debug_assert!(ldb >= n.max(1));
            debug_assert!(b.len() >= ldb * k);
            debug_assert!(ldc >= n.max(1));
            debug_assert!(c.len() >= ldc * m);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(m).for_each(|chunk| {
                        T::scal(n, beta, chunk, 1);
                    });
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .enumerate()
                .take(m)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        Transpose::Trans,
                        k,
                        n,
                        alpha,
                        b,
                        ldb,
                        &a[i..],
                        lda,
                        beta,
                        chunk,
                        1,
                    );
                })
        },
        (Layout::RowMajor, Transpose::ConjTrans, Transpose::NoTrans) => {
            unimplemented!("TODO");
            debug_assert!(lda >= m.max(1));
            debug_assert!(a.len() >= lda * k);
            debug_assert!(ldb >= n.max(1));
            debug_assert!(b.len() >= ldb * k);
            debug_assert!(ldc >= n.max(1));
            debug_assert!(c.len() >= ldc * m);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(m).for_each(|chunk| {
                        T::scal(n, beta, chunk, 1);
                    });
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .enumerate()
                .take(m)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        Transpose::Trans,
                        k,
                        n,
                        alpha,
                        b,
                        ldb,
                        &a[i..],
                        lda,
                        beta,
                        chunk,
                        1,
                    );
                })
        },
        (Layout::RowMajor, Transpose::Trans, Transpose::Trans) => {
            debug_assert!(lda >= m.max(1));
            debug_assert!(a.len() >= lda * k);
            debug_assert!(ldb >= k.max(1));
            debug_assert!(b.len() >= ldb * n);
            debug_assert!(ldc >= n.max(1));
            debug_assert!(c.len() >= ldc * m);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(m).for_each(|chunk| {
                        T::scal(n, beta, chunk, 1);
                    });
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .enumerate()
                .take(m)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        Transpose::NoTrans,
                        n,
                        k,
                        alpha,
                        b,
                        ldb,
                        &a[i..],
                        lda,
                        beta,
                        chunk,
                        1,
                    );
                })
        },
        (Layout::RowMajor, Transpose::ConjTrans, Transpose::Trans) => {
            unimplemented!("TODO");
            debug_assert!(lda >= m.max(1));
            debug_assert!(a.len() >= lda * k);
            debug_assert!(ldb >= k.max(1));
            debug_assert!(b.len() >= ldb * n);
            debug_assert!(ldc >= n.max(1));
            debug_assert!(c.len() >= ldc * m);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(m).for_each(|chunk| {
                        T::scal(n, beta, chunk, 1);
                    });
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .enumerate()
                .take(m)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        Transpose::NoTrans,
                        n,
                        k,
                        alpha,
                        b,
                        ldb,
                        &a[i..],
                        lda,
                        beta,
                        chunk,
                        1,
                    );
                })
        },
        (Layout::RowMajor, Transpose::Trans, Transpose::ConjTrans) => {
            unimplemented!("TODO");
            debug_assert!(lda >= m.max(1));
            debug_assert!(a.len() >= lda * k);
            debug_assert!(ldb >= k.max(1));
            debug_assert!(b.len() >= ldb * n);
            debug_assert!(ldc >= n.max(1));
            debug_assert!(c.len() >= ldc * m);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(m).for_each(|chunk| {
                        T::scal(n, beta, chunk, 1);
                    });
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .enumerate()
                .take(m)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        Transpose::NoTrans,
                        n,
                        k,
                        alpha,
                        b,
                        ldb,
                        &a[i..],
                        lda,
                        beta,
                        chunk,
                        1,
                    );
                })
        },
        (Layout::RowMajor, Transpose::ConjTrans, Transpose::ConjTrans) => {
            unimplemented!("TODO");
            debug_assert!(lda >= m.max(1));
            debug_assert!(a.len() >= lda * k);
            debug_assert!(ldb >= k.max(1));
            debug_assert!(b.len() >= ldb * n);
            debug_assert!(ldc >= n.max(1));
            debug_assert!(c.len() >= ldc * m);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c.chunks_mut(ldc).take(m).for_each(|chunk| {
                        T::scal(n, beta, chunk, 1);
                    });
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .enumerate()
                .take(m)
                .for_each(|(i, chunk)| {
                    T::gemv(
                        layout,
                        Transpose::NoTrans,
                        n,
                        k,
                        alpha,
                        b,
                        ldb,
                        &a[i..],
                        lda,
                        beta,
                        chunk,
                        1,
                    );
                })
        },
    }
}

pub fn cgemm(
    layout: Layout,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    b: &[Complex32],
    ldb: usize,
    beta: Complex32,
    c: &mut [Complex32],
    ldc: usize,
) {
    complex_gemm(
        layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    )
}
pub fn zgemm(
    layout: Layout,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    b: &[Complex64],
    ldb: usize,
    beta: Complex64,
    c: &mut [Complex64],
    ldc: usize,
) {
    complex_gemm(
        layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    )
}
/// Matrix multiplication using a symmetrical matrix
///   C := alpha . A . B + beta . C     [Left]
///   C := alpha . B . A + beta . C     [Right]
/// Where `A` is a symmetric matrix, and `C` is an `m` x `n` matrix
fn symm<T>(
    layout: Layout,
    side: Side,
    uplo: UpLo,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) where
    T: Float + lvl1::Scal + lvl2::Symmetric,
{
    if m.is_zero() || n.is_zero() {
        return;
    }
    match (side, layout) {
        (Side::Left, Layout::ColMajor) => {
            debug_assert!(lda >= n.max(1));
            debug_assert!(a.len() >= lda * m);
            debug_assert!(ldb >= m.max(1));
            debug_assert!(b.len() >= ldb * n);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c
                        .chunks_mut(ldc)
                        .take(n)
                        .for_each(|chunk| T::scal(m, beta, chunk, 1));
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .take(n)
                .enumerate()
                .for_each(|(i, chunk)| {
                    T::symv(
                        layout,
                        uplo,
                        m,
                        alpha,
                        a,
                        lda,
                        &b[i * ldb..],
                        1,
                        beta,
                        chunk,
                        1,
                    );
                });
        },
        (Side::Left, Layout::RowMajor) => {
            debug_assert!(lda >= n.max(1));
            debug_assert!(a.len() >= lda * m);
            debug_assert!(ldb >= n.max(1));
            debug_assert!(b.len() >= ldb * m);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c
                        .chunks_mut(ldc)
                        .take(m)
                        .for_each(|chunk| T::scal(n, beta, chunk, 1));
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .take(m)
                .enumerate()
                .for_each(|(i, chunk)| {
                    T::symv(
                        Layout::ColMajor,
                        uplo.flip(),
                        n,
                        alpha,
                        a,
                        lda,
                        &b[i * ldb..],
                        1,
                        beta,
                        chunk,
                        1,
                    );
                });
        },
        (Side::Right, Layout::ColMajor) => {
            debug_assert!(lda >= m.max(1));
            debug_assert!(a.len() >= lda * n);
            debug_assert!(ldb >= m.max(1));
            debug_assert!(b.len() >= ldb * n);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c
                        .chunks_mut(ldc)
                        .take(n)
                        .for_each(|chunk| T::scal(m, beta, chunk, 1));
                }
                return;
            }
            for i in 0..m {
                T::symv(
                    Layout::RowMajor,
                    uplo.flip(),
                    n,
                    alpha,
                    a,
                    lda,
                    &b[i..],
                    ldb,
                    beta,
                    &mut c[i..],
                    ldc,
                );
            }
        },
        (Side::Right, Layout::RowMajor) => {
            debug_assert!(lda >= m.max(1));
            debug_assert!(a.len() >= lda * n);
            debug_assert!(ldb >= n.max(1));
            debug_assert!(b.len() >= ldb * m);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c
                        .chunks_mut(ldc)
                        .take(m)
                        .for_each(|chunk| T::scal(n, beta, chunk, 1));
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .take(m)
                .enumerate()
                .for_each(|(i, chunk)| {
                    T::symv(
                        Layout::ColMajor,
                        uplo.flip(),
                        n,
                        alpha,
                        a,
                        lda,
                        &b[i * ldb..],
                        1,
                        beta,
                        chunk,
                        1,
                    );
                });
        },
    }
}

pub fn ssymm(
    layout: Layout,
    side: Side,
    uplo: UpLo,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    symm(
        layout, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
    )
}
pub fn dsymm(
    layout: Layout,
    side: Side,
    uplo: UpLo,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    symm(
        layout, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
    )
}
fn hemm<T>(
    layout: Layout,
    side: Side,
    uplo: UpLo,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) where
    T: ComplexFloat + lvl1::Scal + lvl2::Hermitian,
{
    if m.is_zero() || n.is_zero() {
        return;
    }
    match (side, layout) {
        (Side::Left, Layout::ColMajor) => {
            debug_assert!(lda >= n.max(1));
            debug_assert!(a.len() >= lda * m);
            debug_assert!(ldb >= m.max(1));
            debug_assert!(b.len() >= ldb * n);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c
                        .chunks_mut(ldc)
                        .take(n)
                        .for_each(|chunk| T::scal(m, beta, chunk, 1));
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .take(n)
                .enumerate()
                .for_each(|(i, chunk)| {
                    T::hemv(
                        layout,
                        uplo,
                        m,
                        alpha,
                        a,
                        lda,
                        &b[i * ldb..],
                        1,
                        beta,
                        chunk,
                        1,
                    );
                });
        },
        (Side::Left, Layout::RowMajor) => {
            debug_assert!(lda >= n.max(1));
            debug_assert!(a.len() >= lda * m);
            debug_assert!(ldb >= n.max(1));
            debug_assert!(b.len() >= ldb * m);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c
                        .chunks_mut(ldc)
                        .take(m)
                        .for_each(|chunk| T::scal(n, beta, chunk, 1));
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .take(m)
                .enumerate()
                .for_each(|(i, chunk)| {
                    T::hemv(
                        Layout::ColMajor,
                        uplo.flip(),
                        n,
                        alpha,
                        a,
                        lda,
                        &b[i * ldb..],
                        1,
                        beta,
                        chunk,
                        1,
                    );
                });
        },
        (Side::Right, Layout::ColMajor) => {
            debug_assert!(lda >= m.max(1));
            debug_assert!(a.len() >= lda * n);
            debug_assert!(ldb >= m.max(1));
            debug_assert!(b.len() >= ldb * n);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c
                        .chunks_mut(ldc)
                        .take(n)
                        .for_each(|chunk| T::scal(m, beta, chunk, 1));
                }
                return;
            }
            for i in 0..m {
                T::hemv(
                    Layout::RowMajor,
                    uplo.flip(),
                    n,
                    alpha,
                    a,
                    lda,
                    &b[i..],
                    ldb,
                    beta,
                    &mut c[i..],
                    ldc,
                );
            }
        },
        (Side::Right, Layout::RowMajor) => {
            debug_assert!(lda >= m.max(1));
            debug_assert!(a.len() >= lda * n);
            debug_assert!(ldb >= n.max(1));
            debug_assert!(b.len() >= ldb * m);
            debug_assert!(ldc >= m.max(1));
            debug_assert!(c.len() >= ldc * n);
            if alpha.is_zero() {
                if !beta.is_one() {
                    c
                        .chunks_mut(ldc)
                        .take(m)
                        .for_each(|chunk| T::scal(n, beta, chunk, 1));
                }
                return;
            }
            c
                .chunks_mut(ldc)
                .take(m)
                .enumerate()
                .for_each(|(i, chunk)| {
                    T::hemv(
                        Layout::ColMajor,
                        uplo.flip(),
                        n,
                        alpha,
                        a,
                        lda,
                        &b[i * ldb..],
                        1,
                        beta,
                        chunk,
                        1,
                    );
                });
        },
    }
}

pub fn chemm(
    layout: Layout,
    side: Side,
    uplo: UpLo,
    m: usize,
    n: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    b: &[Complex32],
    ldb: usize,
    beta: Complex32,
    c: &mut [Complex32],
    ldc: usize,
) {
    hemm(
        layout, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
    )
}
pub fn zhemm(
    layout: Layout,
    side: Side,
    uplo: UpLo,
    m: usize,
    n: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    b: &[Complex64],
    ldb: usize,
    beta: Complex64,
    c: &mut [Complex64],
    ldc: usize,
) {
    hemm(
        layout, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
    )
}
/// Performs a symmetric rank-k update
///  C := alpha . A . A' + beta . C  [NoTrans]
///  C := alpha . A' . A + beta . C  [Trans]
///
/// Where `C` is an `n`x`n` matrix and `A` is either a `n`x`k` or `k`x`n` matrix
pub fn syrk<T>(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) where
    T: Float + lvl1::Scal + lvl2::Symmetric,
{
    if n.is_zero() {
        return;
    }
    if !beta.is_one() {
        match (layout, uplo) {
            (Layout::RowMajor, UpLo::Lower) | (Layout::ColMajor, UpLo::Upper) => {
                for i in 0..n {
                    T::scal(i + 1, beta, &mut c[i * ldc..], 1);
                }
            },
            (Layout::RowMajor, UpLo::Upper) | (Layout::ColMajor, UpLo::Lower) => {
                for i in 0..n {
                    T::scal(n - i, beta, &mut c[i * ldc + i..], 1);
                }
            },
        }
    }
    if k.is_zero() {
        return;
    }
    match (layout, trans) {
        (Layout::RowMajor, Transpose::NoTrans) | (Layout::ColMajor, Transpose::Trans | Transpose::ConjTrans)=> {
            for i in 0..k {
                T::syr(
                    layout,
                    uplo,
                    n,
                    alpha,
                    &a[i..],
                    lda,
                    c,
                    ldc,
                );
                for j in 0..i {
                    T::syr2(
                        layout,
                        uplo,
                        n,
                        alpha,
                        &a[i..],
                        lda,
                        &a[j..],
                        lda,
                        c,
                        ldc,
                    );
                }
            }
        },
        (Layout::ColMajor, Transpose::NoTrans) | (Layout::RowMajor, Transpose::Trans | Transpose::ConjTrans) => {
            for i in 0..k {
                T::syr(
                    layout,
                    uplo,
                    n,
                    alpha,
                    &a[i * lda..],
                    1,
                    c,
                    ldc,
                );
                for j in 0..i {
                    T::syr2(
                        layout,
                        uplo,
                        n,
                        alpha,
                        &a[i * lda..],
                        1,
                        &a[j * lda..],
                        1,
                        c,
                        ldc,
                    );
                }
            }
        },
    }
}
pub fn ssyrk(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    syrk(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}
pub fn dsyrk(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    syrk(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}
/// Performs a symmetric rank 2k update
///  C := alpha . A . B' + alpha . B . A' + beta . C [NoTrans]
///  C := alpha . A' . B + alpha . B' . A + beta . C [Trans]
pub fn syr2k<T>(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) where
    T: ComplexFloat,
{
    unimplemented!("TODO");
}
pub fn ssyr2k(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    syr2k(
        layout, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    );
}
pub fn dsyr2k(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    syr2k(
        layout, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    );
}
/// Performs a hermitian rank-k update
///  C := alpha . A . Ah + beta . C  [NoTrans]
///  C := alpha . Ah . A + beta . C  [ConjTrans]
pub fn herk<T>(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) where
    T: ComplexFloat,
{
    unimplemented!("TODO");
}
pub fn cherk(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    beta: Complex32,
    c: &mut [Complex32],
    ldc: usize,
) {
    herk(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}
pub fn zherk(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    beta: Complex64,
    c: &mut [Complex64],
    ldc: usize,
) {
    herk(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}
/// Performs a hermitian rank 2k update
///  C := alpha . A . Bh + conj(alpha) . B . Ah + beta . C [NoTrans]
///  C := alpha . Ah . B + conj(alpha) . Bh . A + beta . C [ConjTrans]
pub fn her2k<T>(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) where
    T: ComplexFloat,
{
    unimplemented!("TODO");
}
pub fn cher2k(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    b: &[Complex32],
    ldb: usize,
    beta: Complex32,
    c: &mut [Complex32],
    ldc: usize,
) {
    her2k(
        layout, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    );
}
pub fn zher2k(
    layout: Layout,
    uplo: UpLo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    b: &[Complex64],
    ldb: usize,
    beta: Complex64,
    c: &mut [Complex64],
    ldc: usize,
) {
    her2k(
        layout, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    );
}
/// Computes a matrix-matrix product where one of the input matrices is triangular
///  B := alpha . A . B    [NoTrans]
///  B := alpha . A' . B   [Trans]
pub fn real_trmm<T>(
    layout: Layout,
    side: Side,
    uplo: UpLo,
    transa: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
) where
    T: Float,
{
    unimplemented!("TODO")
}
pub fn strmm(
    layout: Layout,
    side: Side,
    uplo: UpLo,
    transa: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &mut [f32],
    ldb: usize,
) {
    real_trmm(
        layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb,
    );
}
pub fn dtrmm(
    layout: Layout,
    side: Side,
    uplo: UpLo,
    transa: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &mut [f64],
    ldb: usize,
) {
    real_trmm(
        layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb,
    );
}
/// Computes a matrix-matrix product where one of the input matrices is triangular
///  B := alpha . A . B    [NoTrans]
///  B := alpha . A' . B   [Trans]
///  B := alpha . conj(A') . B   [ConjTrans]
pub fn complex_trmm<T>(
    layout: Layout,
    side: Side,
    uplo: UpLo,
    transa: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
) where
    T: ComplexFloat,
{
    unimplemented!("TODO")
}
pub fn ctrmm(
    layout: Layout,
    side: Side,
    uplo: UpLo,
    transa: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    b: &mut [Complex32],
    ldb: usize,
) {
    complex_trmm(
        layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb,
    );
}
pub fn ztrmm(
    layout: Layout,
    side: Side,
    uplo: UpLo,
    transa: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    b: &mut [Complex64],
    ldb: usize,
) {
    complex_trmm(
        layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb,
    );
}
/// Solves a triangular matrix equation
///  A . X = alpha . B     [NoTrans]
///  A' . X = alpha . B    [Trans]
pub fn real_trsm<T>(
    layout: Layout,
    side: Side,
    uplo: UpLo,
    transa: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
) where
    T: Float,
{
    unimplemented!("TODO")
}
pub fn strsm(
    layout: Layout,
    side: Side,
    uplo: UpLo,
    transa: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &mut [f32],
    ldb: usize,
) {
    real_trsm(
        layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb,
    );
}
pub fn dtrsm(
    layout: Layout,
    side: Side,
    uplo: UpLo,
    transa: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &mut [f64],
    ldb: usize,
) {
    real_trsm(
        layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb,
    );
}
/// Solves a triangular matrix equation
///  A . X = alpha . B          [NoTrans]
///  A' . X = alpha . B         [Trans]
///  conj(A') . X = alpha . B   [ConjTrans]
pub fn complex_trsm<T>(
    layout: Layout,
    side: Side,
    uplo: UpLo,
    transa: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
) where
    T: ComplexFloat,
{
    unimplemented!("TODO")
}
pub fn ctrsm(
    layout: Layout,
    side: Side,
    uplo: UpLo,
    transa: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    b: &mut [Complex32],
    ldb: usize,
) {
    complex_trsm(
        layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb,
    );
}
pub fn ztrsm(
    layout: Layout,
    side: Side,
    uplo: UpLo,
    transa: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    b: &mut [Complex64],
    ldb: usize,
) {
    complex_trsm(
        layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb,
    );
}

#[cfg(test)]
mod tests {
    use crate::blas::{dgemm, dsymm, dsyrk, Layout, Side, Transpose, UpLo};
    use approx::AbsDiffEq;

    #[test]
    fn test_gemm() {
        let a = vec![1.0, 2.0, 3.0, 4.0, -1.0, 1.0, -2.0, 1.0, 0.0, 2.0, 1.0, 0.0];
        let b = vec![1.0, 2.0, 3.0, -1.0, -1.0, 3.0, 0.0, 1.0];
        let mut c1 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        dgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            3,
            2,
            4,
            1.0,
            &a[..],
            4,
            &b[..],
            2,
            0.0,
            &mut c1[..],
            2,
        );
        assert!(
            c1.abs_diff_eq(&vec![4.0, 13.0, 4.0, -8.0, 5.0, 1.0,], 1e-21),
            "{:?} != {:?}",
            c1,
            vec![4.0, 13.0, 4.0, -8.0, 5.0, 1.0,],
        );
        let mut c2 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        dgemm(
            Layout::RowMajor,
            Transpose::Trans,
            Transpose::Trans,
            2,
            3,
            4,
            1.0,
            &b[..],
            2,
            &a[..],
            4,
            0.0,
            &mut c2[..],
            3,
        );
        assert!(
            c2.abs_diff_eq(&vec![4.0, 4.0, 5.0, 13.0, -8.0, 1.0,], 1e-21),
            "{:?} != {:?}",
            c2,
            vec![4.0, 4.0, 5.0, 13.0, -8.0, 1.0,],
        );
        let mut c3 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        dgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::Trans,
            3,
            2,
            4,
            1.0,
            &a[..],
            4,
            &b[..],
            4,
            0.0,
            &mut c3[..],
            2,
        );
        assert!(
            c3.abs_diff_eq(&vec![10.0, 9.0, -6.0, 5.0, 7.0, 6.0], 1e-21),
            "{:?} != {:?}",
            c3,
            vec![10.0, 9.0, -6.0, 5.0, 7.0, 6.0],
        );
        let mut c4 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        dgemm(
            Layout::ColMajor,
            Transpose::NoTrans,
            Transpose::Trans,
            3,
            2,
            4,
            1.0,
            &a[..],
            3,
            &b[..],
            2,
            0.0,
            &mut c4[..],
            3,
        );
        assert!(
            c4.abs_diff_eq(&vec![15.0, -2.0, 6.0, -6.0, 9.0, 5.0], 1e-21),
            "{:?} != {:?}",
            c4,
            vec![15.0, -2.0, 6.0, -6.0, 9.0, 5.0],
        );
    }
    #[test]
    fn test_symm() {
        let s = vec![
             1.0,  2.0, -3.0,  4.0,
             2.0,  1.0,  5.0, -1.0,
            -3.0,  5.0,  1.0,  7.0,
             4.0, -1.0,  7.0,  1.0,
        ];
        let b = vec![
             2.0,  3.0,  5.0, f64::NAN, f64::NAN,
            -1.0,  2.0, -3.0, f64::NAN, f64::NAN,
             2.0, -1.0,  5.0, f64::NAN, f64::NAN,
             f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN,
             f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN,
        ];
        let mut c1 = Vec::with_capacity(9);
        let mut c2 = Vec::with_capacity(9);
        for _ in 0..9 {
            c1.push(0.0);
            c2.push(0.0);
        }
        for layout in vec![Layout::ColMajor, Layout::RowMajor] {
            for uplo in vec![UpLo::Lower, UpLo::Upper] {
                dgemm(
                    layout,
                    Transpose::NoTrans,
                    Transpose::NoTrans,
                    3,
                    3,
                    3,
                    1.0,
                    &s[..],
                    4,
                    &b[..],
                    5,
                    0.0,
                    &mut c1[..],
                    3,
                );
                dsymm(
                    layout,
                    Side::Left,
                    uplo,
                    3,
                    3,
                    1.0,
                    &s[..],
                    4,
                    &b[..],
                    5,
                    0.0,
                    &mut c2[..],
                    3,
                );
                c1.fill(0.0);
                c2.fill(0.0);
                assert!(c1.abs_diff_eq(&c2, 1e-21), "{:?} != {:?}", c1, c2);
                dgemm(
                    layout,
                    Transpose::NoTrans,
                    Transpose::NoTrans,
                    3,
                    3,
                    3,
                    1.0,
                    &b[..],
                    5,
                    &s[..],
                    4,
                    0.0,
                    &mut c1[..],
                    3,
                );
                dsymm(
                    layout,
                    Side::Right,
                    uplo,
                    3,
                    3,
                    1.0,
                    &s[..],
                    4,
                    &b[..],
                    5,
                    0.0,
                    &mut c2[..],
                    3,
                );
                c1.fill(0.0);
                c2.fill(0.0);
                assert!(c1.abs_diff_eq(&c2, 1e-21), "{:?} != {:?}", c1, c2);
            }
        }
    }
    #[test]
    fn test_syrk() {
        let a = vec![
             1.0,  2.0, -1.0,
            -7.0, -1.0,  2.0,
             2.0,  1.0,  3.0,
        ];
        let mut c1 = Vec::with_capacity(9);
        let mut c2 = Vec::with_capacity(9);
        for _ in 0..9 {
            c1.push(0.0);
            c2.push(0.0);
        }
        for layout in vec![Layout::RowMajor, Layout::ColMajor] {
            for uplo in vec![UpLo::Lower, UpLo::Upper] {
                for trans in vec![Transpose::NoTrans, Transpose::Trans] {
                    dsyrk(
                        layout,
                        uplo,
                        trans,
                        3,
                        3,
                        1.0,
                        &a[..],
                        3,
                        1.0,
                        &mut c1[..],
                        3,
                    );
                    dgemm(
                        layout,
                        trans,
                        match trans {
                            Transpose::Trans | Transpose::ConjTrans => Transpose::NoTrans,
                            Transpose::NoTrans => Transpose::Trans,
                        },
                        3,
                        3,
                        3,
                        1.0,
                        &a[..],
                        3,
                        &a[..],
                        3,
                        1.0,
                        &mut c2[..],
                        3,
                    );
                    assert!(
                        c1.abs_diff_eq(&c2, 1e-21),
                        "{:?} - {:?} - {:?} - {:?} != {:?}",
                        layout, uplo, trans, c1, c2
                    );
                }
            }
        }
    }
}
