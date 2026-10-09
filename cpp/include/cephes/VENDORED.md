# Cephes inverse beta implementation

Source: SciPy's `xsf` subproject, pinned by SciPy tag `v1.18.1`
(`e4e854eaa8f18d807cd3496028e257e36caa93cc`). The `xsf` subproject
commit is `f7b85f505a94fd398024478d2e747b01dd2d7a6d`:
<https://github.com/scipy/xsf/tree/f7b85f505a94fd398024478d2e747b01dd2d7a6d>.

The files under `xsf/` are copied unchanged from `include/xsf/`. They include
Cephes `incbi`, `incbet`, `ndtri`, gamma, and polynomial helpers and their
transitive headers. `LICENSE` is the upstream BSD-3 license;
`LICENSES_bundled.txt` preserves the upstream Cephes attribution. There are
no local patches to these files. `stdtr.c` is an unchanged, uncompiled
reference copy from SciPy `v1.10.1` (commit
`c1ed5ece8ffbf05356a22a8106affcd11bd3aee0`), when the Cephes `stdtri`
routine was still included. The small `stdtri` formula in our
`cpp/statistics.cpp` ports that routine and extends integer degrees of
freedom to positive real degrees of freedom; the inverse-beta dependencies
come from the newer SciPy/xsf copy above.

SciPy 1.18.1's public `stdtrit` ufunc calls Boost's `t_ppf_double`
(`scipy/special/functions.json`), rather than these Cephes routines. The
Student-t quantile tests compare their results with SciPy within a
relative-error bound.
