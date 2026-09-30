# Licences

deepToolsR is derived from deepTools and is released under the MIT licence
(`LICENSE.txt`). It bundles or adapts the third-party components below; each
licence text is in this folder, and the vendored trees keep their own copies.

| Component | Path in repo | Licence (SPDX) | Copyright holder(s) | Licence file | Upstream version and local changes |
|---|---|---|---|---|---|
| deepTools (base of deepToolsR) | `deeptoolsr/`, `tests/` (test data and ported tests) | MIT | Max Planck Institute for Immunobiology and Epigenetics, 2019 | `deepTools-MIT.txt` (= `../LICENSE.txt`) | derived from deepTools 3.5.6 |
| deepToolsR changes (C++ core, R-suffix tools, etc.) | whole repo | MIT | Krystian Łazowski, 2026 | `deepTools-MIT.txt` (pointer line at the end of `../LICENSE.txt`) | n/a (own code) |
| seaborn colormaps | `deeptoolsr/cm.py` | BSD-3-Clause | Michael L. Waskom, 2012-2019 | `seaborn-BSD-3-Clause.txt` | colormap data copied into `cm.py` |
| htslib (excluding `cram/`) | `cpp/htslib/` | MIT (Expat) | Genome Research Ltd, 2012-2026 (plus other authors named in file headers) | `htslib-MIT-and-BSD-3-Clause.txt` | 1.24 release tarball; subset of upstream files, unmodified |
| htslib `cram/` | `cpp/htslib/cram/` | BSD-3-Clause | Genome Research Ltd, 2012-2026 | `htslib-MIT-and-BSD-3-Clause.txt` | 1.24; unmodified |
| htscodecs (all files except the two below) | `cpp/htslib/htscodecs/` | BSD-3-Clause | Genome Research Ltd | `htscodecs-BSD-3-Clause.md` | 1.6.7 (bundled in htslib 1.24); subset of upstream files, unmodified |
| htscodecs `c_range_coder.h` | `cpp/htslib/htscodecs/` | public domain (derived from work by Eugene Shelwien) | Eugene Shelwien | `htscodecs-BSD-3-Clause.md` | 1.6.7; unmodified |
| htscodecs `rANS_byte.h`, `rANS_word.h` | `cpp/htslib/htscodecs/` | CC0-1.0 / public domain (derived from ryg_rans) | Fabian Giesen (waived) | `htscodecs-BSD-3-Clause.md` | 1.6.7; unmodified |
| libdeflate | `cpp/libdeflate/` | MIT | Eric Biggers, 2016; Google LLC, 2024 | `libdeflate-MIT.txt` | v1.22; subset of upstream files, unmodified |
| libBigWig | `cpp/libBigWig/` | MIT | Devon Ryan, 2015 | `libBigWig-MIT.txt` | 0.4.8; modified, see `cpp/libBigWig/CHANGES_deepToolsR.md` |
| `bwStrdup` in libBigWig (`bwRead.c`) | `cpp/libBigWig/bwRead.c` | MIT (taken from musl `strdup.c`) | Rich Felker et al., 2005-2020 | `musl-MIT.txt` | part of libBigWig 0.4.8; unmodified |
| fastcluster (Ward core) | `cpp/include/fastcluster/` | BSD-2-Clause | Daniel Müllner, 2011; Google Inc. (changes from 1.1.24) | `fastcluster-BSD-2-Clause.txt` | v1.3.0; subset of upstream files (`fastcluster.cpp` only), unmodified |
| fast_float | `cpp/include/fast_float/` | MIT (upstream is MIT OR Apache-2.0 OR BSL-1.0; MIT chosen) | The fast_float authors, 2021 | `fast_float-MIT.txt` | v8.2.3; modified, see `cpp/include/fast_float/CHANGES_deepToolsR.md` |
| SciPy xsf / Cephes (special functions, incomplete beta) | `cpp/include/cephes/` | BSD-3-Clause | SciPy developers, 2024; Stephen L. Moshier (Cephes, 1984-2000) | `scipy-xsf-BSD-3-Clause.txt`, `scipy-xsf-bundled-licenses.txt` | xsf commit f7b85f5 (pinned by SciPy v1.18.1); subset of upstream files, unmodified |
| xsf bundled: Cephes | `cpp/include/cephes/xsf/cephes/` | BSD-3-Clause | Stephen L. Moshier | `scipy-xsf-bundled-licenses.txt` | as xsf; subset, unmodified |
| xsf bundled: Faddeeva | `cpp/include/cephes/xsf/faddeeva.h` | MIT | see bundled list | `scipy-xsf-bundled-licenses.txt` | as xsf; unmodified |
| xsf bundled: qd | `cpp/include/cephes/xsf/cephes/dd_real.h` | BSD-3-Clause-LBNL | see bundled list | `scipy-xsf-bundled-licenses.txt` | as xsf; unmodified |
| xsf bundled: mdspan | `cpp/include/cephes/xsf/third_party/kokkos/mdspan.hpp` | Apache-2.0 WITH LLVM-exception | see bundled list | `scipy-xsf-bundled-licenses.txt` | as xsf; unmodified |
| `stdtr.c` (Cephes Student t) | `cpp/include/cephes/stdtr.c` | BSD-3-Clause (Cephes) | Stephen L. Moshier, 1984-1995 | `scipy-xsf-bundled-licenses.txt` (Cephes entry) | SciPy v1.10.1; unmodified (reference copy, not compiled) |
| digestible (t-digest) | `cpp/include/digestible/digestible.h` | Apache-2.0 | Ted Dunning and contributors (t-digest); digestible authors | `Apache-2.0.txt`, `digestible-NOTICE.txt` | commit 93b870d (no release tags); modified, see `cpp/include/digestible/CHANGES_deepToolsR.md` |
| stb_image_resize2 v2.17 | `cpp/include/stb/stb_image_resize2.h` | MIT (upstream: MIT or public domain; MIT chosen) | Sean Barrett, 2017 | `stb-MIT.txt` | v2.17 (commit cf79a48); unmodified |

"Subset of upstream files" means build files, docs, tests and tools that
deepToolsR does not use were left out; the files that are included match
the named upstream version. Each `CHANGES_deepToolsR.md` lists the local
changes to a modified component.

Ring buffer in `cpp/compute_matrix_io.cpp` is adapted from Krystian Łazowski's own
Metaplotter project; no separate licence applies. Behaviour-matching
reimplementations (SciPy fcluster numbering, NumPy quantile/linspace,
Matplotlib resampling) are not copied code.

## Python dependencies (not bundled)

Installed by pip from `pyproject.toml`, each under its own licence:
numpy, matplotlib, deeptoolsintervals, uniseg. Optional/test extras
(deepTools 3.5.6, pysam, pyBigWig, scipy, pytest, ...) are likewise not
distributed here.

The prebuilt Windows `libhts.a` archives are not part of the public release.
