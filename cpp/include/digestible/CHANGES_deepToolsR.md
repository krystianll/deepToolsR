# Local changes to digestible

Upstream: digestible (t-digest), <https://github.com/SpirentOrion/digestible>,
file `include/digestible/digestible.h` at commit
`93b870d737e8a18a36cda3823d6eb237bdf20215` (latest commit to the file; the
project has no release tags). Licence: Apache-2.0 (see
`LICENSES/Apache-2.0.txt` and `LICENSES/digestible-NOTICE.txt`).

Subset: only `digestible.h` is vendored.

## Changes

- `digestible.h`: in `operator<` for `centroid`, the comparison
  `lhs.mean < rhs.mean` is written with parenthesised operands,
  `(lhs.mean) < (rhs.mean)`. Formatting only; no behaviour change.
- `digestible.h`: a header comment marks the file as modified (Apache-2.0
  section 4(b)).
