# Local changes to fast_float

Upstream: fast_float, <https://github.com/fastfloat/fast_float>, tag `v8.2.3`
(`FASTFLOAT_VERSION 8.2.3` in `float_common.h`). Licence: MIT (chosen from
MIT OR Apache-2.0 OR BSL-1.0; `LICENSE-MIT`).

Subset: only the headers under `include/fast_float/` and `LICENSE-MIT` are
vendored. `LICENSE-MIT` has the upstream text with different line wrapping,
and each header ends with one extra blank line.

## Changes

- `float_common.h`: removed the trailing `return true;` from
  `fastfloat_strncasecmp3` and `fastfloat_strncasecmp5`. Every branch above
  it already returns, so the statement is unreachable; removing it silences
  unreachable-code warnings. No behaviour change.
