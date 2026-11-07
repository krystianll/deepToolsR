# Local changes to htslib

Upstream: htslib by Genome Research Ltd, <https://github.com/samtools/htslib>,
release `1.24`. Licences: MIT (Expat) and, for `cram/`, BSD-3-Clause
(`LICENSES/htslib-MIT-and-BSD-3-Clause.txt`).

Subset: only the library sources deepToolsR builds are vendored. Every other
file is unmodified.

## Changes

- `cram/cram_io.c` (Windows only, `#if defined(_WIN32)`, marked `deepToolsR:`):
  `PATH_MAX` is redefined to 4096 after the file's own `PATH_MAX` fallback.
  MinGW's `PATH_MAX` is `MAX_PATH` (260), so the reference FASTA and `.fai`
  path buffers in this file truncated paths longer than 259 characters even
  when Windows long paths are enabled, and a CRAM's `--referenceFasta` could
  not be loaded. All `PATH_MAX` buffers involved are local to `cram_io.c`;
  no header or structure uses it.
