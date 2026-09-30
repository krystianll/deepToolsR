# Local changes to libBigWig

Upstream: libBigWig by Devon Ryan, <https://github.com/dpryan79/libBigWig>,
tag `0.4.8` (`LIBBIGWIG_VERSION 0.4.8` in `bigWig.h`). Licence: MIT (`LICENSE`).

Subset: only the library sources and `LICENSE` are vendored; the build files,
docs, tests and CI files are omitted. deepToolsR builds it with `NOCURL`
(local files only).

## Changes

- `bwWrite.c`, `bwRead.c`, `bigWig.h`: data blocks are compressed with
  libdeflate's zlib encoder instead of zlib `compress()` (2-3x faster, same
  zlib stream format). `bwWriteBuffer_t` gains `compressLevel` (default
  `Z_DEFAULT_COMPRESSION`, mapped to level 6; clamped to 12) and a lazily
  allocated `compressor`, freed in `bwDestroyWriteBuffer`. The compression
  buffer is sized to the larger of zlib's and libdeflate's bounds.
- `bwRead.c`, `io.c`, `bigWig.h`: new `bwCloseChecked()` / `urlCloseChecked()`
  return a nonzero status when finalising or `fclose` fails, so truncated
  output is reported to the caller; `bwClose()`/`urlClose()` wrap them.
- `bwValues.c`, `bigWig.h`: new `bwGetOverlappingIntervalsChecked()` tells an
  empty result apart from a failed read; `bwGetOverlappingIntervals()` wraps it.
- `bwRead.c`, `bwValues.c`, `bwValues.h`, `bwCommon.h`: hardening against
  malformed or corrupt files: bounded index-tree depth (`BW_MAX_TREE_DEPTH`)
  and rejection of ancestor cycles, child/item counts checked against the
  block size, chromosome-list sizes checked against the file size, bounds
  checks on decompressed block sizes and item counts, safe `realloc` use, and
  a leak fix in `pushIntervals` failure handling. Malformed R-tree headers
  are tolerated by skipping (not stopping at) out-of-range children, and
  over-allocation in block merging is removed.
- `bwWrite.c`: zoom-level construction validates intervals and iterator state
  instead of dereferencing NULL or out-of-range data.
- `bwRead.c`: writers open files in binary mode (`"w+b"`), required on
  Windows to avoid newline translation corrupting blocks.
- `bwRead.c`, `io.c`: portability for MSVC/C++ (no `void*` arithmetic,
  `unistd.h` only on POSIX, `bwCleanup(void)`, `CURLINFO_CONTENT_LENGTH_DOWNLOAD_T`).
- `bwRead.c`, `bwWrite.c`, `bwStats.c`, `bwValues.c`, `io.c`: diagnostic
  `fprintf`/`printf` messages to stderr/stdout are commented out; errors are
  reported through return values instead. In `io.c` this also comments out
  the HTTP range `sprintf` calls, which are only used in builds with libcurl
  (deepToolsR builds with `NOCURL`).
