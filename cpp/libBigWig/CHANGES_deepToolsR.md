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
- `bwWrite.c`, `bwRead.c`, `bigWig.h`: new `bwSetZoomBuilder(fp, builder, ctx)`.
  When a builder is set, `bwFinalize` keeps `makeZoomLevels`' choice of levels
  but calls the builder (`bwZoomBuilder_t`) instead of `constructZoomLevels`;
  the builder returns, per level, caller-owned, already compressed blocks
  (`bwZoomBlock_t`: extent, size, bytes) and their count (stored in `nNodes`,
  which `writeZoomLevels` only compares for duplicate levels).
  `writeZoomLevels` writes those blocks and their index entries instead of
  compressing its own zoom buffers; the zoom R-tree and headers are written as
  before. `bwWriteBuffer_t` gains `zoomBuilder`, `zoomBuilderCtx` and
  `zoomBlocks` (freed in `bwDestroyWriteBuffer`). deepToolsR's builder
  (`cpp/bigwig_writer.hpp`) also writes the sum and sum of squares of the last
  record of each block and level, which upstream leaves 0.
- `bwRead.c`: `bwCloseChecked` frees the zoom-buffer chains that a failed or
  cancelled `bwFinalize` leaves behind (`bwDestroyWriteBuffer` takes the
  header's level count); upstream frees only the array and leaks them.
- `bwWrite.c`: the allocation-failure path of `makeZoomLevels` sets the freed
  zoom-buffer arrays to NULL, so closing the file does not free them again.
- `bwRead.c`: writers open files in binary mode (`"w+b"`), required on
  Windows to avoid newline translation corrupting blocks.
- `bwRead.c`, `io.c`: portability for MSVC/C++ (no `void*` arithmetic,
  `unistd.h` only on POSIX, `bwCleanup(void)`, `CURLINFO_CONTENT_LENGTH_DOWNLOAD_T`).
- `bwRead.c`, `bwWrite.c`, `bwStats.c`, `bwValues.c`, `io.c`: diagnostic
  `fprintf`/`printf` messages to stderr/stdout are commented out; errors are
  reported through return values instead. In `io.c` this also comments out
  the HTTP range `sprintf` calls, which are only used in builds with libcurl
  (deepToolsR builds with `NOCURL`).
- `bigWigIO.h`, `io.c`, `bwRead.c`, `bwWrite.c`: local-file positions are
  64-bit everywhere (`bwFileTell`/`bwFileSeek`: `_ftelli64`/`_fseeki64` on
  Windows, where `long` is 32-bit; `ftell`/`fseek` elsewhere). `bwTell`
  returns `int64_t`, and the raw `ftell`/`fseek` calls in `bwWrite.c`
  (`writeAtPos`, chromosome tree, header offsets, zoom data offsets) use the
  wrappers, so files over 2 GiB work on Windows.
- `bwWrite.c`: `bwFinalize` returns an error for a write-open that failed
  before its URL or write buffer existed; upstream dereferenced the NULL
  buffer when `bwOpen` cleaned up (a crash on an unopenable output).
- `bwValues.c`: an interval with `end <= start` (empty, reversed, or a
  wrapped `start + span`) makes the block read fail instead of being
  returned or skipped.
- `bwValues.c`, `bwRead.c`, `bwCommon.h`: an R-tree entry whose chromosome
  ids are reversed or outside the chromosome list, or whose block (or child
  node) lies outside the file, makes the index read fail; upstream silently
  dropped the data. The file size comes from the new internal
  `bwFileSize()`, which the chromosome-list checks also use.
- `bwWrite.c`: when no zoom level is possible (`makeZoomLevels`' mean width
  overflow), `bwFinalize` writes the file with none; upstream failed (or
  dereferenced the missing zoom headers). `bwAddIntervals` returns the
  `flushBuffer` errors it ignored, and `bwCloseChecked` fails when the
  output stream has its error flag set.
- `bwWrite.c`: the header summary's maximum starts at `-DBL_MAX` (was
  `DBL_MIN`, the smallest positive double) and `updateStats` updates minimum
  and maximum independently, so a descending or all-negative track records
  its true maximum; `span * value` is multiplied in double, not float.
- `bwWrite.c`: `makeZoomLevels` computes the first zoom size in 64 bits, so a
  mean interval width of 2^30 or more gives no zoom levels instead of
  wrapping to a tiny zoom size.
- `bwWrite.c`, `bigWig.h`: a zoom level's data count is its record count (as
  the format specifies), not its block count; `bwZoomBlock_t` gains
  `records`.
- `bwValues.c`: an empty R-tree root whose index declares items, an empty
  child node, and a data section whose chromosome id is outside the
  chromosome list make the read fail; upstream returned or wrote an empty
  track.
