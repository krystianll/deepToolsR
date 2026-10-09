# deepToolsR 0.10.2-beta release benchmarks

deepToolsR 0.10.2-beta compared with deepTools 3.5.6 (Python) and deepTools
4.0.0 (Rust core). All three tools were timed in this run, on the same
machine, with the same data and the same harness. Each result is the median of
3 runs, with the min and max also given. Every run is in `runs.csv` and
`runs.json`, the per-case summary is in `summary.csv`, and the output
comparisons are in `parity.json`.

All benchmarks are complete: 399 recorded runs, all succeeded (no crashes, no
timeouts). Figures are in `figures/` (PNG and SVG).

## Headline

- **bamCoverage, ChIP, binSize 1, `-p 8`:** deepToolsR 6.2 s / 139 MB;
  deepTools 3.5.6 159 s / 3.0 GB; deepTools 4.0.0 12.9 s / 3.7 GB.
- **Same ChIP data as CRAM, binSize 1, `-p 8`:** with an external reference
  deepToolsR 7.0 s / 127 MB, 3.5.6 178 s / 4.0 GB, 4.0.0 13.7 s / 3.9 GB. With
  an embedded reference deepToolsR 9.5 s / 148 MB, 3.5.6 196 s / 4.9 GB,
  4.0.0 14.1 s / 4.3 GB.
- **Stranded RNA, both strands, binSize 1, `-p 8`:** deepToolsR 11.4 s /
  213 MB in one run; 3.5.6 484 s / 3.1 GB; 4.0.0 30.2 s / 2.9 GB.
- **computeMatrix, 20k genes x 4 samples, `-p 8`:** deepToolsR 5.5 s /
  511 MB; 3.5.6 167 s / 3.3 GB; 4.0.0 20.4 s / 1.2 GB. 4.0.0's computeMatrix
  does not get faster beyond `-p 4`.
- **Stranded computeMatrix, same size, `-p 8`:** deepToolsR 6.1 s / 496 MB in
  one run; 3.5.6 236 s / 2.2 GB and 4.0.0 81.6 s / 1.3 GB using split + rbind.
- **plotHeatmap on the big matrix, same pixel geometry for every tool:**
  deepToolsR 5.6 s at `-p 1` and 2.0 s at `-p 8`; 3.5.6 12.4 s / 2.3 GB;
  4.0.0 9.6 s / 2.3 GB (both single-threaded). deepToolsR peaks at
  0.5-0.7 GB.
- **Outputs:** deepToolsR reproduces 3.5.6 (differences explained below) and
  matches 4.0.0 exactly for RNA coverage and every matrix. For every tool,
  CRAM output is identical to its own BAM output.

Memory is the peak physical footprint (see Method), not RSS.

## Machine and versions

| | |
|---|---|
| Machine | Apple M4 Pro, 14 cores (10 performance + 4 efficiency), 48 GB RAM |
| OS | macOS 27.0.1 (arm64) |
| deepToolsR | 0.10.2-beta (pip version 0.10.2b0, clean non-editable install) |
| deepTools | 3.5.6 (pip, Python 3.14) and 4.0.0 (conda) |
| samtools | 1.24 (CRAM creation only) |
| Harness Python | 3.14.7, psutil |

`machine.json` holds the same details as recorded by the harness.

## Data

- **ChIP-seq (BAM):** ENCODE HCT116 H3K4me3, ENCSR333OPW replicate 1, file
  [ENCFF772IUE](https://www.encodeproject.org/files/ENCFF772IUE/). GRCh38
  filtered alignments, single-end, 28,418,040 mapped reads (`samtools flagstat`).
- **RNA:** POINT-seq (nascent RNA), paired-end, HCT116, 70.8 M alignments
  (unpublished).
- **computeMatrix input:** POINT-seq sense-strand coverage, HCT116, 4
  conditions (unpublished). binSize 1 bigWigs, labelled S1–S4 in all outputs.
  Each sample has a plus-strand and a minus-strand track. The minus-strand
  track is stored as negative values. The regular matrix benchmarks use the
  plus-strand track only; the stranded benchmark uses both.
- **Regions:** 20,107 protein-coding genes (`gene` records with
  `gene_type "protein_coding"`) from GENCODE v50 basic annotation, as a
  6-column BED. The computeMatrix gene sets are nested random draws from it
  (numpy `default_rng(20260930).permutation`, first 3,000 / 10,000 / 20,000
  genes, written as sorted BEDs beside the source BED).

No data is included in the repository.

- **ChIP-seq (CRAM):** the same reads as CRAM, in two variants made with
  samtools 1.24 (commands under "Reproducing"). *Reference CRAM* is encoded
  against the GRCh38 no-alt analysis set (`GRCh38_no_alt_analysis_set.fasta`),
  the reference ENCODE aligns to. *Embedded-reference CRAM* is made without a
  reference, so samtools embeds one built from the reads.

## Method

- **Machine and conditions:** all runs were made on the machine above with the
  VMs shut down and background file sync paused. Before each run the harness
  waits for an idle gate: system CPU below 15% and no pytest, pip, cmake or
  busy VM process. A run during which one of these appears is discarded and
  repeated.
- **Repeats and statistics:** every configuration is run 3 times. The tables
  give the median with min and max. The figures plot the mean of the n = 3
  runs with a shaded two-sided 95% t-interval:
  mean ± t(0.975, n−1) · sd / √n, with t(0.975, 2) = 4.303. `summary.csv`
  holds both the medians and the means with their intervals.
- **Wall time:** measured around each process.
- **Memory (headline: peak physical footprint):** while a run is going, the
  harness polls the process tree every 50 ms to find the processes in it. For
  each process it records the macOS kernel's own lifetime peak physical
  footprint (`ri_lifetime_max_phys_footprint` from `proc_pid_rusage`, the same
  number `/usr/bin/time -l` prints as "peak memory footprint"; the harness
  agrees with it within 1%). The run's value is the sum of those peaks over
  all processes in the tree. The kernel value only grows, so a short peak
  between two polls is not missed. deepToolsR and deepTools 4.0.0 run as one
  process with threads, so for them the figure is exact. deepTools 3.5.6 forks
  worker processes whose peaks need not coincide in time, so its figures at
  `-p` above 1 are an upper bound.
- **Why not RSS:** on macOS, RSS also counts memory that was freed but that
  the allocator has returned and the OS has not yet reclaimed (reusable
  pages), so it overstates real use, most for threaded, allocation-heavy runs.
  Periodic sampling of RSS can also miss a short peak. Summed RSS is still
  recorded in the CSVs (`rss_*` columns of `summary.csv`, `peak_rss_mb` in
  `runs.csv`) as a secondary upper bound, but it is not used in the tables.
  On Linux the harness would record per-process `VmHWM` instead; that was not
  used here.
- **Multi-step cases:** for cases that need several commands (see "Option
  differences") the wall time is the sum of the steps and the footprint is the
  largest of the steps.
- **CRAM creation and reference handling:** both CRAMs are made from the same
  BAM with samtools 1.24: `samtools view -C -T GRCh38_no_alt_analysis_set.fasta`
  (reference CRAM) and `samtools view -C` without `-T` (embedded-reference
  CRAM). For the reference CRAM, deepToolsR is given `--referenceFasta`;
  deepTools 3.5.6 and 4.0.0 find the FASTA through the `UR:` path in the CRAM
  header. For the embedded-reference CRAM no tool is given a reference. Every
  CRAM run sets `REF_PATH` to a non-existent local cache, so htslib never
  tries to download a reference from the EBI server during timing.
- **Failed runs:** a run that exits non-zero (for example a crash) is recorded
  with its status and not retried; the remaining repeats still run. The `n`
  and `n_failed` columns of `summary.csv` give the successful and failed
  counts. In this run there were no failures.
- **Timeout:** 1800 s per run. No run reached it.
- **Parity:** after the timing runs, `bench.py parity` compares outputs of the
  `-p 8` runs: bigWigs base by base over chr19, chr21 and chr22 (maximum
  absolute difference and the number of bases outside a tolerance of 1e-3
  relative), matrices cell by cell with rows matched by region. 38 checks were
  made; see `parity.json` and the sections below.
- **Figures:** line graphs, one panel per benchmark case, with a separate
  wall-time figure (`*_wall`) and peak-footprint figure (`*_mem`) for each
  group. The x axis is the processor count (`-p` 1, 4, 8), or the matrix size
  for plotHeatmap and plotProfile. The y axis is log-scaled when a panel's
  range spans more than 10x.

### Option differences

- **ChIP bamCoverage (BAM and CRAM):** all three tools use
  `--binSize {50,1} --normalizeUsing CPM -p {1,4,8}`. For the reference CRAM,
  deepToolsR also gets `--referenceFasta`.
- **RNA bamCoverage:** every tool runs at `-p 1`, `-p 4` and `-p 8`, except
  deepTools 3.5.6 at binSize 1, which runs at `-p 8` only (owner decision:
  its two strand runs take too long at lower thread counts). That case is a
  single point in the figures.
  - 3.5.6 and 4.0.0 run twice, with `--filterRNAstrand forward` and then
    `reverse`. Time is the sum of the two runs; footprint is the larger of
    the two.
  - deepToolsR writes both strands in one run with
    `--strandedness reverse --filterRNAstrand split`. It also needs
    `--normalizeUsing read-count`: its default unnormalized output is mean
    per-base coverage, while deepTools' default is reads per bin.
  - A chr21 check showed the deepToolsR split outputs identical to 3.5.6's two
    runs (max difference 0).
- **computeMatrix:** identical options for all three tools
  (`--binSize 10 --samplesLabel S1… -p {1,4,8}`), at three sizes:

  | Size | Genes | Samples | Mode | Bins per sample |
  |---|---:|---:|---|---:|
  | small | 3,000 | 1 (S1) | reference-point `--referencePoint TSS -b 3000 -a 3000` | 600 |
  | medium | 10,000 | 1 (S1) | reference-point, same settings | 600 |
  | big | 20,000 | 4 (S1–S4) | scale-regions `-m 4000 -b 3000 -a 3000` | 1,000 |

- **computeMatrix, stranded:** each gene takes the signal of its own strand.
  Sizes are medium (sample S1) and big (S1–S4), with the same settings as
  above, at `-p 1`, `-p 4` and `-p 8`; stranded big was run at `-p 8` only
  (owner decision).
  - deepToolsR: one run with the plus-strand tracks as `-S` and the matching
    minus-strand tracks as `--scoreFileNameMinus`.
  - 3.5.6 and 4.0.0 have no equivalent. They run three commands: computeMatrix
    on the + strand genes with the plus-strand tracks, computeMatrix on the −
    strand genes with the minus-strand tracks, then `computeMatrixOperations
    rbind`. Each tool uses its own rbind (4.0.0 has one). Time is the sum of
    the three commands; footprint is the largest of the three.
  - Row order differs between the two approaches. deepToolsR keeps the
    combined gene order, while rbind puts all + genes before all − genes.
    Parity therefore matches rows by region.
- **plotHeatmap / plotProfile:** deepTools 3.5.6 and 4.0.0 plot
  single-threaded. deepToolsR is run at an explicit `-p 1` and `-p 8`. For
  each size, every tool plots the same matrix (deepToolsR's `-p 8` output),
  with the same figure geometry. See "Plot geometry" below.

## Results: bamCoverage, ChIP-seq (BAM, CPM)

Wall time in seconds, median (min to max):

| binSize | -p | deepToolsR | deepTools 3.5.6 | deepTools 4.0.0 |
|---:|---:|---:|---:|---:|
| 50 | 1 | **8.1** (7.9–8.2) | 100 (99.9–102) | 21.0 (20.9–21.2) |
| 50 | 4 | **2.7** (2.7–2.7) | 40.8 (40.5–41.1) | 11.4 (11.1–11.4) |
| 50 | 8 | **2.8** (2.7–2.8) | 31.3 (31.3–31.5) | 9.5 (9.4–9.6) |
| 1 | 1 | **17.4** (17.4–17.6) | 792 (789–798) | 27.3 (27.2–27.6) |
| 1 | 4 | **6.0** (6.0–6.1) | 248 (246–252) | 15.0 (14.9–15.2) |
| 1 | 8 | **6.2** (6.1–6.2) | 159 (157–161) | 12.9 (12.9–12.9) |

Peak physical footprint of the process tree in MB, median (min to max):

| binSize | -p | deepToolsR | deepTools 3.5.6 | deepTools 4.0.0 |
|---:|---:|---:|---:|---:|
| 50 | 1 | **67** (65–69) | 196 (193–198) | 206 (173–230) |
| 50 | 4 | **71** (71–73) | 455 (455–461) | 222 (209–226) |
| 50 | 8 | **87** (87–88) | 712 (711–712) | 258 (236–259) |
| 1 | 1 | **85** (81–86) | 355 (354–355) | 1,078 (1,074–1,079) |
| 1 | 4 | **93** (91–93) | 1,636 (1,634–1,657) | 2,780 (2,532–3,114) |
| 1 | 8 | **139** (138–139) | 3,048 (3,048–3,057) | 3,731 (3,580–4,631) |

Figures: `figures/bamcov_chip_wall.{png,svg}` and
`figures/bamcov_chip_mem.{png,svg}`.

Reading the tables:

- **Speed:** deepToolsR is 12x faster than 3.5.6 at binSize 50 (`-p 1`:
  100.4 / 8.1 = 12.4) and 45x faster at binSize 1 (`-p 1`: 792.5 / 17.4 =
  45.5). It is 2.6-4.2x faster than 4.0.0 at binSize 50 (21.0 / 8.1 = 2.6,
  11.4 / 2.7 = 4.2, 9.5 / 2.8 = 3.5) and 1.6-2.5x faster at binSize 1
  (27.3 / 17.4 = 1.6, 15.0 / 6.0 = 2.5, 12.9 / 6.2 = 2.1).
- **Scaling:** deepToolsR's wall time does not improve from `-p 4` to `-p 8`
  on this BAM (reading the file is the limit): 2.7 s and 2.8 s at binSize 50,
  6.0 s and 6.2 s at binSize 1.
- **Memory:** deepToolsR stays under 150 MB in every configuration (largest
  single run 139 MB). At binSize 1, 4.0.0's footprint grows with the thread
  count: 1,078 MB at `-p 1`, 2,780 MB at `-p 4`, 3,731 MB median and 4,631 MB
  max at `-p 8`.

### Output parity (chr19, chr21, chr22, every base; `-p 8` outputs)

| Comparison | binSize 50 | binSize 1 |
|---|---|---|
| deepToolsR vs 3.5.6 | max abs diff 4.8e-6 (float32 rounding) | max abs diff 4.6e-6 |
| deepToolsR vs 4.0.0 | max abs diff 0.005 | max abs diff 0.005 |

The small, widespread difference against 4.0.0 is consistent with 4.0.0
computing the CPM scale factor slightly differently. 3.5.6 and deepToolsR
agree.

## Results: bamCoverage, ChIP-seq as CRAM (CPM)

The same 28,418,040 reads as CRAM, same options as the BAM runs. Two variants
(see Data and Method): a CRAM encoded against the GRCh38 no-alt analysis set,
and a CRAM with an embedded reference.

### Reference CRAM

Wall time in seconds, median (min to max):

| binSize | -p | deepToolsR | deepTools 3.5.6 | deepTools 4.0.0 |
|---:|---:|---:|---:|---:|
| 50 | 1 | **10.8** (10.7–10.8) | 253 (251–254) | 29.4 (29.3–29.4) |
| 50 | 4 | **4.0** (3.9–4.0) | 82.3 (82.0–82.4) | 13.2 (13.0–13.2) |
| 50 | 8 | **3.7** (3.7–3.8) | 52.1 (52.1–52.5) | 10.7 (10.4–10.7) |
| 1 | 1 | **19.3** (19.1–19.5) | 941 (927–945) | 34.2 (34.2–34.2) |
| 1 | 4 | **8.2** (8.2–8.3) | 285 (283–292) | 16.6 (16.6–16.9) |
| 1 | 8 | **7.0** (7.0–7.1) | 178 (178–183) | 13.7 (13.7–14.0) |

Peak physical footprint in MB, median (min to max):

| binSize | -p | deepToolsR | deepTools 3.5.6 | deepTools 4.0.0 |
|---:|---:|---:|---:|---:|
| 50 | 1 | **61** (59–62) | 195 (188–197) | 203 (180–205) |
| 50 | 4 | **64** (63–68) | 942 (932–960) | 215 (213–222) |
| 50 | 8 | **82** (82–83) | 1,677 (1,665–1,679) | 263 (253–275) |
| 1 | 1 | **79** (78–82) | 362 (361–364) | 1,109 (1,109–1,111) |
| 1 | 4 | **89** (87–89) | 2,106 (2,099–2,109) | 2,511 (2,419–2,521) |
| 1 | 8 | **127** (125–128) | 3,998 (3,983–4,021) | 3,863 (3,723–4,122) |

### Embedded-reference CRAM

Wall time in seconds, median (min to max):

| binSize | -p | deepToolsR | deepTools 3.5.6 | deepTools 4.0.0 |
|---:|---:|---:|---:|---:|
| 50 | 1 | **15.1** (15.1–15.1) | 345 (342–345) | 30.0 (29.9–30.1) |
| 50 | 4 | **7.1** (7.1–7.1) | 107 (106–109) | 13.7 (13.3–13.8) |
| 50 | 8 | **6.1** (6.1–6.2) | 68.5 (67.7–68.6) | 10.8 (10.7–11.0) |
| 1 | 1 | **24.1** (24.0–24.1) | 1,023 (1,021–1,046) | 35.7 (35.5–35.7) |
| 1 | 4 | **12.6** (12.5–12.7) | 316 (313–319) | 17.1 (17.0–17.2) |
| 1 | 8 | **9.5** (9.5–9.6) | 196 (194–197) | 14.1 (14.0–14.6) |

Peak physical footprint in MB, median (min to max):

| binSize | -p | deepToolsR | deepTools 3.5.6 | deepTools 4.0.0 |
|---:|---:|---:|---:|---:|
| 50 | 1 | **109** (108–110) | 228 (227–230) | 186 (185–187) |
| 50 | 4 | **111** (109–113) | 1,529 (1,500–1,546) | 212 (198–226) |
| 50 | 8 | **115** (113–117) | 2,773 (2,701–2,790) | 268 (248–296) |
| 1 | 1 | **122** (121–125) | 402 (400–404) | 1,110 (1,088–1,111) |
| 1 | 4 | **128** (126–130) | 2,533 (2,496–2,539) | 2,725 (2,552–2,896) |
| 1 | 8 | **148** (147–160) | 4,853 (4,806–4,892) | 4,312 (3,841–4,345) |

Figures: `figures/bamcov_chip_cram_wall.{png,svg}` and
`figures/bamcov_chip_cram_mem.{png,svg}`.

Reading the tables:

- **Speed against the other tools:** across all twelve CRAM cases deepToolsR
  is 11-49x faster than 3.5.6 (smallest: embedded, binSize 50, `-p 8`,
  68.5 / 6.1 = 11.1; largest: reference, binSize 1, `-p 1`, 941.3 / 19.3 =
  48.8) and 1.4-3.3x faster than 4.0.0 (smallest: embedded, binSize 1, `-p 4`,
  17.1 / 12.6 = 1.35; largest: reference, binSize 50, `-p 4`, 13.2 / 4.0 =
  3.3).
- **CRAM against BAM, same tool:** deepToolsR needs 1.1-1.5x the BAM time with
  the reference CRAM (for example binSize 1, `-p 8`: 7.0 / 6.2 = 1.14) and
  1.4-2.6x with the embedded-reference CRAM (for example binSize 50, `-p 4`:
  7.1 / 2.7 = 2.6). 4.0.0 needs 1.1-1.4x the BAM time with the reference CRAM
  and 1.1-1.4x with the embedded one. 3.5.6 needs 1.1-2.5x (reference) and
  1.2-3.4x (embedded).
- **Memory:** deepToolsR stays under 160 MB in every CRAM configuration
  (largest single run 160 MB). At binSize 1 and `-p 8`, 3.5.6 and 4.0.0 reach
  4.0 GB and 3.9 GB (reference CRAM) and 4.9 GB and 4.3 GB (embedded), against
  127 MB and 148 MB for deepToolsR.

### CRAM output parity (chr19, chr21, chr22 plus all other chromosomes of the bigWig, `-p 8`)

- For each tool, its CRAM output is identical (max difference 0) to its own
  BAM output, for both CRAM variants and both bin sizes.
- deepToolsR's CRAM output against 3.5.6's CRAM output differs only by float32
  rounding: max abs diff 4.8e-6 at binSize 50 and 4.6e-6 at binSize 1, for
  both variants, as for BAM.

## Results: bamCoverage, stranded RNA (both strands)

deepToolsR writes both strands in one run. 3.5.6 and 4.0.0 need two runs;
their time is the sum of the two runs and their footprint the larger of the
two. 3.5.6 at binSize 1 was run at `-p 8` only.

Wall time in seconds, median (min to max):

| binSize | -p | deepToolsR | deepTools 3.5.6 | deepTools 4.0.0 |
|---:|---:|---:|---:|---:|
| 10 | 1 | **19.0** (18.9–19.4) | 1,434 (1,409–1,435) | 89.6 (89.6–89.8) |
| 10 | 4 | **5.9** (5.7–5.9) | 438 (424–438) | 36.1 (35.9–36.2) |
| 10 | 8 | **5.5** (5.4–5.5) | 258 (257–265) | 25.3 (25.0–25.5) |
| 1 | 1 | **35.1** (34.9–35.8) | not run | 98.3 (98.2–98.4) |
| 1 | 4 | **11.5** (11.5–11.7) | not run | 39.5 (39.3–39.5) |
| 1 | 8 | **11.4** (11.4–11.4) | 484 (479–491) | 30.2 (30.1–30.6) |

Peak physical footprint in MB, median (min to max):

| binSize | -p | deepToolsR | deepTools 3.5.6 | deepTools 4.0.0 |
|---:|---:|---:|---:|---:|
| 10 | 1 | **77** (75–83) | 192 (192–201) | 465 (457–475) |
| 10 | 4 | **101** (100–102) | 545 (544–557) | 487 (484–496) |
| 10 | 8 | **148** (147–148) | 922 (914–925) | 483 (472–496) |
| 1 | 1 | **96** (94–97) | not run | 1,093 (1,086–1,115) |
| 1 | 4 | **131** (131–141) | not run | 1,802 (1,630–1,970) |
| 1 | 8 | **213** (205–214) | 3,084 (3,077–3,117) | 2,910 (2,594–3,003) |

Reading the tables:

- **Speed:** deepToolsR is 47-76x faster than 3.5.6 at binSize 10
  (1,434 / 19.0 = 75.6, 438 / 5.9 = 74.6, 258 / 5.5 = 47.0) and 42x at
  binSize 1, `-p 8` (484 / 11.4 = 42.4). It is 4.6-6.2x faster than 4.0.0 at
  binSize 10 (89.6 / 19.0 = 4.7, 36.1 / 5.9 = 6.2, 25.3 / 5.5 = 4.6) and
  2.6-3.4x at binSize 1 (98.3 / 35.1 = 2.8, 39.5 / 11.5 = 3.4,
  30.2 / 11.4 = 2.6).
- **Memory:** deepToolsR stays under 215 MB in every configuration (largest
  median 213 MB, binSize 1, `-p 8`).

**Parity** (chr19, chr21, chr22, every base; `-p 8`): the deepToolsR
`.plus`/`.minus` outputs are identical (max difference 0) to both the 3.5.6
and the 4.0.0 `--filterRNAstrand forward`/`reverse` outputs, at binSize 10
and binSize 1.

Figures: `figures/rna_wall.*` and `figures/rna_mem.*`.

## Results: computeMatrix

Wall time in seconds, median (min to max):

| Size | -p | deepToolsR | deepTools 3.5.6 | deepTools 4.0.0 |
|---|---:|---:|---:|---:|
| small | 1 | **0.7** (0.6–1.0) | 19.1 (18.9–19.4) | 0.8 (0.8–0.8) |
| small | 4 | **0.3** (0.3–0.4) | 6.0 (6.0–6.0) | 0.8 (0.8–0.8) |
| small | 8 | **0.3** (0.2–0.3) | 3.8 (3.7–3.8) | 0.8 (0.8–0.8) |
| medium | 1 | **1.8** (1.8–2.0) | 59.7 (59.7–59.9) | 2.4 (2.3–2.4) |
| medium | 4 | **0.7** (0.7–0.7) | 18.5 (18.4–18.5) | 2.3 (2.3–2.3) |
| medium | 8 | **0.5** (0.5–0.5) | 11.1 (10.9–11.4) | 2.3 (2.3–2.3) |
| big | 1 | **35.6** (35.4–36.9) | 846 (843–848) | 48.0 (48.0–48.4) |
| big | 4 | **10.4** (10.4–10.4) | 268 (267–268) | 20.4 (20.1–20.5) |
| big | 8 | **5.5** (5.5–5.5) | 167 (165–170) | 20.4 (20.1–20.4) |

Peak physical footprint in MB, median (min to max):

| Size | -p | deepToolsR | deepTools 3.5.6 | deepTools 4.0.0 |
|---|---:|---:|---:|---:|
| small | 1 | **42** (41–43) | 95 (94–95) | 138 (137–140) |
| small | 4 | **51** (51–51) | 311 (308–313) | 137 (137–139) |
| small | 8 | **67** (66–68) | 500 (498–501) | 138 (138–138) |
| medium | 1 | **70** (70–71) | 209 (208–209) | 389 (379–423) |
| medium | 4 | **78** (74–79) | 440 (440–441) | 419 (415–419) |
| medium | 8 | **90** (90–93) | 629 (629–632) | 384 (383–425) |
| big | 1 | **487** (486–488) | 1,621 (1,620–1,622) | 1,156 (1,135–1,266) |
| big | 4 | **494** (478–496) | 2,730 (2,685–2,775) | 1,220 (1,180–1,270) |
| big | 8 | **511** (510–515) | 3,291 (3,258–3,296) | 1,248 (1,169–1,249) |

4.0.0's computeMatrix time is flat from `-p 1` to `-p 8` for small and
medium (0.82, 0.76, 0.77 s and 2.35, 2.32, 2.34 s), and flat from `-p 4` to
`-p 8` for big (20.4 s both). deepToolsR keeps getting faster up to `-p 8` for
medium (1.8, 0.7, 0.5 s) and big (35.6, 10.4, 5.5 s); for small it is flat from
`-p 4` to `-p 8` (0.30 s and 0.29 s).

Figures: `figures/matrix_wall.*` and `figures/matrix_mem.*`.

## Results: computeMatrix, stranded

Stranded big was run at `-p 8` only (owner decision).

| Size | -p | Wall (s): deepToolsR | 3.5.6 | 4.0.0 | Footprint (MB): deepToolsR | 3.5.6 | 4.0.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| medium | 1 | **2.0** | 66.4 | 6.4 | **71** | 160 | 221 |
| medium | 4 | **0.8** | 23.7 | 6.3 | **78** | 345 | 220 |
| medium | 8 | **0.5** | 16.1 | 6.4 | **98** | 548 | 219 |
| big | 8 | **6.1** | 236 | 81.6 | **496** | 2,172 | 1,318 |

The values are medians; min and max are in `summary.csv`. For 3.5.6 and
4.0.0 the time covers all three commands (two computeMatrix runs and rbind),
and the footprint is the largest of the three.

Figures: `figures/matrix_stranded_wall.*` and `figures/matrix_stranded_mem.*`.

## Results: plotHeatmap and plotProfile

deepTools 3.5.6 and 4.0.0 plot single-threaded. deepToolsR was run at `-p 1`
and at `-p 8`. For each size, every tool plots the same matrix
(deepToolsR's `-p 8` computeMatrix output) at identical pixel geometry.

Wall time in seconds, median (min to max):

| Plot | Size | deepToolsR -p 1 | deepToolsR -p 8 | deepTools 3.5.6 | deepTools 4.0.0 |
|---|---|---:|---:|---:|---:|
| plotHeatmap | small | 0.54 (0.52–0.85) | **0.47** (0.47–0.47) | 0.72 (0.72–0.72) | 0.60 (0.59–0.66) |
| plotHeatmap | medium | 0.78 (0.77–0.78) | **0.53** (0.53–0.53) | 1.34 (1.34–1.35) | 1.09 (1.07–1.10) |
| plotHeatmap | big | 5.55 (5.52–5.58) | **2.03** (2.02–2.05) | 12.38 (12.35–12.39) | 9.60 (9.56–9.80) |
| plotProfile | small | 0.40 (0.40–0.40) | **0.41** (0.40–0.41) | 0.52 (0.47–0.53) | 0.47 (0.46–0.47) |
| plotProfile | medium | 0.53 (0.53–0.53) | **0.42** (0.41–0.43) | 0.83 (0.82–0.83) | 0.83 (0.82–0.83) |
| plotProfile | big | 2.67 (2.65–2.75) | **1.37** (1.36–1.39) | 6.05 (6.05–6.08) | 6.04 (6.01–6.05) |

Peak physical footprint in MB, median:

| Plot | Size | deepToolsR -p 1 | deepToolsR -p 8 | deepTools 3.5.6 | deepTools 4.0.0 |
|---|---|---:|---:|---:|---:|
| plotHeatmap | small | 131 | **123** | 232 | 233 |
| plotHeatmap | medium | **158** | 160 | 561 | 564 |
| plotHeatmap | big | **486** | 728 | 2,314 | 2,306 |
| plotProfile | small | **82** | 83 | 105 | 104 |
| plotProfile | medium | **102** | 103 | 230 | 231 |
| plotProfile | big | **410** | 413 | 1,407 | 1,407 |

Single-threaded against single-threaded, deepToolsR plotHeatmap is 2.2x
faster than 3.5.6 on the big matrix (12.38 / 5.55 = 2.2), and plotProfile is
2.3x faster (6.05 / 2.67 = 2.3). deepToolsR's footprint on the big matrix is
486 MB (`-p 1`) and 728 MB (`-p 8`) for plotHeatmap, against about 2.3 GB for
both deepTools versions. For plotProfile, 3.5.6 and 4.0.0 are practically
identical (6.05 s and 6.04 s on the big matrix); for plotHeatmap, 4.0.0 is
22% faster than 3.5.6 on the big matrix (9.60 vs 12.38 s) with the same
footprint. 4.0.0 keeps the Python (matplotlib) plotting code, and only its
computational core is in Rust.

**Note on thread counts.** deepToolsR's plotting tools default to
`-p auto`, which means half the logical CPUs: 7 threads on this machine. The
harness therefore always passes an explicit `-p` to every deepToolsR command,
so the `-p 1` and `-p 8` columns above are exactly those thread counts.

Figures: `figures/plot_wall.*` and `figures/plot_mem.*`.

### Plot geometry

The tools' default figure sizes differ, and so does their default sort order.
deepTools uses a 28 cm tall heatmap stack, 4 cm per sample, a 7 × 11 cm
profile and 200 dpi, sorting in descending order. deepToolsR defaults to
10 × 5 cm and 5 × 5 cm panels, sorting in ascending order. To compare like
with like, every run sets `--dpi 200`, and every plotHeatmap run also sets
`--sortRegions descend`.

deepTools 3.5.6 is the reference, with
`--heatmapHeight 28 --heatmapWidth 4` and `--plotHeight 7 --plotWidth 11`.

The same centimetre values do not give the same pixels across tools:

- deepToolsR applies them to the data panel itself. `--heatmapHeight` is the
  whole heatmap-stack height and `--heatmapWidth` is the per-sample cell
  width. At 28 × 4 cm it drew a 2,205 × 315 px heatmap.
- 3.5.6 and 4.0.0 apply them to the figure grid, which also holds the profile
  row, the colour bar and the labels. The heatmap itself came out at
  1,680 × 212 px in 3.5.6 and a different size in 4.0.0.

So the deepToolsR and 4.0.0 values were calibrated until each data panel
matched 3.5.6 in pixels. Panel sizes were measured from the axis frame lines
in the benchmark PNGs.

| Tool | plotHeatmap height / width (cm) | plotProfile height / width (cm) |
|---|---|---|
| 3.5.6 (reference) | 28 / 4 | 7 / 11 |
| deepToolsR | 21.34 / 2.69 (small, medium), 3.18 (big) | 4.71 / 9.54 (small, medium), 10.26 (big) |
| 4.0.0 | 28.96 / 4.64 (small), 4.40 (medium), 4.54 (big) | 7.60 / 10.73 (small, medium), 10.88 (big) |

Resulting data-panel sizes in pixels (height × width per sample, frame
included):

| Plot | Size | 3.5.6 | deepToolsR | 4.0.0 |
|---|---|---|---|---|
| plotHeatmap | small | 1,691 × 212 | 1,691 × 211 | 1,690 × 209 |
| plotHeatmap | medium | 1,691 × 212 | 1,691 × 211 | 1,690 × 211 |
| plotHeatmap | big (4 samples) | 1,691 × 250 | 1,691 × 250 | 1,690 × 247–248 |
| plotProfile | small, medium | 382 × 751 | 382 × 751 | 377 × 749 |
| plotProfile | big (4 samples) | 382 × 808 | 382 × 808 | 377 × 807–808 |

Whole-image sizes still differ, because of titles, colour bars and margins:

| Plot | Size | 3.5.6 | deepToolsR | 4.0.0 |
|---|---|---|---|---|
| plotHeatmap | small | 401 × 2,165 | 445 × 2,073 | 444 × 2,645 |
| plotHeatmap | medium | 401 × 2,165 | 445 × 2,073 | 425 × 2,626 |
| plotHeatmap | big | 1,308 × 2,165 | 1,302 × 2,111 | 1,508 × 2,637 |
| plotProfile | small | 866 × 551 | 847 × 524 | 836 × 549 |
| plotProfile | medium | 866 × 551 | 869 × 524 | 858 × 549 |
| plotProfile | big | 3,464 × 551 | 3,407 × 524 | 3,418 × 549 |

Every plot run uses the calibrated geometry, because the default figure
settings are not comparable across tools. The deepToolsR `-p 1` and `-p 8`
runs use the calibrated sizes of the deepToolsR rows in the geometry table.
Their panel sizes were measured: 1,691 × 250 px for the big heatmap and 382 × 751 px
for the small profile, identical at both thread counts.

## Matrix parity

All comparisons use the `-p 8` matrices, with rows matched by region
(chrom, start, end, strand). Every tool has the same row count in each
comparison. Row counts are 3,000, 9,999 and 19,997: regions with identical
coordinates share one key, so a few duplicate genes are counted once.

**deepToolsR vs 4.0.0:** identical (max difference 0) for small, medium, big,
stranded medium and stranded big.

**deepToolsR vs 3.5.6:**

- **small:** max difference 2.5e-4. **medium:** 3.9e-3. These are float32
  rounding on very large values; the medium maximum is in an MT-ND1 row with
  values around 82,455.
- **big (scale-regions):** 7,166 of 80.0 M cells differ, in 373 of 19,997
  rows. The largest difference is 3.29.
  - 6,149 of those cells are body bins at a bin-edge tie. They come in
    adjacent pairs where a body-bin edge (region length × bin index / 400)
    lands exactly on an integer. For example, a 122,512 bp gene has an
    integer edge every 25 bins: 122,512 × 75 / 400 = 22,971.
  - 3.5.6 computes bin edges with `np.linspace(..., dtype=int)`, which
    truncates floats; at some of these ties the float lands just below the
    integer (e.g. 8171.999...) and the edge moves 1 bp early. deepToolsR and
    4.0.0 place it exactly. One base moves to the neighbouring bin, so both
    bins' means differ; this is a 1 bp edge shift, not a signal difference.
    On binSize 1 nascent-RNA coverage a single base can carry a large value,
    hence differences of a few units.
  - None of the differing body cells is outside such a tie.
  - The other 1,017 cells are in flank bins of 4 rows (three chrM genes and
    GSTA4), whose values reach tens of thousands. These are float32 rounding
    differences.
- **Stranded matrices:** the deepToolsR one-run matrix equals the split +
  rbind matrices, compared row by row by region. It is identical to 4.0.0.
  Against 3.5.6 it shows exactly the unstranded differences: 3.9e-3 for
  medium and 4.71 for big, from the same bin-edge ties.
- **Row order:** rbind puts all + strand genes before all − strand genes,
  while deepToolsR keeps the gene order of the BED. This is why rows are
  matched by region rather than by position.

## Reproducing

The data paths below are placeholders. The harness is portable: tool
directories and data are arguments, and `--tools` runs a subset (for example
deepToolsR only).

The two CRAMs are made from the ChIP BAM with samtools 1.24:

```sh
# reference CRAM: encoded against the GRCh38 no-alt analysis set
samtools view -C -T GRCh38_no_alt_analysis_set.fasta \
    -o DATA/ENCFF772IUE.ref.cram DATA/ENCFF772IUE.bam
samtools index DATA/ENCFF772IUE.ref.cram
# embedded-reference CRAM: no -T, samtools embeds a reference built from the reads
samtools view -C -o DATA/ENCFF772IUE.noref.cram DATA/ENCFF772IUE.bam
samtools index DATA/ENCFF772IUE.noref.cram
```

```sh
python benchmarks/release/bench.py run \
    --chip-bam DATA/ENCFF772IUE.bam --rna-bam RNA.bam \
    --chip-cram-ref DATA/ENCFF772IUE.ref.cram \
    --chip-cram-noref DATA/ENCFF772IUE.noref.cram \
    --reference-fasta GRCh38_no_alt_analysis_set.fasta \
    --regions DATA/gencode.v50.protein_coding_genes.bed --workdir DATA/runs \
    --matrix-bigwigs S1.plus.bw S2.plus.bw S3.plus.bw S4.plus.bw \
    --matrix-bigwigs-minus S1.minus.bw S2.minus.bw S3.minus.bw S4.minus.bw \
    --r-bin .venv/bin --v356-bin DT356/bin --v400-bin DT400/bin
python benchmarks/release/bench.py parity --workdir DATA/runs
python benchmarks/release/bench.py report --workdir DATA/runs --out benchmarks/release
```

To build the gene BED from the GENCODE GTF:

```sh
gzcat gencode.v50.basic.annotation.gtf.gz | awk -F'\t' -v OFS='\t' \
  '$3=="gene" && $9~/gene_type "protein_coding"/ {match($9,/gene_name "[^"]+"/);
   print $1,$4-1,$5,substr($9,RSTART+11,RLENGTH-12),".",$7}' \
  | sort -k1,1 -k2,2n > gencode.v50.protein_coding_genes.bed
```

### Resuming

`run` is resumable: it skips every run already recorded in
`WORKDIR/runs.jsonl`. Rerunning the same `run` command therefore continues
with the missing work. `--only bamcov_chip bamcov_chip_cram rna matrix
matrix_stranded plot` restricts it to those groups. After that, rerun
`parity` and `report`.
