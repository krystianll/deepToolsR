# deepToolsR 0.9.0 release benchmarks

deepToolsR 0.9.0 compared with deepTools 3.5.6 (Python) and deepTools 4.0.0
(Rust core). Each result is the median of 3 runs, with the min and max also
given. Every run is in `runs.csv` and `runs.json`, the per-case summary is in
`summary.csv`, and the output comparisons are in `parity.json`.

All benchmarks are complete: 276 recorded runs. Figures are in `figures/`
(PNG and SVG).

## Headline

- **bamCoverage, ChIP, binSize 1, `-p 8`:** deepToolsR 9.6 s / 424 MB;
  deepTools 3.5.6 161 s / 3.5 GB; deepTools 4.0.0 12.9 s / 5.4 GB.
- **Stranded RNA, both strands, binSize 1, `-p 8`:** deepToolsR 16.2 s /
  582 MB in one run; 3.5.6 496 s / 3.6 GB; 4.0.0 30.8 s / 5.3 GB.
- **computeMatrix, 20k genes × 4 samples, `-p 8`:** deepToolsR 5.5 s /
  878 MB; 3.5.6 171 s / 3.6 GB; 4.0.0 20.3 s / 1.6 GB. 4.0.0's computeMatrix
  does not get faster beyond `-p 4`.
- **Stranded computeMatrix, same size, `-p 8`:** deepToolsR 6.2 s in one run;
  3.5.6 236 s and 4.0.0 82 s using split + rbind.
- **plotHeatmap on the big matrix, same pixel geometry for every tool:**
  deepToolsR 5.8 s at `-p 1` and 2.2 s at `-p 8`; 3.5.6 12.7 s / 4.2 GB;
  4.0.0 9.8 s / 4.2 GB (both single-threaded). deepToolsR peaks at
  0.7–0.9 GB.
- **Outputs:** deepToolsR reproduces 3.5.6 (differences explained below) and
  matches 4.0.0 exactly for RNA coverage and every matrix.

## Machine and versions

| | |
|---|---|
| Machine | Apple M4 Pro, 14 cores (10 performance + 4 efficiency), 48 GB RAM |
| OS | macOS 27.0.1 (arm64) |
| deepToolsR | 0.9.0 (source commit `2cca3fa`, native build rebuilt before timing) |
| deepTools | 3.5.6 (pip, Python 3.14) and 4.0.0 (conda) |
| Harness Python | 3.14.7, psutil |

`machine.json` holds the same details as recorded by the harness.

## Data

- **ChIP-seq:** ENCODE HCT116 H3K4me3, ENCSR333OPW replicate 1, file
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

## Method

- **Repeats and statistics:** every configuration is run 3 times. The tables
  give the median with min and max. The figures plot the mean of the n = 3
  runs with a shaded two-sided 95% t-interval:
  mean ± t(0.975, n−1) · sd / √n, with t(0.975, 2) = 4.303. `summary.csv`
  holds both the medians and the means with their intervals.
- **Figures:** line graphs, one panel per benchmark case, with a separate
  wall-time figure and peak-RSS figure for each group. The x axis is the
  processor count (`-p` 1, 4, 8), or the matrix size for plotHeatmap and
  plotProfile. The y axis is log-scaled when a
  panel's range spans more than 10x. A timed-out run is drawn as a hollow,
  labelled marker at the time it was stopped.

- **Wall time:** measured around each process.
- **Peak RSS:** the process tree is sampled every 50 ms with psutil, and RSS is
  summed over the process and all of its children. deepTools 3.5.6 forks
  worker processes, so pages the workers share are counted once per worker.
  Its multi-threaded figures are therefore an upper bound. deepToolsR and
  4.0.0 run as a single process with threads, so this does not apply to them.
- **Quiet machine:** before each run the harness waits until system CPU is
  below 15% and no pytest, pip, cmake or busy VM process is present. A run
  during which one appears is discarded and repeated. During the runs, the VMs
  were shut down and background file sync was paused.
- **Timeout:** 1800 s per run.

### Option differences

- **ChIP bamCoverage:** all three tools use
  `--binSize {50,1} --normalizeUsing CPM -p {1,4,8}`.
- **RNA bamCoverage:** every tool runs at `-p 1`, `-p 4` and `-p 8`, except
  deepTools 3.5.6 at binSize 1, which runs at `-p 8` only (owner decision:
  its two strand runs take too long at lower thread counts). That case is a
  single point in the figures.
  - 3.5.6 and 4.0.0 run twice, with `--filterRNAstrand forward` and then
    `reverse`. Time is the sum of the two runs; RSS is the larger of the two.
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
  above, at `-p 1`, `-p 4` and `-p 8`.
  - deepToolsR: one run with the plus-strand tracks as `-S` and the matching
    minus-strand tracks as `--scoreFileNameMinus`.
  - 3.5.6 and 4.0.0 have no equivalent. They run three commands: computeMatrix
    on the + strand genes with the plus-strand tracks, computeMatrix on the −
    strand genes with the minus-strand tracks, then `computeMatrixOperations
    rbind`. Each tool uses its own rbind (4.0.0 has one). Time is the sum of
    the three commands; RSS is the largest of the three.
  - Row order differs between the two approaches. deepToolsR keeps the
    combined gene order, while rbind puts all + genes before all − genes.
    Parity therefore matches rows by region.
- **plotHeatmap / plotProfile:** deepTools 3.5.6 and 4.0.0 plot
  single-threaded. deepToolsR is run at an explicit `-p 1` and `-p 8`. For
  each size, every tool
  plots the same matrix (deepToolsR's `-p 8` output), with the same figure
  geometry. See "Plot geometry" below.

## Results: bamCoverage, ChIP-seq (CPM)

Wall time in seconds, median (min to max):

| binSize | -p | deepToolsR | deepTools 3.5.6 | deepTools 4.0.0 |
|---:|---:|---:|---:|---:|
| 50 | 1 | **8.8** (8.8–9.2) | 100.3 (100.1–101.1) | 21.4 (21.2–21.5) |
| 50 | 4 | **4.5** (4.5–4.5) | 41.9 (41.0–42.0) | 11.2 (11.2–11.3) |
| 50 | 8 | **4.5** (4.5–4.5) | 31.2 (31.1–31.2) | 9.5 (9.5–9.6) |
| 1 | 1 | **17.1** (17.1–17.1) | 786.0 (779.8–794.1) | 26.9 (26.8–27.7) |
| 1 | 4 | **9.6** (9.6–9.6) | 250.0 (244.1–253.6) | 14.8 (14.6–14.9) |
| 1 | 8 | **9.6** (9.6–9.7) | 161.4 (157.4–162.6) | 12.9 (12.7–13.0) |

Peak RSS of the process tree in MB, median (min to max):

| binSize | -p | deepToolsR | deepTools 3.5.6 | deepTools 4.0.0 |
|---:|---:|---:|---:|---:|
| 50 | 1 | **267** (250–292) | 489 (461–499) | 466 (412–475) |
| 50 | 4 | **301** (238–322) | 700 (687–702) | 609 (582–618) |
| 50 | 8 | **294** (264–317) | 1,255 (1,252–1,257) | 796 (716–814) |
| 1 | 1 | **397** (396–400) | 872 (861–884) | 1,986 (1,985–1,991) |
| 1 | 4 | **485** (416–529) | 2,131 (2,104–2,155) | 3,671 (3,235–3,857) |
| 1 | 8 | **424** (411–428) | 3,460 (3,214–3,483) | 5,446 (4,888–7,106) |

Figures: `figures/bamcov_chip_wall.{png,svg}` and
`figures/bamcov_chip_rss.{png,svg}`.

Reading the tables:

- **Speed:** deepToolsR is 11x faster than 3.5.6 at binSize 50 (`-p 1`) and
  46x faster at binSize 1 (`-p 1`). It is 2.1–2.4x faster than 4.0.0 at
  binSize 50 and 1.3–1.6x faster at binSize 1.
- **Scaling:** deepToolsR's wall time does not improve from `-p 4` to `-p 8`
  on this BAM (reading the file is the limit).
- **Memory:** deepToolsR stays under 530 MB in every configuration. At
  binSize 1, 4.0.0's memory grows with the thread count, reaching 5.4 GB
  median and 7.1 GB max at `-p 8`.

### Output parity (chr19, chr21, chr22, every base; `-p 8` outputs)

| Comparison | binSize 50 | binSize 1 |
|---|---|---|
| deepToolsR vs 3.5.6 | max abs diff 4.8e-6 (float32 rounding) | max abs diff 4.6e-6 |
| deepToolsR vs 4.0.0 | max abs diff 0.005 | max abs diff 0.005 |

The small, widespread difference against 4.0.0 is consistent with 4.0.0
computing the CPM scale factor slightly differently. 3.5.6 and deepToolsR
agree.

## Results: bamCoverage, stranded RNA (both strands)

deepToolsR writes both strands in one run. 3.5.6 and 4.0.0 need two runs;
their time is the sum of the two runs and their RSS the larger of the two.
3.5.6 at binSize 1 was run at `-p 8` only.

Wall time in seconds, median (min to max):

| binSize | -p | deepToolsR | deepTools 3.5.6 | deepTools 4.0.0 |
|---:|---:|---:|---:|---:|
| 10 | 1 | **19.6** (19.5–19.7) | 1,454 (1,441–1,459) | 93.2 (92.6–94.2) |
| 10 | 4 | **8.7** (8.6–8.7) | 438 (426–445) | 36.1 (35.1–37.8) |
| 10 | 8 | **8.5** (8.5–9.5) | 257 (250–264) | 25.1 (24.8–26.1) |
| 1 | 1 | **31.5** (31.3–32.4) | not run | 99.4 (99.4–99.9) |
| 1 | 4 | **16.4** (16.4–16.4) | not run | 41.0 (39.7–43.3) |
| 1 | 8 | **16.2** (16.2–16.3) | 496 (481–497) | 30.8 (30.1–30.9) |

Peak RSS in MB, median (min to max):

| binSize | -p | deepToolsR | deepTools 3.5.6 | deepTools 4.0.0 |
|---:|---:|---:|---:|---:|
| 10 | 1 | **353** (298–361) | 496 (490–497) | 984 (970–1,004) |
| 10 | 4 | **412** (291–418) | 1,200 (1,168–1,279) | 1,140 (1,102–1,182) |
| 10 | 8 | **391** (298–429) | 2,322 (2,317–2,352) | 1,478 (1,466–1,540) |
| 1 | 1 | **576** (528–578) | not run | 2,046 (2,040–2,087) |
| 1 | 4 | **560** (554–580) | not run | 3,564 (3,478–3,600) |
| 1 | 8 | **582** (579–597) | 3,588 (3,515–3,607) | 5,252 (5,031–5,547) |

**Parity** (chr19, chr21, chr22, every base; `-p 8`): the deepToolsR
`.plus`/`.minus` outputs are identical (max difference 0) to both the 3.5.6
and the 4.0.0 `--filterRNAstrand forward`/`reverse` outputs, at binSize 10
and binSize 1.

Figures: `figures/rna_wall.*` and `figures/rna_rss.*`.

## Results: computeMatrix

Wall time in seconds, median (min to max):

| Size | -p | deepToolsR | deepTools 3.5.6 | deepTools 4.0.0 |
|---|---:|---:|---:|---:|
| small | 1 | **0.7** (0.6–0.9) | 19.2 (19.2–19.3) | 0.8 (0.8–0.9) |
| small | 4 | **0.4** (0.4–0.4) | 6.1 (6.0–6.1) | 0.8 (0.8–0.8) |
| small | 8 | **0.3** (0.3–0.3) | 3.7 (3.7–3.8) | 0.8 (0.8–0.8) |
| medium | 1 | **1.8** (1.8–2.0) | 60.4 (60.2–60.7) | 2.4 (2.3–2.4) |
| medium | 4 | **0.7** (0.7–0.7) | 18.3 (18.3–18.5) | 2.3 (2.3–2.4) |
| medium | 8 | **0.5** (0.5–0.5) | 11.2 (11.1–11.3) | 2.3 (2.3–2.4) |
| big | 1 | **35.8** (35.5–37.4) | 876 (867–905) | 49.2 (49.1–50.0) |
| big | 4 | **10.4** (10.4–10.5) | 268 (265–270) | 20.1 (20.1–20.4) |
| big | 8 | **5.5** (5.5–5.6) | 171 (171–172) | 20.3 (20.2–20.5) |

Peak RSS in MB, median (min to max):

| Size | -p | deepToolsR | deepTools 3.5.6 | deepTools 4.0.0 |
|---|---:|---:|---:|---:|
| small | 1 | **66** (66–68) | 119 (118–120) | 150 (150–151) |
| small | 4 | **64** (64–65) | 412 (407–412) | 151 (148–151) |
| small | 8 | **88** (86–90) | 668 (664–674) | 152 (151–153) |
| medium | 1 | **96** (96–97) | 239 (237–239) | 424 (415–432) |
| medium | 4 | **106** (106–108) | 512 (508–516) | 433 (431–436) |
| medium | 8 | **120** (119–126) | 796 (792–800) | 418 (412–441) |
| big | 1 | **850** (835–864) | 2,500 (2,498–2,505) | 1,498 (1,491–1,606) |
| big | 4 | **850** (845–851) | 2,412 (2,352–2,722) | 1,577 (1,577–1,620) |
| big | 8 | **878** (872–878) | 3,582 (3,428–3,683) | 1,574 (1,569–1,596) |

4.0.0's computeMatrix time is flat from `-p 1` to `-p 8` for small and
medium, and flat from `-p 4` to `-p 8` for big. deepToolsR keeps getting
faster up to `-p 8`.

Figures: `figures/matrix_wall.*` and `figures/matrix_rss.*`.

## Results: computeMatrix, stranded

Stranded big was run at `-p 8` only (owner decision). The three deepToolsR
stranded-big runs at `-p 1` that were recorded before that decision are kept
in `runs.json` and `summary.csv`, but are not shown in the table or figures.

| Size | -p | Wall (s): deepToolsR | 3.5.6 | 4.0.0 | RSS (MB): deepToolsR | 3.5.6 | 4.0.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| medium | 1 | **2.0** | 67.3 | 6.5 | **100** | 210 | 234 |
| medium | 4 | **0.8** | 23.8 | 6.5 | **116** | 444 | 233 |
| medium | 8 | **0.5** | 16.2 | 6.5 | **122** | 713 | 235 |
| big | 8 | **6.2** | 236 | 82.0 | **963** | 2,965 | 1,686 |

The values are medians; min and max are in `summary.csv`. For 3.5.6 and
4.0.0 the time covers all three commands (two computeMatrix runs and rbind),
and RSS is the largest of the three.

Figures: `figures/matrix_stranded_wall.*` and `figures/matrix_stranded_rss.*`.

## Results: plotHeatmap and plotProfile

deepTools 3.5.6 and 4.0.0 plot single-threaded. deepToolsR was run at `-p 1`
and at `-p 8`. For each size, every tool plots the same matrix
(deepToolsR's `-p 8` computeMatrix output) at identical pixel geometry.

Wall time in seconds, median (min to max):

| Plot | Size | deepToolsR -p 1 | deepToolsR -p 8 | deepTools 3.5.6 | deepTools 4.0.0 |
|---|---|---:|---:|---:|---:|
| plotHeatmap | small | 0.55 (0.55–0.75) | **0.48** (0.48–0.48) | 0.74 (0.74–0.74) | 0.60 (0.60–0.67) |
| plotHeatmap | medium | 0.81 (0.80–0.81) | **0.55** (0.55–0.60) | 1.39 (1.39–1.40) | 1.13 (1.12–1.13) |
| plotHeatmap | big | 5.78 (5.70–5.84) | **2.15** (2.15–2.16) | 12.72 (12.72–12.74) | 9.83 (9.79–9.90) |
| plotProfile | small | 0.47 (0.42–0.49) | **0.41** (0.41–0.47) | 0.54 (0.54–0.54) | 0.54 (0.54–0.54) |
| plotProfile | medium | 0.54 (0.54–0.55) | **0.48** (0.48–0.48) | 0.86 (0.86–0.87) | 0.87 (0.86–0.87) |
| plotProfile | big | 2.67 (2.66–2.72) | **1.46** (1.39–1.46) | 6.25 (6.24–6.27) | 6.27 (6.25–6.28) |

Peak RSS in MB, median:

| Plot | Size | deepToolsR -p 1 | deepToolsR -p 8 | deepTools 3.5.6 | deepTools 4.0.0 |
|---|---|---:|---:|---:|---:|
| plotHeatmap | small | **177** | 193 | 290 | 287 |
| plotHeatmap | medium | 217 | **215** | 790 | 756 |
| plotHeatmap | big | **680** | 910 | 4,196 | 4,204 |
| plotProfile | small | **110** | 112 | 144 | 141 |
| plotProfile | medium | **133** | 135 | 308 | 310 |
| plotProfile | big | 451 | **450** | 1,684 | 1,708 |

Single-threaded against single-threaded, deepToolsR plotHeatmap is 2.2x
faster than 3.5.6 on the big matrix, and plotProfile is 2.3x faster. 3.5.6
and 4.0.0 are nearly identical here: 4.0.0 keeps the Python (matplotlib)
plotting code, and only its computational core is in Rust.

**Note on thread counts.** deepToolsR's plotting tools default to
`-p auto`, which means half the logical CPUs: 7 threads on this machine. An
earlier version of this benchmark ran deepToolsR's plots without `-p` and
labelled them single-threaded, when they actually used 7 threads. Those
records were replaced by the explicit `-p 1` and `-p 8` runs above. The old
records are kept outside the repository. The harness now always passes `-p`
to every deepToolsR command.

Figures: `figures/plot_wall.*` and `figures/plot_rss.*`.

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

The first plot benchmark used default figure settings, which were not
comparable. Its 54 records were replaced, and the old records are kept
outside the repository. The deepToolsR `-p 1` and `-p 8` runs use the same
calibrated sizes as the deepToolsR rows in the geometry table. Their panel
sizes were re-measured: 1,691 × 250 px for the big heatmap and 382 × 751 px
for the small profile, identical at both thread counts.

Figures: `figures/plot_wall.*` and `figures/plot_rss.*`.

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

```sh
python benchmarks/release/bench.py run \
    --chip-bam DATA/ENCFF772IUE.bam --rna-bam RNA.bam \
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
with the missing work: the rest of the RNA benchmarks, then computeMatrix, stranded computeMatrix,
then the plots. `--only rna matrix matrix_stranded plot` restricts it to those groups. After
that, rerun `parity` and `report`.
