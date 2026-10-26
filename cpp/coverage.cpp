// Native BAM coverage, filtering, normalization and bounded-window output.
// The production writer is shared by all coverage operations.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "parallel.hpp"
#include "numeric.hpp"

#include <cstdio>

#include <htslib/bgzf.h>
#include <htslib/cram.h>
#include <htslib/hfile.h>
#include <htslib/hts.h>
#include <htslib/sam.h>
#include <libdeflate.h>

#include "libBigWig/bigWig.h"
#define DTP_HTSLIB_HANDLES
#include "handles.hpp"
#include "display_path.hpp"
#include "streaming.hpp"
#include "writer_common.hpp"

namespace py = pybind11;

namespace {

using dtp::OutFmt;
using dtp::StreamWriter;
using dtp::Window;
using dtp::run_windows;
using dtp::validate_writer_options;
using dtp::checked_float;
using dtp::HtsFilePtr;
using dtp::open_alignments;
using dtp::HdrPtr;
using dtp::IdxPtr;
using dtp::BamRecordPtr;
using dtp::IteratorPtr;
using dtp::BigWigOutput;
using dtp::TempFile;
using dtp::FilePtr;

constexpr uint32_t MAX_WINDOW_BINS = 1u << 18;
constexpr uint64_t TARGET_WINDOW_BP = 1u << 20;

inline void poll_coverage_work(uint64_t count) {
    if ((count & 4095u) != 0) return;
    if (dtp::running_in_parallel_worker()) {
        if (dtp::interruption_requested())
            throw std::runtime_error("coverage computation cancelled");
    } else {
        dtp::check_python_signals();
    }
}

uint32_t coverage_window_bins(uint32_t bin_size, int requested = 0) {
    if (bin_size == 0)
        throw std::runtime_error("binSize must be > 0");
    if (requested < 0)
        throw std::runtime_error("window_size must be >= 0");
    const uint64_t bp_limited = std::max<uint64_t>(
        1, TARGET_WINDOW_BP / static_cast<uint64_t>(bin_size));
    const uint64_t requested_limit = requested > 0
        ? static_cast<uint64_t>(requested) : MAX_WINDOW_BINS;
    return static_cast<uint32_t>(std::max<uint64_t>(
        1, std::min<uint64_t>({MAX_WINDOW_BINS, bp_limited,
                               requested_limit})));
}

// Version string of the vendored htslib -- a trivial proof the library is
// actually linked and callable.
std::string htslib_version() {
    return std::string(hts_version());
}

inline bool readable_file(const std::string &path) {
    std::FILE *handle = std::fopen(path.c_str(), "rb");
    if (!handle) return false;
    std::fclose(handle);
    return true;
}

inline std::string replace_extension(const std::string &path,
                                     const std::string &extension) {
    const std::size_t slash = path.find_last_of("/\\");
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos ||
        (slash != std::string::npos && dot < slash))
        return path + extension;
    return path.substr(0, dot) + extension;
}

// Resolve the local sidecar using the same candidate order as the selected
// backend, chosen by the detected format (not the file name). The CLI includes
// this exact file in its input/output alias check before any writer is opened.
std::string resolve_bam_index_path(const std::string &bam_path) {
    std::vector<std::string> candidates;
    const HtsFilePtr file(readable_file(bam_path) ? hts_open(bam_path.c_str(), "r") : nullptr);
    if (file && hts_get_format(file.get())->format == cram) {
        candidates = {bam_path + ".crai",
                      replace_extension(bam_path, ".crai")};
    } else {
        candidates = {bam_path + ".csi", replace_extension(bam_path, ".csi"),
                      bam_path + ".bai", replace_extension(bam_path, ".bai")};
    }
    for (const auto &candidate : candidates)
        if (readable_file(candidate)) return candidate;
    return "";
}

using dtp::display_path;

// An open BAM/CRAM/SAM(.gz) with its header and, when `indexed`, its index.
// Every handle that iterates loads its own index: a CRAM index sets the
// region on the handle that loaded it, so one index cannot serve two handles.
// `reference` is the indexed FASTA a CRAM decodes against ("" lets htslib
// resolve it from the header UR/M5 tags, REF_PATH and REF_CACHE).
struct Alignments {
    HtsFilePtr file;
    HdrPtr hdr;
    IdxPtr idx;
    std::string path, reference;
};

Alignments open_reads(const std::string &path, const std::string &reference, bool indexed) {
    Alignments a{{}, {}, {}, path, reference};
    a.file = open_alignments(path, "could not open BAM/CRAM: " + display_path(path));
    if (a.file.get()->format.category != sequence_data)
        throw std::runtime_error("not a sequence (BAM/SAM/CRAM) file: " + display_path(path));
    if (a.file.get()->format.format == cram) {
        // Decode only the fields coverage reads: without SEQ/QUAL htslib needs
        // no reference except for multi-reference slices (TLEN decoding).
        hts_set_opt(a.file.get(), CRAM_OPT_REQUIRED_FIELDS,
                    SAM_QNAME | SAM_FLAG | SAM_RNAME | SAM_POS | SAM_MAPQ | SAM_CIGAR |
                    SAM_RNEXT | SAM_PNEXT | SAM_TLEN | SAM_RGAUX);
        if (!reference.empty() && hts_set_fai_filename(a.file.get(), reference.c_str()) != 0)
            throw std::runtime_error("could not load the reference FASTA: " + display_path(reference));
    }
    a.hdr.reset(sam_hdr_read(a.file.get()));
    if (!a.hdr.get()) throw std::runtime_error("could not read header: " + display_path(path));
    if (indexed) {
        a.idx.reset(sam_index_load(a.file.get(), path.c_str()));
        if (!a.idx.get())
            throw std::runtime_error(
                "no usable index for '" + display_path(path) +
                "'. Coordinate-sort and index the file (samtools index).");
    }
    return a;
}

// A read status below -1 is a decoding failure.
int checked_read(int status) {
    if (status >= -1) return status;
    throw std::runtime_error(
        "error reading BAM/CRAM data (truncated or corrupt input, or a CRAM whose "
        "reference is needed but not found: pass --referenceFasta FILE)");
}

// References a .bai/.csi lists statistics for (htslib reads past the index
// for a target it does not list); 0 for CRAI, which stores none.
int indexed_references(const Alignments &a) {
    return hts_idx_fmt(a.idx.get()) == HTS_FMT_CRAI ? 0 : hts_idx_nseq(a.idx.get());
}

// A damaged or foreign index can load and still return no reads where the
// file has them (htslib then yields an empty track). Probe each reference the
// index counts reads on, and the reference of the file's first placed record
// (CRAI has no counts), and require the index to find a read there.
std::runtime_error index_mismatch(const std::string &path, const std::string &why) {
    return std::runtime_error("the index of '" + display_path(path) +
                              "' does not match the file (" + why +
                              "); re-index it (samtools index)");
}

// htslib loops forever querying a bin id it cannot place (one above INT_MAX
// in a damaged .bai/.csi), so read the loaded index file itself and reject bin
// ids the format cannot hold or counts that run past its end. CRAI has no bins.
void check_index_bins(const std::string &path) {
    const std::size_t delim = path.find(HTS_IDX_DELIM);
    const std::string index = delim == std::string::npos
        ? resolve_bam_index_path(path) : path.substr(delim + std::strlen(HTS_IDX_DELIM));
    BGZF *opened = index.empty() ? nullptr : bgzf_open(index.c_str(), "r");
    if (!opened) return;
    std::unique_ptr<BGZF, int (*)(BGZF *)> fp(opened, bgzf_close);
    uint8_t buffer[4096];
    const auto skip = [&](uint64_t n) {
        while (n) {
            const std::size_t step = std::min<uint64_t>(n, sizeof buffer);
            if (bgzf_read(fp.get(), buffer, step) != static_cast<ssize_t>(step))
                throw index_mismatch(path, "it ends early");
            n -= step;
        }
    };
    const auto u32 = [&] {
        skip(4);
        return uint32_t(buffer[0]) | uint32_t(buffer[1]) << 8 |
               uint32_t(buffer[2]) << 16 | uint32_t(buffer[3]) << 24;
    };
    skip(4);
    const bool csi = std::memcmp(buffer, "CSI\1", 4) == 0;
    if (!csi && std::memcmp(buffer, "BAI\1", 4) != 0) return;
    uint64_t max_bin = 37449;
    if (csi) {
        u32();  // min_shift
        const uint32_t depth = u32();
        if (depth > 19) throw index_mismatch(path, "its depth is impossible");
        max_bin = ((uint64_t(1) << ((depth + 1) * 3)) - 1) / 7;
        skip(u32());  // aux
    }
    for (uint32_t refs = u32(); refs; --refs) {
        for (uint32_t bins = u32(); bins; --bins) {
            const uint32_t bin = u32();
            if (bin > max_bin + 1)
                throw index_mismatch(path, "bin id " + std::to_string(bin) + " is impossible");
            if (csi) skip(8);  // loffset
            skip(uint64_t(u32()) * 16);
        }
        if (!csi) skip(uint64_t(u32()) * 8);
    }
}

void check_index(Alignments &a, const std::string &path) {
    check_index_bins(path);
    const int n = sam_hdr_nref(a.hdr.get());
    const auto mismatch = [&](const std::string &why) { return index_mismatch(path, why); };
    const int listed = indexed_references(a);
    if (listed > n) throw mismatch("it lists more references than the header");
    std::vector<int> probes;
    for (int tid = 0; tid < listed; ++tid) {
        uint64_t mapped = 0, unmapped = 0;
        if (hts_idx_get_stat(a.idx.get(), tid, &mapped, &unmapped) >= 0 && mapped + unmapped)
            probes.push_back(tid);
    }
    BamRecordPtr record(bam_init1());
    if (!record.get()) throw std::runtime_error("could not allocate a BAM record");
    // Only placement is needed: a CRAM then skips sequence and reference decoding.
    hts_set_opt(a.file.get(), CRAM_OPT_REQUIRED_FIELDS, SAM_FLAG | SAM_RNAME | SAM_POS);
    if (checked_read(sam_read1(a.file.get(), a.hdr.get(), record.get())) >= 0 &&
        record.get()->core.tid >= 0 && record.get()->core.tid < n)
        probes.push_back(record.get()->core.tid);
    std::sort(probes.begin(), probes.end());
    probes.erase(std::unique(probes.begin(), probes.end()), probes.end());
    for (int tid : probes) {
        IteratorPtr it(sam_itr_queryi(a.idx.get(), tid, 0, sam_hdr_tid2len(a.hdr.get(), tid)));
        if (!it.get() || sam_itr_next(a.file.get(), it.get(), record.get()) < 0)
            throw mismatch(std::string("it finds no reads on ") +
                           sam_hdr_tid2name(a.hdr.get(), tid));
    }
    hts_set_opt(a.file.get(), CRAM_OPT_REQUIRED_FIELDS, INT_MAX);
}

// Mapped and unmapped records per reference plus the unplaced count, from the
// index statistics (.bai/.csi). CRAI stores none, so a CRAM is counted by one
// pass that decodes only the flag and reference fields.
struct ReadCounts {
    std::vector<uint64_t> mapped, unmapped;
    uint64_t no_coor = 0;
};

ReadCounts read_counts(Alignments &a, const std::string &path) {
    const int n = sam_hdr_nref(a.hdr.get());
    ReadCounts counts{std::vector<uint64_t>(n), std::vector<uint64_t>(n)};
    if (hts_idx_fmt(a.idx.get()) != HTS_FMT_CRAI) {
        for (int tid = 0; tid < std::min(n, indexed_references(a)); ++tid)
            // -1: no index metadata for a target without reads
            if (hts_idx_get_stat(a.idx.get(), tid, &counts.mapped[tid], &counts.unmapped[tid]) < 0)
                counts.mapped[tid] = counts.unmapped[tid] = 0;
        counts.no_coor = hts_idx_get_n_no_coor(a.idx.get());
        return counts;
    }
    Alignments scan = open_reads(a.path, a.reference, false);
    hts_set_opt(scan.file.get(), CRAM_OPT_REQUIRED_FIELDS, SAM_FLAG | SAM_RNAME);
    BamRecordPtr record(bam_init1());
    if (!record.get()) throw std::runtime_error("could not allocate a BAM record");
    while (checked_read(sam_read1(scan.file.get(), scan.hdr.get(), record.get())) >= 0) {
        const int tid = record.get()->core.tid;
        if (tid < 0 || tid >= n) ++counts.no_coor;
        else if (record.get()->core.flag & BAM_FUNMAP) ++counts.unmapped[tid];
        else ++counts.mapped[tid];
    }
    return counts;
}

// Per-reference read counts (see read_counts). Returns one dict per target plus
// the count of unplaced reads. This is the data the corrected normalization
// denominator is built from (mapped-alignment count per target, honoring
// --ignoreForNormalization by dropping chromosomes here).
py::dict bam_index_stats(const std::string &path, const std::string &reference_fasta) {
    Alignments a = open_reads(path, reference_fasta, true);
    check_index(a, path);
    const ReadCounts counts = read_counts(a, path);
    const int n = sam_hdr_nref(a.hdr.get());
    py::list targets;
    uint64_t total_mapped = 0;
    uint64_t total_unmapped = 0;
    for (int tid = 0; tid < n; ++tid) {
        total_mapped += counts.mapped[tid];
        total_unmapped += counts.unmapped[tid];
        py::dict t;
        t["name"] = std::string(sam_hdr_tid2name(a.hdr.get(), tid));
        t["length"] = static_cast<uint64_t>(sam_hdr_tid2len(a.hdr.get(), tid));
        t["mapped"] = counts.mapped[tid];
        t["unmapped"] = counts.unmapped[tid];
        targets.append(std::move(t));
    }

    py::dict out;
    out["targets"] = targets;
    out["n_targets"] = n;
    out["total_mapped"] = total_mapped;      // NB: includes secondary/supplementary
    out["total_unmapped"] = total_unmapped;
    out["n_no_coor"] = counts.no_coor;
    return out;
}

// --- P1: single-thread binned coverage -------------------------------------
//
// Coverage is built genomecov-style: a difference-array sweep over each read's
// *aligned blocks* (CIGAR M/=/X; D and N gap the block, matching pysam
// get_blocks and Bioconductor coverage()), giving true per-base depth. Bins
// aggregate that depth:
//   mean  = sum(depth over bin) / bin_width   (Bioconductor binnedAverage /
//           tileGenome; == deeptools at binSize 1)   [default]
//   sum   = sum(depth over bin)               (total coverage in the bin)
//   count = #reads overlapping the bin, deduped per read (bamCoverage parity)
//
// mean/sum use the O(1)-per-block bin-resolution accumulator below (no per-base
// array): edge bins get their partial overlap, fully-covered interior bins get
// binSize via a bin-level range delta. count uses a per-read bin-dedup delta.

enum class Agg { Mean, Sum, Count };

// SAM FLAG bits, named per the SAM spec / Picard "explain flags"
// (broadinstitute.github.io/picard/explain-flags.html). Own constants rather
// than htslib's BAM_F* so the filtering code reads self-documenting.
constexpr uint16_t SAM_READ_PAIRED    = 0x001;  // read paired
constexpr uint16_t SAM_PROPER_PAIR    = 0x002;  // read mapped in proper pair
constexpr uint16_t SAM_READ_UNMAPPED  = 0x004;  // read unmapped
constexpr uint16_t SAM_MATE_UNMAPPED  = 0x008;  // mate unmapped
constexpr uint16_t SAM_READ_REVERSE   = 0x010;  // read reverse strand
constexpr uint16_t SAM_MATE_REVERSE   = 0x020;  // mate reverse strand
constexpr uint16_t SAM_FIRST_IN_PAIR  = 0x040;  // first in pair (read1)
constexpr uint16_t SAM_SECOND_IN_PAIR = 0x080;  // second in pair (read2)
constexpr uint16_t SAM_SECONDARY      = 0x100;  // not primary alignment
constexpr uint16_t SAM_QC_FAIL        = 0x200;  // fails platform/vendor QC
constexpr uint16_t SAM_DUPLICATE      = 0x400;  // PCR or optical duplicate
constexpr uint16_t SAM_SUPPLEMENTARY  = 0x800;  // supplementary alignment

// --samFlagInclude: (bits that must be set, bits that must be unset) patterns.
// A read passes when the list is empty or it matches any pattern.
using FlagPatterns = std::vector<std::pair<uint16_t, uint16_t>>;
inline bool flag_included(uint16_t flag, const FlagPatterns &patterns) {
    if (patterns.empty()) return true;
    for (const auto &[set, unset] : patterns)
        if ((flag & set) == set && !(flag & unset)) return true;
    return false;
}

// Library strandedness: which read of a pair is sense to the transcript.
//   Forward = second/fr/secondstrand: read1 sense.
//   Reverse = first/rf/firststrand (dUTP): read2 sense.
enum class Strand { None, Forward, Reverse };
// Which transcript strand to output.
enum class StrandFilter { None, Forward, Reverse };
// Default flag policy: Primary excludes secondary+supplementary; Deeptools keeps
// stock behaviour (skip only unmapped).
enum class FilterMode { Primary, Deeptools };
// Collapse a fragment to one end / its centre (single base by default).
enum class Collapse { None, FivePrime, ThreePrime, Center };
// Orient the fragment-touch test (P8) by strand (needs strandedness).
enum class TouchStrand { Ignore, Sense, Antisense };

// Transcript strand of a read (+1 forward / -1 reverse) given library type.
inline int transcript_strand(const bam1_t *b, Strand s) {
    const uint16_t f = b->core.flag;
    const int read_sign = (f & SAM_READ_REVERSE) ? -1 : +1;
    const bool is_read2 = (f & SAM_READ_PAIRED) && (f & SAM_SECOND_IN_PAIR);
    if (s == Strand::Forward)  // read1 sense
        return is_read2 ? -read_sign : read_sign;
    // Strand::Reverse -- read2 sense (dUTP)
    return is_read2 ? read_sign : -read_sign;
}

// Extract aligned blocks [start,end) from a BAM record's CIGAR. D/N gap the
// block; query-only ops (I/S/H/P) do not advance the reference.
inline void extract_blocks(const bam1_t *b,
                           std::vector<std::pair<int64_t, int64_t>> &out) {
    out.clear();
    const uint32_t *cig = bam_get_cigar(b);
    const int n = b->core.n_cigar;
    int64_t pos = b->core.pos;
    int64_t bstart = 0, bend = 0;
    bool open = false;
    for (int k = 0; k < n; ++k) {
        const int op = bam_cigar_op(cig[k]);
        const int64_t len = bam_cigar_oplen(cig[k]);
        if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
            if (!open) { bstart = pos; open = true; }
            bend = pos + len;
            pos += len;
        } else if (op == BAM_CDEL || op == BAM_CREF_SKIP) {
            if (open) { out.emplace_back(bstart, bend); open = false; }
            pos += len;
        }
        // BAM_CINS / BAM_CSOFT_CLIP / BAM_CHARD_CLIP / BAM_CPAD: no ref advance.
    }
    if (open) out.emplace_back(bstart, bend);
}

// A read's introns: its CIGAR N gaps (deletions are not introns).
inline void extract_introns(const bam1_t *b,
                            std::vector<std::pair<int64_t, int64_t>> &out) {
    out.clear();
    const uint32_t *cig = bam_get_cigar(b);
    int64_t pos = b->core.pos;
    for (uint32_t k = 0; k < b->core.n_cigar; ++k) {
        const int op = bam_cigar_op(cig[k]);
        const int64_t len = bam_cigar_oplen(cig[k]);
        if (op == BAM_CREF_SKIP) out.emplace_back(pos, pos + len);
        if (bam_cigar_type(op) & 2) pos += len;
    }
}

// Removes the start-sorted `gaps` from the start-sorted, disjoint `iv`.
inline void subtract_gaps(std::vector<std::pair<int64_t, int64_t>> &iv,
                          const std::vector<std::pair<int64_t, int64_t>> &gaps) {
    if (gaps.empty()) return;
    std::vector<std::pair<int64_t, int64_t>> out;
    std::size_t g = 0;
    for (auto [s, e] : iv) {
        while (g < gaps.size() && gaps[g].second <= s) ++g;
        for (std::size_t k = g; k < gaps.size() && gaps[k].first < e; ++k) {
            if (gaps[k].first > s) out.emplace_back(s, gaps[k].first);
            s = std::max(s, gaps[k].second);
        }
        if (e > s) out.emplace_back(s, e);
    }
    iv.swap(out);
}

// Joins the start-sorted `other` into the start-sorted, disjoint `iv`, so an
// overlap of the two is covered once.
inline void join_blocks(std::vector<std::pair<int64_t, int64_t>> &iv,
                        const std::vector<std::pair<int64_t, int64_t>> &other) {
    std::vector<std::pair<int64_t, int64_t>> all(iv.size() + other.size());
    std::merge(iv.begin(), iv.end(), other.begin(), other.end(), all.begin());
    iv.clear();
    for (const auto &block : all) {
        if (!iv.empty() && block.first <= iv.back().second)
            iv.back().second = std::max(iv.back().second, block.second);
        else iv.push_back(block);
    }
}

// All parsed coverage options, shared by the serial and parallel drivers so the
// per-read logic has a single source of truth.
struct FetchBounds {
    int64_t begin = std::numeric_limits<int64_t>::max();
    int64_t end = 0;
    void include(const FetchBounds &other) {
        begin = std::min(begin, other.begin);
        end = std::max(end, other.end);
    }
};

struct CovParams {
    std::string reference;  // indexed FASTA for CRAM decoding, or ""
    uint32_t bin_size;
    Agg agg;
    int min_mapq;
    FlagPatterns include;
    uint16_t exclude;
    int min_frag;
    int max_frag;
    bool extend;
    int extend_len;
    bool skip_introns;  // with extend: never fill an alignment's own introns
    Strand strand;
    StrandFilter sfilter;
    Collapse collapse;
    int collapsed_len;
    bool fragments;  // --countUnit fragment: a pair's mates count once together
    int64_t halo;  // optional extra query padding; never required for correctness
    uint32_t window_bins;
    const std::vector<std::vector<FetchBounds>> *fetch_bounds = nullptr;
    double scale;        // normalization x scaleFactor (stream 0 / plus)
    double scale_minus;  // normalization x scaleFactor x scaleFactorMinus (minus)
    bool rpkm;           // divide read-count output by each bin's actual width
    int nstreams;  // 1, or 2 for --filterRNAstrand split
    bool split;    // route reads to stream 0 (+)/1 (-) by transcript strand
    // Accumulate one stream in two lanes by transcript strand, for a
    // strand-aware black/whitelist mask; the masked lanes are then summed.
    bool strand_lanes;
    bool overlap_filter;  // whitelist/blacklist are fragment-touch filters, not masks
    bool has_blacklist;
    bool has_whitelist;
    TouchStrand touch_strand;  // orient the touch test by strand
};

// Upstream getTLen uses reference-aligned length (including D, excluding N)
// when a single-end/untemplated alignment has TLEN zero.
inline uint64_t fragment_length_for_filter(const bam1_t *rec) {
    const int64_t tlen = rec->core.isize;
    if (tlen != 0)
        return tlen < 0 ? static_cast<uint64_t>(-(tlen + 1)) + 1
                        : static_cast<uint64_t>(tlen);
    uint64_t length = 0;
    const uint32_t *cigar = bam_get_cigar(rec);
    for (uint32_t i = 0; i < rec->core.n_cigar; ++i) {
        const int op = bam_cigar_op(cigar[i]);
        if (op == BAM_CMATCH || op == BAM_CDEL || op == BAM_CEQUAL || op == BAM_CDIFF)
            length += bam_cigar_oplen(cigar[i]);
    }
    return length;
}

inline bool is_proper_fragment(const bam1_t *rec) {
    const uint16_t flag = rec->core.flag;
    if (!(flag & SAM_READ_PAIRED) || !(flag & SAM_PROPER_PAIR) ||
        (flag & SAM_MATE_UNMAPPED) || rec->core.mtid != rec->core.tid ||
        rec->core.mpos < 0 || rec->core.isize == 0)
        return false;
    const bool reverse = flag & SAM_READ_REVERSE;
    const bool mate_reverse = flag & SAM_MATE_REVERSE;
    if (reverse == mate_reverse) return false;
    // Do not trust the aligner's proper-pair bit on its own: a proper fragment
    // must be inward-facing as in stock deepTools' is_proper_pair().  Long
    // genuine fragments remain supported; R deliberately has no hidden
    // maximum insert-size cut-off here.
    return reverse ? rec->core.pos >= rec->core.mpos
                   : rec->core.pos <= rec->core.mpos;
}

// A proper fragment spans [forward mate's aligned start, reverse mate's aligned
// end), whatever the aligner's TLEN sign or soft-clip-inclusive |TLEN|. The
// reverse mate holds both bounds; a forward mate alone extends by |TLEN| as in
// deepTools 3.5.6. Returns false for a non-proper record.
inline bool proper_fragment_bounds(const bam1_t *rec, uint32_t chrom_len,
                                   int64_t &fs, int64_t &fe) {
    if (!is_proper_fragment(rec)) return false;
    if (rec->core.flag & SAM_READ_REVERSE) {
        fs = rec->core.mpos;
        fe = std::min<int64_t>(bam_endpos(rec), chrom_len);
    } else {
        fs = rec->core.pos;
        fe = fs >= chrom_len ? fs : fs + static_cast<int64_t>(std::min<uint64_t>(
            fragment_length_for_filter(rec), chrom_len - fs));
    }
    return true;
}

// A proper-flagged pair whose reverse mate starts left of its forward mate and
// overlaps it (a read-through of a short insert). Only the reverse mate can see
// the overlap; the forward mate learns it from its mate's summary. A pair
// spanning a circular origin does not overlap and stays two single reads.
inline bool is_dovetail(const bam1_t *rec) {
    const uint16_t flag = rec->core.flag;
    if (!(flag & SAM_READ_PAIRED) || !(flag & SAM_PROPER_PAIR) ||
        (flag & SAM_MATE_UNMAPPED) || rec->core.mtid != rec->core.tid ||
        rec->core.mpos < 0 || rec->core.isize == 0)
        return false;
    const bool reverse = flag & SAM_READ_REVERSE;
    if (reverse == bool(flag & SAM_MATE_REVERSE)) return false;
    return reverse ? rec->core.pos < rec->core.mpos && bam_endpos(rec) > rec->core.mpos
                   : rec->core.pos > rec->core.mpos;
}

// pysam.AlignedSegment.infer_query_length(always=False): query-consuming CIGAR
// operations, excluding hard clips: the read length behind the estimated
// --extendReads value.
inline int64_t inferred_query_length(const bam1_t *rec) {
    if (rec->core.l_qseq > 0) return rec->core.l_qseq;
    int64_t length = 0;
    const uint32_t *cigar = bam_get_cigar(rec);
    for (uint32_t i = 0; i < rec->core.n_cigar; ++i) {
        const int op = bam_cigar_op(cigar[i]);
        if (op == BAM_CMATCH || op == BAM_CINS || op == BAM_CSOFT_CLIP ||
            op == BAM_CEQUAL || op == BAM_CDIFF)
            length += bam_cigar_oplen(cigar[i]);
    }
    return length;
}

// Strand-split region set: `all` (strand-agnostic, for masking / touch_strand=ignore)
// plus per-strand merged sets for strand-aware touch. An unstranded feature (strand 0)
// goes into all three, so it matches either fragment orientation.
struct RegionSet {
    std::vector<std::pair<uint32_t, uint32_t>> all, plus, minus;
    bool empty() const { return all.empty(); }
};

// Single-record numerator filters (shared by shape_read and the `filtered`
// normalization-denominator counter).
inline bool passes_filters(const bam1_t *rec, const CovParams &P) {
    const uint16_t flag = rec->core.flag;
    if (flag & P.exclude) return false;
    if (!flag_included(flag, P.include)) return false;
    if (rec->core.qual < P.min_mapq) return false;
    if (P.min_frag > 0 || P.max_frag > 0) {
        const uint64_t length = fragment_length_for_filter(rec);
        if (P.min_frag > 0 && length < static_cast<uint64_t>(P.min_frag)) return false;
        if (P.max_frag > 0 && length > static_cast<uint64_t>(P.max_frag)) return false;
    }
    if (P.sfilter != StrandFilter::None) {
        const int ts = transcript_strand(rec, P.strand);
        if (P.sfilter == StrandFilter::Forward && ts != +1) return false;
        if (P.sfilter == StrandFilter::Reverse && ts != -1) return false;
    }
    return true;
}

// Collapses [fs, fe) to its 5'/3' end (by `sign`) or centre, clamped to the
// span and to [0, limit).
inline void collapse_span(int64_t fs, int64_t fe, int sign, const CovParams &P,
                          int64_t limit, std::vector<std::pair<int64_t, int64_t>> &iv) {
    const int64_t L = P.collapsed_len;
    int64_t a, b;
    if (P.collapse == Collapse::Center) {
        a = (fs + fe) / 2 - L / 2;
        b = a + L;
    } else {
        const bool five_at_left = (sign >= 0);
        const bool anchor_left =
            (P.collapse == Collapse::FivePrime) ? five_at_left : !five_at_left;
        if (anchor_left) { a = fs; b = fs + L; }
        else { a = fe - L; b = fe; }
    }
    a = std::max<int64_t>({a, fs, 0});
    b = std::min({b, fe, limit});
    if (b > a) iv.emplace_back(a, b);
}

// Apply filters + build coverage intervals for one record. Returns the output
// stream index (0 for now; strand-split is P5) or -1 to skip the read. `chrom_len`
// is used to clamp fragment/collapse intervals to the reference.
inline int shape_read(const bam1_t *rec, const CovParams &P, uint32_t chrom_len,
                      std::vector<std::pair<int64_t, int64_t>> &iv,
                      int &read_weight, bool already_eligible = false) {
    const uint16_t flag = rec->core.flag;
    if (!already_eligible && !passes_filters(rec, P)) return -1;

    iv.clear();
    read_weight = 1;
    if (P.collapse != Collapse::None) {
        // Counting reads, each read collapses within its own aligned span;
        // counting fragments, a proper pair collapses within its fragment.
        int64_t fs = rec->core.pos, fe = bam_endpos(rec);
        const bool proper = P.fragments && proper_fragment_bounds(rec, chrom_len, fs, fe);
        if (fe <= fs) return -1;
        // An unstranded proper fragment takes its forward mate's orientation.
        const bool reverse = !proper && (flag & SAM_READ_REVERSE);
        const int sign = (P.strand != Strand::None)
                             ? transcript_strand(rec, P.strand)
                             : (reverse ? -1 : +1);
        collapse_span(fs, fe, sign, P, chrom_len, iv);
    } else if (P.extend) {
        int64_t fs, fe;
        if (!proper_fragment_bounds(rec, chrom_len, fs, fe)) {
            // Any other read becomes the solid fragment from its 5' end, never
            // shorter than its own genomic span.
            const int64_t length = std::max<int64_t>(
                P.extend_len, bam_endpos(rec) - rec->core.pos);
            if (flag & SAM_READ_REVERSE) { fe = bam_endpos(rec); fs = fe - length; }
            else { fs = rec->core.pos; fe = fs + length; }
        }
        if (fs < 0) fs = 0;
        if (fe > chrom_len) fe = chrom_len;
        if (fe > fs) iv.emplace_back(fs, fe);
        if (P.skip_introns) {
            std::vector<std::pair<int64_t, int64_t>> introns;
            extract_introns(rec, introns);
            subtract_gaps(iv, introns);
        }
    } else {
        extract_blocks(rec, iv);
    }
    if (iv.empty()) return -1;
    // Route to a strand lane when splitting or masking by strand; one lane otherwise.
    if (P.split || P.strand_lanes) return (transcript_strand(rec, P.strand) == +1) ? 0 : 1;
    return 0;
}

// Accumulate a read's intervals into a window's local bins [win_bin0, win_bin0+nbins),
// clamping to [ws_bp, we_bp). Mirrors the P1/P2 math, window-local.
inline void accumulate_read(const std::vector<std::pair<int64_t, int64_t>> &iv,
                            const CovParams &P, uint32_t win_bin0, uint32_t nbins,
                            uint32_t ws_bp, uint32_t we_bp,
                            std::vector<int64_t> &acc, std::vector<int64_t> &rng,
                            std::vector<int64_t> &cdiff, int read_weight) {
    const int64_t bs_bin = static_cast<int64_t>(win_bin0);
    if (P.agg == Agg::Count) {
        int64_t last_be = -1;  // exclusive local bin already counted for this read
        for (const auto &blk : iv) {
            int64_t s = blk.first, e = blk.second;
            if (s < ws_bp) s = ws_bp;
            if (e > we_bp) e = we_bp;
            if (e <= s) continue;
            int64_t bs = s / P.bin_size - bs_bin;
            int64_t be = (e - 1) / P.bin_size + 1 - bs_bin;
            if (bs < last_be) bs = last_be;
            if (bs >= be) continue;
            cdiff[bs] += read_weight;
            cdiff[be] -= read_weight;
            last_be = be;
        }
    } else {
        for (const auto &blk : iv) {
            int64_t s = blk.first, e = blk.second;
            if (s < ws_bp) s = ws_bp;
            if (e > we_bp) e = we_bp;
            if (e <= s) continue;
            const int64_t fs = s / P.bin_size - bs_bin;
            const int64_t fe = (e - 1) / P.bin_size - bs_bin;
            if (fs == fe) {
                acc[fs] += (e - s) * read_weight;
            } else {
                acc[fs] += ((fs + bs_bin + 1) * static_cast<int64_t>(P.bin_size) - s) * read_weight;
                acc[fe] += (e - (fe + bs_bin) * static_cast<int64_t>(P.bin_size)) * read_weight;
                if (fe - 1 >= fs + 1) {
                    rng[fs + 1] += int64_t(P.bin_size) * read_weight;
                    rng[fe] -= int64_t(P.bin_size) * read_weight;
                }
            }
        }
    }
    (void)nbins;
}

// Does any aligned block overlap the (merged, disjoint, start-sorted) region set?
// O(log n) per block. Splice-aware: caller passes CIGAR-derived blocks.
inline bool touches(const std::vector<std::pair<int64_t, int64_t>> &blocks,
                    const std::vector<std::pair<uint32_t, uint32_t>> &regions) {
    if (regions.empty()) return false;
    for (const auto &b : blocks) {
        // First region with start >= block.end; the candidate (strictly start <
        // block.end) is the one before it. Half-open, so a region touching at the
        // block boundary does not overlap.
        auto it = std::lower_bound(
            regions.begin(), regions.end(), static_cast<uint32_t>(b.second),
            [](const std::pair<uint32_t, uint32_t> &r, uint32_t v) { return r.first < v; });
        if (it != regions.begin()) {
            --it;
            if (static_cast<int64_t>(it->second) > b.first) return true;  // overlap
        }
    }
    return false;
}

// Merge overlapping/adjacent intervals into a disjoint, start-sorted set.
inline void merge_intervals(std::vector<std::pair<uint32_t, uint32_t>> &v) {
    if (v.empty()) return;
    std::sort(v.begin(), v.end());
    std::vector<std::pair<uint32_t, uint32_t>> out;
    out.push_back(v[0]);
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i].first <= out.back().second) {
            if (v[i].second > out.back().second) out.back().second = v[i].second;
        } else {
            out.push_back(v[i]);
        }
    }
    v.swap(out);
}

int resolve_header_tid(sam_hdr_t *header, const std::string &chromosome) {
    const int exact = sam_hdr_name2tid(header, chromosome.c_str());
    if (exact >= 0) return exact;
    std::string alternate;
    if (chromosome.compare(0, 3, "chr") == 0) {
        alternate = chromosome.substr(3);
        if (alternate == "M") alternate = "MT";
    } else {
        alternate = chromosome == "MT" ? "chrM" : "chr" + chromosome;
    }
    return sam_hdr_name2tid(header, alternate.c_str());
}

// Parse a {chrom: [(start,end[,strand]), ...]} dict into per-tid strand-split,
// merged RegionSets. strand: +1 -> {all,plus}; -1 -> {all,minus}; 0/absent ->
// {all,plus,minus} (unstranded feature matches either orientation).
inline std::vector<RegionSet> parse_region_sets(py::object obj, sam_hdr_t *hdr,
                                                int n_targets) {
    std::vector<RegionSet> sets(n_targets);
    if (obj.is_none()) return sets;
    for (auto item : obj.cast<py::dict>()) {
        const std::string chrom = py::str(item.first).cast<std::string>();
        const int tid = resolve_header_tid(hdr, chrom);
        if (tid < 0) continue;
        for (auto iv : item.second.cast<py::list>()) {
            auto t = iv.cast<py::tuple>();
            const uint32_t s = t[0].cast<uint32_t>(), e = t[1].cast<uint32_t>();
            const int strand = t.size() >= 3 ? t[2].cast<int>() : 0;
            if (e <= s) continue;
            sets[tid].all.emplace_back(s, e);
            if (strand >= 0) sets[tid].plus.emplace_back(s, e);
            if (strand <= 0) sets[tid].minus.emplace_back(s, e);
        }
    }
    for (auto &rs : sets) {
        merge_intervals(rs.all); merge_intervals(rs.plus); merge_intervals(rs.minus);
    }
    return sets;
}

// Choose which region vector to test a fragment against, given the touch-strand
// mode and the fragment's transcript strand (+1/-1).
inline const std::vector<std::pair<uint32_t, uint32_t>> &
pick_regions(const RegionSet &rs, TouchStrand ts, int fstrand) {
    if (ts == TouchStrand::Ignore) return rs.all;
    const bool sense = (ts == TouchStrand::Sense);
    const bool want_plus = sense ? (fstrand > 0) : (fstrand <= 0);
    return want_plus ? rs.plus : rs.minus;
}

// Range min/max updates followed by one materialization. A long fragment updates
// O(log windows) nodes, not every covered bin/window. Memory depends on the output
// window count, never on the read count or the largest template span.
class FetchRangeTree {
    std::size_t n_;
    std::vector<FetchBounds> tree_;
public:
    explicit FetchRangeTree(std::size_t n) : n_(n), tree_(2 * n) {}
    void include(std::size_t first, std::size_t end, const FetchBounds &bounds) {
        for (first += n_, end += n_; first < end; first /= 2, end /= 2) {
            if (first & 1) tree_[first++].include(bounds);
            if (end & 1) tree_[--end].include(bounds);
        }
    }
    void merge(const FetchRangeTree &other) {
        for (std::size_t i = 1; i < tree_.size(); ++i) tree_[i].include(other.tree_[i]);
    }
    std::vector<FetchBounds> finish() {
        for (std::size_t i = 1; i < n_; ++i) {
            tree_[2 * i].include(tree_[i]);
            tree_[2 * i + 1].include(tree_[i]);
        }
        return {tree_.begin() + n_, tree_.end()};
    }
};

struct MateInfo {
    bool eligible = false;
    // |TLEN| of an eligible proper mate: always for a reverse mate; for a forward
    // mate only when its reverse mate completes the fragment (see `pairs_`).
    uint64_t span = 0;
    uint8_t touch = 0;  // bit 0 whitelist; bit 1 blacklist
    std::vector<std::pair<int64_t, int64_t>> introns;  // partner mate's, --skipIntrons
    // Partner mate's aligned blocks, when its placer joins them (counting fragments).
    std::vector<std::pair<int64_t, int64_t>> blocks;
};

inline bool same_mate_info(const MateInfo &a, const MateInfo &b) {
    return a.eligible == b.eligible && a.span == b.span &&
           a.touch == b.touch && a.introns == b.introns && a.blocks == b.blocks;
}

std::string same_end_key(const bam1_t *rec) {
    std::string result(bam_get_qname(rec));
    result.push_back('\0');
    const uint8_t *rg = bam_aux_get(rec, "RG");
    const char *group = rg ? bam_aux2Z(rg) : nullptr;
    if (group) result.append(group);
    result.push_back('\0');
    const int64_t mate_coordinates[] = {rec->core.mtid, rec->core.mpos};
    result.append(reinterpret_cast<const char *>(mate_coordinates),
                  sizeof(mate_coordinates));
    const uint16_t end = rec->core.flag &
        (SAM_FIRST_IN_PAIR | SAM_SECOND_IN_PAIR);
    result.append(reinterpret_cast<const char *>(&end), sizeof(end));
    return result;
}

uint8_t record_touch(const bam1_t *rec, const CovParams &P,
                     const std::vector<RegionSet> &black,
                     const std::vector<RegionSet> &white,
                     std::vector<std::pair<int64_t, int64_t>> &blocks) {
    const int tid = rec->core.tid;
    if (tid < 0 || static_cast<std::size_t>(tid) >= white.size())
        throw std::runtime_error("BAM record has an invalid reference id");
    extract_blocks(rec, blocks);
    const int strand = transcript_strand(rec, P.strand);
    uint8_t touch = 0;
    if (P.has_whitelist &&
        touches(blocks, pick_regions(white[tid], P.touch_strand, strand)))
        touch |= 1;
    if (P.has_blacklist &&
        touches(blocks, pick_regions(black[tid], P.touch_strand, strand)))
        touch |= 2;
    return touch;
}

// Stable across platforms and runs, unlike std::hash: two independent 64-bit
// hashes of the key bytes (FNV-1a and a rotate-multiply hash), each with a
// splitmix64 finaliser. Keys are joined by hash across all coordinates, so the
// pair keeps accidental collisions out of reach (about n^2 / 2^129).
struct KeyHash {
    uint64_t hi = 0, lo = 0;
    bool operator<(const KeyHash &o) const { return std::tie(hi, lo) < std::tie(o.hi, o.lo); }
    bool operator==(const KeyHash &o) const { return hi == o.hi && lo == o.lo; }
};

uint64_t splitmix_finish(uint64_t hash) {
    hash ^= hash >> 30; hash *= 0xbf58476d1ce4e5b9ull;
    hash ^= hash >> 27; hash *= 0x94d049bb133111ebull;
    return hash ^ (hash >> 31);
}

KeyHash stable_key_hash(const std::string &key) {
    uint64_t a = 1469598103934665603ull, b = key.size() ^ 0x9e3779b97f4a7c15ull;
    for (const unsigned char c : key) {
        a ^= c; a *= 1099511628211ull;
        b ^= c; b = ((b << 23) | (b >> 41)) * 0xff51afd7ed558ccdull;
    }
    return {splitmix_finish(a), splitmix_finish(b)};
}

// Spill streams read and write through a buffer of our size, so that it
// counts against the budget.
std::size_t spill_stream_bytes(std::size_t share) {
    return std::min<std::size_t>(64u << 10, std::max<std::size_t>(64, share / 16));
}

template <class Stream>
void open_spill(Stream &stream, std::vector<char> &buffer, std::size_t bytes, const std::string &path) {
    constexpr bool output = std::is_base_of<std::ostream, Stream>::value;
    buffer.resize(bytes);
    stream.rdbuf()->pubsetbuf(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    stream.open(path, std::ios::binary | (output ? std::ios::out | std::ios::trunc : std::ios::in));
    if (!stream)
        throw std::runtime_error(std::string("could not ") + (output ? "create" : "open") +
                                 " the non-primary overlap index spill file");
}

template <class T>
void write_records(std::ofstream &out, const T *data, std::size_t count) {
    out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(count * sizeof(T)));
    if (!out) throw std::runtime_error("could not write the non-primary overlap index spill file");
}

template <class T>
std::size_t read_records(std::ifstream &in, T *data, std::size_t count) {
    in.read(reinterpret_cast<char *>(data), static_cast<std::streamsize>(count * sizeof(T)));
    if (in.bad() || in.gcount() % sizeof(T))
        throw std::runtime_error("could not read the non-primary overlap index spill file");
    return static_cast<std::size_t>(in.gcount()) / sizeof(T);
}

// Grows a vector without ever reserving beyond `limit` elements, so that its
// capacity (what the budget accounts) never exceeds the limit.
template <class T>
void bounded_push(std::vector<T> &items, const T &item, std::size_t limit) {
    if (items.size() == items.capacity())
        items.reserve(std::min(std::max<std::size_t>(16, 2 * items.capacity()), limit));
    items.push_back(item);
}

// Byte accounting of the non-primary index: every component reports what it
// holds, and the high-water mark of the sum is the peak.
struct NonprimaryStats {
    std::size_t live = 0, peak_bytes = 0, passes = 0;
    bool spilled = false;
    void account(std::size_t &held, std::size_t now) {
        live += now - held;
        held = now;
        peak_bytes = std::max(peak_bytes, live);
    }
};

// Sorts plain records (ordered by T::operator<) within `share` bytes. Records
// beyond the share are sorted into runs in a temporary file; finish() merges
// runs at most fan_in at a time until one merge remains, and next() streams
// the records back in order, from memory or from that merge.
template <class T>
class SpillSorter {
    struct Source { std::vector<char> buffer; std::ifstream in; uint64_t left = 0; };
    using Head = std::pair<T, std::size_t>;
    static bool later(const Head &a, const Head &b) { return b.first < a.first; }

    NonprimaryStats &stats_;
    const std::size_t stream_bytes_, limit_, fan_in_;  // limit_ records in memory
    TempFile files_[2];
    int from_ = 0;  // the file holding runs_
    bool spilled_ = false;
    std::vector<T> items_;
    std::size_t at_ = 0, held_ = 0;
    uint64_t pushed_ = 0;
    std::vector<char> out_buffer_;
    std::ofstream out_;
    std::vector<std::pair<uint64_t, uint64_t>> runs_;  // (first record, records)
    std::vector<Source> sources_;
    std::vector<Head> heads_;

    void account() {
        stats_.account(held_, items_.capacity() * sizeof(T) + out_buffer_.capacity() +
                       sources_.size() * (stream_bytes_ + sizeof(Head)));
    }
    void close_output() {
        out_.close();
        if (!out_) throw std::runtime_error("could not write the non-primary overlap index spill file");
        std::vector<char>().swap(out_buffer_);
        account();
    }
    void spill() {
        if (!spilled_) {
            open_spill(out_, out_buffer_, stream_bytes_, files_[0].path);
            spilled_ = stats_.spilled = true;
        }
        std::sort(items_.begin(), items_.end());
        write_records(out_, items_.data(), items_.size());
        runs_.emplace_back(runs_.empty() ? 0 : runs_.back().first + runs_.back().second, items_.size());
        items_.clear();
        account();
    }
    static bool pull(Source &source, T &item) {
        if (!source.left) return false;
        if (read_records(source.in, &item, 1) != 1)
            throw std::runtime_error("truncated non-primary overlap index spill file");
        --source.left;
        return true;
    }
    void open(std::size_t first, std::size_t width) {
        sources_.clear();
        heads_.clear();
        sources_.resize(width);
        for (std::size_t i = 0; i < width; ++i) {
            Source &source = sources_[i];
            open_spill(source.in, source.buffer, stream_bytes_, files_[from_].path);
            source.in.seekg(static_cast<std::streamoff>(runs_[first + i].first * sizeof(T)));
            source.left = runs_[first + i].second;
            T item;
            if (pull(source, item)) heads_.emplace_back(item, i);
        }
        std::make_heap(heads_.begin(), heads_.end(), later);
        account();
    }

public:
    SpillSorter(std::size_t share, const std::string &path, NonprimaryStats &stats)
        : stats_(stats), stream_bytes_(spill_stream_bytes(share)),
          limit_(std::max<std::size_t>(1, (share - std::min(share, stream_bytes_)) / sizeof(T))),
          fan_in_(std::max<std::size_t>(2, share / stream_bytes_ - 1)),
          files_{TempFile(path + ".0"), TempFile(path + ".1")} {}
    ~SpillSorter() { release(); }

    bool full() const { return items_.size() == limit_; }
    const std::vector<T> &items() const { return items_; }  // sorted after finish() unless spilled
    bool spilled() const { return spilled_; }
    uint64_t pushed() const { return pushed_; }

    void push(const T &item) {
        if (full()) spill();
        bounded_push(items_, item, limit_);
        ++pushed_;
        account();
    }

    void finish() {
        if (!spilled_) { std::sort(items_.begin(), items_.end()); return; }
        if (!items_.empty()) spill();
        std::vector<T>().swap(items_);
        close_output();
        while (runs_.size() > fan_in_) {
            open_spill(out_, out_buffer_, stream_bytes_, files_[1 - from_].path);
            std::vector<std::pair<uint64_t, uint64_t>> merged;
            uint64_t written = 0;
            for (std::size_t group = 0; group < runs_.size(); group += fan_in_) {
                const uint64_t first = written;
                open(group, std::min(fan_in_, runs_.size() - group));
                for (T item; next(item); ++written) write_records(out_, &item, 1);
                merged.emplace_back(first, written - first);
            }
            close_output();
            runs_.swap(merged);
            from_ = 1 - from_;
        }
        open(0, runs_.size());
    }

    bool next(T &item) {
        if (!spilled_) {
            if (at_ == items_.size()) return false;
            item = items_[at_++];
            return true;
        }
        if (heads_.empty()) return false;
        std::pop_heap(heads_.begin(), heads_.end(), later);
        item = heads_.back().first;
        const std::size_t i = heads_.back().second;
        heads_.pop_back();
        T following;
        if (pull(sources_[i], following)) {
            heads_.emplace_back(following, i);
            std::push_heap(heads_.begin(), heads_.end(), later);
        }
        return true;
    }

    void release() {
        std::vector<T>().swap(items_);
        std::vector<Source>().swap(sources_);
        std::vector<Head>().swap(heads_);
        account();
    }
};

// Overlap-filter touch bits inherited from non-primary alignments. One entry
// per alignment that inherits: (tid, pos, hash of its same-end key, bits),
// sorted. Held in memory, or in a sorted spill file with every block_-th entry
// in memory. Read-only once built; each MateLookup reads through its own Reader.
// Caveat: two distinct same-end keys at one coordinate whose 64-bit hashes
// collide would share bits (probability about n^2 / 2^65 per coordinate).
class NonprimaryTouchIndex {
public:
    struct Entry {
        int64_t pos = 0;
        uint64_t hash = 0;
        int32_t tid = 0;
        uint8_t touch = 0;
        bool operator<(const Entry &o) const { return std::tie(tid, pos, hash) < std::tie(o.tid, o.pos, o.hash); }
        bool same(const Entry &o) const { return tid == o.tid && pos == o.pos && hash == o.hash; }
    };
    bool empty() const { return count_ == 0; }

    class Reader {
        const NonprimaryTouchIndex &index_;
        std::ifstream in_;
        std::vector<Entry> block_;
        std::size_t cached_ = std::numeric_limits<std::size_t>::max();
    public:
        explicit Reader(const NonprimaryTouchIndex &index) : index_(index) {}
        uint8_t touch(const bam1_t *rec) {
            if (index_.empty()) return 0;
            Entry wanted;
            wanted.tid = rec->core.tid; wanted.pos = rec->core.pos;
            const Entry *first = index_.memory_.data(), *last = first + index_.memory_.size();
            if (index_.file_.path.empty()) {
                // Most records inherit nothing: rule out their coordinate before hashing.
                const Entry *at = std::lower_bound(first, last, wanted);
                if (at == last || at->tid != wanted.tid || at->pos != wanted.pos) return 0;
                wanted.hash = stable_key_hash(same_end_key(rec)).hi;
                first = at;
            } else {
                wanted.hash = stable_key_hash(same_end_key(rec)).hi;
                const std::size_t block = std::upper_bound(first, last, wanted) - first;
                if (!block) return 0;
                load(block - 1);
                first = block_.data(); last = first + block_.size();
            }
            const Entry *found = std::lower_bound(first, last, wanted);
            return found != last && found->same(wanted) ? found->touch : 0;
        }
    private:
        // Window processing is coordinate ordered, so one cached block serves
        // nearly every lookup of a window.
        void load(std::size_t block) {
            if (block == cached_) return;
            if (!in_.is_open()) {
                in_.open(index_.file_.path, std::ios::binary);
                if (!in_) throw std::runtime_error("could not open the non-primary overlap index spill file");
            }
            const uint64_t begin = static_cast<uint64_t>(block) * index_.block_;
            block_.resize(std::min<uint64_t>(index_.block_, index_.count_ - begin));
            in_.clear();
            in_.seekg(static_cast<std::streamoff>(begin * sizeof(Entry)));
            if (read_records(in_, block_.data(), block_.size()) != block_.size())
                throw std::runtime_error("truncated non-primary overlap index spill file");
            cached_ = block;
        }
    };

    std::vector<Entry> memory_;  // all entries, or each block's first entry
    TempFile file_;
    uint64_t count_ = 0;
    std::size_t block_ = 1;
};

// Non-primary alignments can occur after their primary pair in coordinate
// order. An index of the same-end identities whose secondary or supplementary
// alignments touch a filter region lets the primary pair (and every eligible
// alignment of that read end) receive their touch decision in either
// direction. Each identity must resolve to one eligible primary coordinate;
// ambiguous identities are rejected rather than mixed.
//
// An external-sort hash join in two sequential BAM passes, within `budget`
// bytes (plus a few records of slack at tiny budgets):
//  1. (key hash, touch) for each touching non-primary alignment. Beyond a
//     quarter of the budget they are sorted into runs on disk, and a Bloom
//     filter of another quarter remembers their keys.
//  2. (key hash, coordinate, primary, eligible) for each alignment whose key
//     is known (exactly while pass 1 fit in memory, else by the Bloom filter;
//     false positives find no touch in the join), sorted the same way.
//  3. Both streams are merged by key hash; each valid identity yields one
//     (tid, pos, hash, touch) entry per eligible alignment, sorted by
//     coordinate into the index (a quarter of the budget, then disk).
NonprimaryTouchIndex build_nonprimary_touch_index(
        const std::string &path, const CovParams &P,
        const std::vector<RegionSet> &black,
        const std::vector<RegionSet> &white,
        std::size_t budget, const std::string &spill_path, NonprimaryStats &stats) {
    using Entry = NonprimaryTouchIndex::Entry;
    struct Touching {
        KeyHash key;
        uint8_t touch;
        bool operator<(const Touching &o) const { return key < o.key; }
    };
    struct Occurrence {  // primaries first within a key
        KeyHash key;
        int64_t pos;
        int32_t tid;
        bool primary, eligible;
        bool operator<(const Occurrence &o) const { return std::tie(key, o.primary) < std::tie(o.key, primary); }
    };
    constexpr std::size_t MAX_KEY_BYTES = 4096;
    const std::size_t share = std::max<std::size_t>(1, budget / 4);

    std::vector<std::pair<int64_t, int64_t>> blocks;
    auto scan = [&](auto &&visit) {
        ++stats.passes;
        Alignments reads = open_reads(path, P.reference, false);
        htsFile *file = reads.file.get();
        sam_hdr_t *header = reads.hdr.get();
        BamRecordPtr record{bam_init1()};
        if (!record.get()) throw std::runtime_error("could not allocate a BAM record");
        uint64_t scanned = 0;
        while (checked_read(sam_read1(file, header, record.get())) >= 0) {
            poll_coverage_work(++scanned);
            const bam1_t *rec = record.get();
            // Every same-end key in the index has non-negative mate coordinates.
            if (rec->core.mtid < 0 || rec->core.mpos < 0) continue;
            visit(rec);
        }
    };
    auto primary_pair = [](const bam1_t *rec) {
        const uint16_t flag = rec->core.flag;
        return (flag & SAM_READ_PAIRED) && !(flag & SAM_MATE_UNMAPPED) &&
               !(flag & (SAM_SECONDARY | SAM_SUPPLEMENTARY));
    };

    std::vector<uint64_t> bloom;
    std::size_t bloom_held = 0;
    auto bloom_probe = [&](const KeyHash &key, bool add) {
        bool present = true;
        for (uint64_t i = 0; i < 3; ++i) {
            const uint64_t bit = (key.hi + i * key.lo) % (bloom.size() * 64);
            const uint64_t mask = uint64_t(1) << (bit % 64);
            present = present && (bloom[bit / 64] & mask);
            if (add) bloom[bit / 64] |= mask;
        }
        return present;
    };

    SpillSorter<Touching> touching(share, spill_path + ".n", stats);
    scan([&](const bam1_t *rec) {
        const uint16_t flag = rec->core.flag;
        if (!(flag & SAM_READ_PAIRED) || (flag & SAM_MATE_UNMAPPED) ||
            !(flag & (SAM_SECONDARY | SAM_SUPPLEMENTARY)) || !passes_filters(rec, P)) return;
        const uint8_t touch = record_touch(rec, P, black, white, blocks);
        if (!touch) return;
        const std::string key = same_end_key(rec);
        if (key.size() > MAX_KEY_BYTES) return;
        const Touching item{stable_key_hash(key), touch};
        if (touching.full() && bloom.empty()) {  // about to spill: remember keys from now on
            bloom.assign(std::max<std::size_t>(1, share / sizeof(uint64_t)), 0);
            stats.account(bloom_held, bloom.capacity() * sizeof(uint64_t));
            for (const Touching &held : touching.items()) bloom_probe(held.key, true);
        }
        if (!bloom.empty()) bloom_probe(item.key, true);
        touching.push(item);
    });
    if (!touching.pushed()) return {};
    touching.finish();

    SpillSorter<Occurrence> occurrences(share, spill_path + ".o", stats);
    scan([&](const bam1_t *rec) {
        const bool primary = primary_pair(rec), eligible = passes_filters(rec, P);
        if (!primary && !eligible) return;
        const std::string key = same_end_key(rec);
        if (key.size() > MAX_KEY_BYTES) return;
        const KeyHash hash = stable_key_hash(key);
        const auto &known = touching.items();
        if (bloom.empty() ? !std::binary_search(known.begin(), known.end(), Touching{hash, 0})
                          : !bloom_probe(hash, false)) return;
        occurrences.push(Occurrence{hash, rec->core.pos, rec->core.tid, primary, eligible});
    });
    std::vector<uint64_t>().swap(bloom);
    stats.account(bloom_held, 0);
    occurrences.finish();

    SpillSorter<Entry> results(share, spill_path + ".r", stats);
    Touching n;
    Occurrence o;
    bool has_n = touching.next(n), has_o = occurrences.next(o);
    while (has_o) {
        const KeyHash key = o.key;
        uint8_t touch = 0;
        for (; has_n && n.key < key; has_n = touching.next(n)) {}
        for (; has_n && n.key == key; has_n = touching.next(n)) touch |= n.touch;
        Occurrence primary{};
        bool has_primary = false, ambiguous = false;
        for (; has_o && o.key == key && o.primary; has_o = occurrences.next(o)) {
            ambiguous = ambiguous || (has_primary && (o.tid != primary.tid || o.pos != primary.pos ||
                                                      o.eligible != primary.eligible));
            if (!has_primary) { primary = o; has_primary = true; }
        }
        const bool valid = touch && has_primary && primary.eligible && !ambiguous;
        if (valid) results.push(Entry{primary.pos, key.hi, primary.tid, touch});
        for (; has_o && o.key == key; has_o = occurrences.next(o))
            if (valid && o.eligible) results.push(Entry{o.pos, key.hi, o.tid, touch});
    }
    touching.release();
    occurrences.release();
    results.finish();

    // Merge entries of one coordinate and key; when spilled, write them to the
    // index file and keep each block's first entry in memory.
    NonprimaryTouchIndex index;
    std::vector<char> out_buffer;
    std::ofstream out;
    if (results.spilled()) {
        index.file_ = TempFile(spill_path + ".index");
        open_spill(out, out_buffer, spill_stream_bytes(share), index.file_.path);
        index.block_ = std::max<std::size_t>(64, static_cast<std::size_t>(results.pushed() / (share / sizeof(Entry) + 1) + 1));
    }
    std::size_t index_held = 0;
    index.memory_.reserve(static_cast<std::size_t>(results.pushed() / index.block_ + 1));
    stats.account(index_held, index.memory_.capacity() * sizeof(Entry) + out_buffer.capacity());
    auto put = [&](const Entry &entry) {
        if (index.count_++ % index.block_ == 0) index.memory_.push_back(entry);
        if (out.is_open()) write_records(out, &entry, 1);
    };
    Entry entry, last;
    bool any = false;
    while (results.next(entry)) {
        if (any && last.same(entry)) { last.touch |= entry.touch; continue; }
        if (any) put(last);
        last = entry;
        any = true;
    }
    if (any) put(last);
    if (out.is_open()) {
        out.close();
        if (!out) throw std::runtime_error("could not write the non-primary overlap index spill file");
    }
    index.memory_.shrink_to_fit();
    return index;
}

// A CRAM region query decodes every whole container it touches, and the
// windows next to each other (and each window's mate prefetch) touch the same
// containers, so per-query decoding repeats the work several times. All
// workers share these decoded containers instead: each one is decoded once,
// kept while in use plus a small byte budget of recently used ones, and
// queries walk them with htslib's own CRAM range rules (cram_first_slice,
// cram_next_slice and cram_get_seq in cram_decode.c), so a query visits
// exactly the records sam_itr_queryi returns, in the same order.
class CramContainers {
    struct Block {
        int ref = -2;
        hts_pos_t start = 0, span = 0;  // container header coordinates
        bool empty = false;             // zero-length container
        bool end = false;               // no container at this offset (EOF)
        int64_t next = -1;              // offset of the following container
        std::vector<bam1_core_t> core;
        std::vector<std::size_t> offset;  // record i's data: [offset[i], offset[i + 1])
        std::vector<uint8_t> data;
        std::size_t bytes() const {
            return sizeof(Block) + core.capacity() * sizeof(bam1_core_t) +
                   offset.capacity() * sizeof(std::size_t) + data.capacity();
        }
    };
    struct Entry { std::shared_ptr<const Block> block; uint64_t used = 0; };

    const std::string path_, reference_;
    const std::size_t budget_;
    Alignments index_;  // finds a query's first container, like cram_seek_to_refpos
    std::mutex index_mutex_, mutex_;
    std::condition_variable loaded_;
    std::unordered_map<int64_t, Entry> blocks_;  // a null block is being decoded
    std::vector<Alignments> decoders_;           // idle range-free handles
    std::size_t bytes_ = 0;
    uint64_t clock_ = 0;
    std::atomic<uint64_t> decodes_{0};

    std::shared_ptr<const Block> decode(int64_t offset) {
        Alignments reads;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!decoders_.empty()) { reads = std::move(decoders_.back()); decoders_.pop_back(); }
        }
        if (!reads.file.get()) reads = open_reads(path_, reference_, false);
        auto block = std::make_shared<Block>();
        cram_fd *fd = reads.file.get()->fp.cram;
        const auto fail = [&] { return std::runtime_error("could not read a CRAM container of " + display_path(path_)); };
        if (cram_seek(fd, offset, SEEK_SET) != 0) throw fail();
        cram_container *header = cram_read_container(fd);
        if (!header) {
            if (!cram_eof(fd)) throw fail();
            block->end = true;
        } else {
            const int32_t records = cram_container_get_num_records(header);
            const int32_t length = cram_container_get_length(header);
            cram_container_get_coords(header, &block->ref, &block->start, &block->span);
            cram_free_container(header);
            block->next = htell(cram_fd_get_fp(fd)) + length;
            block->empty = length == 0;
            // The handle has no range, so it reads the container's records in
            // order; reading all of them leaves it ready for any next seek.
            if (!block->empty && records > 0) {
                if (cram_seek(fd, offset, SEEK_SET) != 0) throw fail();
                BamRecordPtr record(bam_init1());
                if (!record.get()) throw std::runtime_error("could not allocate a BAM record");
                block->core.reserve(records);
                block->offset.reserve(records + 1);
                block->offset.push_back(0);
                for (int32_t i = 0; i < records; ++i) {
                    if (checked_read(sam_read1(reads.file.get(), reads.hdr.get(), record.get())) < 0)
                        throw fail();
                    const bam1_t *r = record.get();
                    block->core.push_back(r->core);
                    block->data.insert(block->data.end(), r->data, r->data + r->l_data);
                    block->offset.push_back(block->data.size());
                }
                block->data.shrink_to_fit();
            }
        }
        decodes_.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(mutex_);
        decoders_.push_back(std::move(reads));
        return block;
    }

    // The decoded container at `offset`, decoding it once for all callers.
    std::shared_ptr<const Block> get(int64_t offset) {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            auto found = blocks_.find(offset);
            if (found == blocks_.end()) break;
            if (found->second.block) {
                found->second.used = ++clock_;
                return found->second.block;
            }
            loaded_.wait(lock);
        }
        blocks_.emplace(offset, Entry{});
        lock.unlock();
        std::shared_ptr<const Block> block;
        try {
            block = decode(offset);
        } catch (...) {
            lock.lock();
            blocks_.erase(offset);
            loaded_.notify_all();
            throw;
        }
        lock.lock();
        blocks_[offset] = Entry{block, ++clock_};
        bytes_ += block->bytes();
        // Evict the least recently used containers no query holds.
        while (bytes_ > budget_) {
            auto victim = blocks_.end();
            for (auto it = blocks_.begin(); it != blocks_.end(); ++it)
                if (it->second.block && it->second.block.use_count() == 1 &&
                    (victim == blocks_.end() || it->second.used < victim->second.used))
                    victim = it;
            if (victim == blocks_.end()) break;
            bytes_ -= victim->second.block->bytes();
            blocks_.erase(victim);
        }
        loaded_.notify_all();
        return block;
    }

public:
    CramContainers(const std::string &path, const std::string &reference, std::size_t budget)
        : path_(path), reference_(reference), budget_(budget),
          index_(open_reads(path, reference, true)) {}

    uint64_t decodes() const { return decodes_.load(); }

    // Visit the records sam_itr_queryi(tid, beg, end) returns, in file order.
    template <class Visit>
    void query(int tid, int64_t beg, int64_t end, Visit &&visit) {
        const hts_pos_t start = beg + 1;  // CRAM ranges are 1-based, inclusive
        off_t first = 0;
        {
            std::lock_guard<std::mutex> lock(index_mutex_);
            if (cram_index_extents(index_.file.get()->fp.cram, tid, start, end, &first, nullptr) != 0)
                return;  // no index entry: htslib reports no data
        }
        bam1_t view{};
        view.mempolicy = BAM_USER_OWNS_DATA;
        bool at_first = true;
        for (int64_t offset = first;;) {
            const std::shared_ptr<const Block> block = get(offset);
            if (block->end) return;
            offset = block->next;
            if (block->empty) continue;
            if (block->ref != -2) {
                const bool before = block->start + block->span - 1 < start;
                if (at_first) {
                    if (block->ref < tid || (block->ref == tid && before)) continue;
                    if (block->ref != tid) return;
                } else {
                    if (block->ref != tid || block->start > end) return;
                    if (before) continue;
                }
            }
            at_first = false;
            for (std::size_t i = 0; i < block->core.size(); ++i) {
                const bam1_core_t &core = block->core[i];
                if (core.tid < tid && core.tid != -1) continue;
                if (core.tid != tid || core.pos + 1 > end) return;
                view.core = core;
                view.data = const_cast<uint8_t *>(block->data.data() + block->offset[i]);
                view.l_data = view.m_data = static_cast<uint32_t>(block->offset[i + 1] - block->offset[i]);
                // htslib's CRAM record end: the last reference base, at least POS.
                const hts_pos_t aend = (!(core.flag & BAM_FUNMAP) && core.n_cigar)
                    ? std::max<hts_pos_t>(core.pos + 1, core.pos + bam_cigar2rlen(core.n_cigar, bam_get_cigar(&view)))
                    : core.pos + 1;
                if (aend < start) continue;
                visit(&view);
            }
        }
    }
};

// Visit the records of sam_itr_queryi(tid, beg, end) on `reads`, or, for a CRAM,
// the same records from the shared decoded containers.
template <class Visit>
void for_each_record(CramContainers *cram, Alignments &reads, bam1_t *record,
                     int tid, int64_t beg, int64_t end, const char *error, Visit &&visit) {
    if (cram) {
        cram->query(tid, beg, end, visit);
        return;
    }
    IteratorPtr iterator(sam_itr_queryi(reads.idx.get(), tid, beg, end));
    if (!iterator.get()) throw std::runtime_error(error);
    while (checked_read(sam_itr_next(reads.file.get(), iterator.get(), record)) >= 0)
        visit(record);
}

// The planning pass reads the file as chunks of whole windows, in parallel.
// Each record belongs to the chunk holding its start; the last chunk of a
// reference reaches to its end, so it also holds records placed past it.
struct Chunk { int tid; int64_t begin, end; };

std::vector<Chunk> coverage_chunks(const CovParams &P, const std::vector<uint32_t> &lengths) {
    const int64_t width = static_cast<int64_t>(P.window_bins) * P.bin_size * 16;
    std::vector<Chunk> chunks;
    for (std::size_t tid = 0; tid < lengths.size(); ++tid)
        for (int64_t begin = 0; begin == 0 || begin < lengths[tid]; begin += width)
            chunks.push_back({static_cast<int>(tid), begin,
                              begin + width < lengths[tid] ? begin + width : HTS_POS_MAX});
    return chunks;
}

// `observe(worker, chunk, record)` sees each placed record once; a worker
// handles its chunks one at a time, each in coordinate order.
template<class Observe>
std::vector<std::vector<FetchBounds>> prepare_coverage(
        const std::string &path, const CovParams &P, const std::vector<uint32_t> &lengths,
        const std::vector<Chunk> &chunks, std::size_t workers,
        bool plan_fetches, Observe observe) {
    std::vector<std::vector<FetchRangeTree>> trees(workers);
    if (plan_fetches) for (auto &worker_trees : trees) for (uint32_t length : lengths) {
        const uint64_t bins = (static_cast<uint64_t>(length) + P.bin_size - 1) / P.bin_size;
        worker_trees.emplace_back((bins + P.window_bins - 1) / P.window_bins);
    }
    std::atomic<std::size_t> next{0};
    dtp::parallel_rows(workers, static_cast<int>(workers), [&](std::size_t worker) {
        Alignments reads = open_reads(path, P.reference, true);
        BamRecordPtr record(bam_init1());
        if (!record.get()) throw std::runtime_error("could not allocate a BAM record");
        std::vector<std::pair<int64_t, int64_t>> intervals;
        uint64_t scanned = 0;
        for (std::size_t c; !dtp::interruption_requested() &&
                            (c = next.fetch_add(1, std::memory_order_relaxed)) < chunks.size();) {
            const Chunk &chunk = chunks[c];
            for_each_record(nullptr, reads, record.get(), chunk.tid, chunk.begin, chunk.end,
                            "could not query BAM/CRAM region", [&](const bam1_t *rec) {
                poll_coverage_work(++scanned);
                if (rec->core.pos < chunk.begin) return;
                observe(worker, c, rec);
                if (!plan_fetches) return;
                const int tid = rec->core.tid;
                int weight = 1;
                if (shape_read(rec, P, lengths[tid], intervals, weight) < 0) return;
                // Counting fragments, a placer may also cover its partner's blocks
                // (the union, or a dovetailed span): plan its whole fragment.
                if (P.fragments && (rec->core.flag & SAM_READ_PAIRED) && rec->core.mtid == tid &&
                    rec->core.mpos >= 0 && rec->core.mpos <= rec->core.pos) {
                    const int64_t end = std::max<int64_t>(
                        bam_endpos(rec), rec->core.mpos + fragment_length_for_filter(rec));
                    intervals.emplace_back(rec->core.mpos, std::min<int64_t>(end, lengths[tid]));
                }
                // Fetching one base at an alignment's start is sufficient to retrieve it.
                const FetchBounds bounds{rec->core.pos, rec->core.pos + 1};
                for (const auto &interval : intervals) {
                    const uint64_t first = interval.first / P.bin_size / P.window_bins;
                    const uint64_t last = (interval.second - 1) / P.bin_size / P.window_bins;
                    trees[worker][tid].include(first, last + 1, bounds);
                }
            });
        }
    });
    std::vector<std::vector<FetchBounds>> result;
    if (plan_fetches) for (std::size_t tid = 0; tid < lengths.size(); ++tid) {
        for (std::size_t worker = 1; worker < workers; ++worker) trees[0][tid].merge(trees[worker][tid]);
        result.push_back(trees[0][tid].finish());
    }
    return result;
}

// Each worker has its own lazy htslib handle. Mates are paired in-stream: the
// query a caller already streams (a window, or a whole reference) is the
// evidence. Each primary pair seen gets one entry, which collects every
// record that is either end. A read whose mate starts earlier in the stream
// is decided at once; one whose mate starts later waits in the entry until
// the stream has passed the mate's coordinate, when every record starting
// there has been seen, and the entry is dropped. A read whose mate lies
// outside the stream (another reference, before or after the query, a
// non-primary requester, or budget pressure) is deferred into a batch with
// its own fixed budget; a full batch resolves all its mates with one
// sequential query per cluster of nearby coordinates. Accumulators are
// order-independent sums, so deferring a read never changes the output.
// High-water marks are reported through the test-only knob.
struct MateBudget {
    std::size_t cache_entries = 32768, cache_bytes = 8u << 20;
    std::size_t batch_entries = 16384, batch_bytes = 4u << 20;
    std::atomic<std::size_t> peak_cache_entries{0}, peak_cache_bytes{0};
    std::atomic<std::size_t> peak_batch_entries{0}, peak_batch_bytes{0};
    std::size_t nonprimary_bytes = 64u << 20;  // see build_nonprimary_touch_index
};

class MateLookup {
    using Intervals = std::vector<std::pair<int64_t, int64_t>>;
    const std::string &path_;
    const CovParams &P_;
    MateBudget &budget_;
    const std::vector<RegionSet> &black_, &white_;
    NonprimaryTouchIndex::Reader nonprimary_touch_;
    // `fragment_`: reads are fragments, placed once per proper pair. `pairs_`:
    // the placer also needs its partner mate (complete()); a dovetailed pair's
    // placer always does. `plain_`: counting fragments without --extendReads or
    // --collapse, a fragment is the union of its mates' aligned blocks.
    const bool fragment_, plain_, pairs_;
    const bool asks_;  // some read may need its mate: the stream is tracked
    CramContainers *cram_;  // shared decoded containers of a CRAM, else null
    Alignments reads_;
    BamRecordPtr record_{bam_init1()};
    Intervals blocks_, shape_;
    std::string key_;
    // Reused summaries: the observed record's, and a flush candidate's (a
    // flush can run inside observe()).
    MateInfo info_, candidate_;
    // What a read's keep/drop decision needs from its mate.
    struct Need {
        bool drop = false, touch = false, once = false, pair = false, dovetail = false;
        uint64_t span = 0;
        int64_t start = 0, end = 0;  // a dovetailed placer's aligned span
        int sign = 1;                // and its fragment orientation (--collapse)
    };
    // One fragment's mates: the placer places it, its partner is counted
    // through it. The reverse mate places an inward pair; counting fragments,
    // the right-hand forward mate places a dovetailed pair.
    struct Role { bool placer = false, partner = false, dovetail = false; };
    Role role(const bam1_t *rec) const {
        Role r;
        if (!fragment_) return r;
        const bool reverse = rec->core.flag & SAM_READ_REVERSE;
        if (is_proper_fragment(rec)) r.placer = reverse;
        else if (P_.fragments && is_dovetail(rec)) r.placer = !reverse, r.dovetail = true;
        else return r;
        r.partner = !r.placer;
        return r;
    }
    // A read waiting for its mate, and what it adds once decided.
    struct Request {
        uint8_t touch;
        Need need;
        int stream, weight;
        Intervals iv;
    };
    struct Deferred {  // batched: the mate is found by its key
        std::string key;
        int32_t tid;
        int64_t pos;  // the mate's
        bool loose;
        Request read;
        std::size_t bytes() const {
            return sizeof(Deferred) + key.size() + read.iv.size() * sizeof(read.iv[0]);
        }
    };
    // Every primary record with one key: the first summary, and whether
    // another one differed (an ambiguous mate counts as none).
    struct Match { MateInfo info; bool found = false, ambiguous = false; };
    static void add_candidate(Match &match, const MateInfo &candidate) {
        if (!match.found) match.info = candidate, match.found = true, match.ambiguous = false;
        else if (!same_mate_info(match.info, candidate)) match.ambiguous = true;
    }
    static MateInfo value(const Match &match) { return match.ambiguous ? MateInfo{} : match.info; }

    // The stream, and one entry per primary pair seen in it. A pair's key is
    // its name, read group and both ends (position, end flag) in coordinate
    // order; side 0 is the first end. A pair whose ends coincide has one side.
    static constexpr std::size_t END_BYTES = sizeof(int64_t) + sizeof(uint16_t);
    struct Pair {
        Match side[2];
        int32_t waiting = -1;  // first waiting read (index into waiting_)
        std::size_t bytes = 0;
    };
    using PairMap = std::unordered_map<std::string, Pair>;
    using PairEntry = PairMap::value_type;
    PairMap pairs_map_;
    // Dropped entries' nodes, reused with their key and vector capacity.
    std::vector<PairMap::node_type> spare_pairs_;
    // Min-heap by the later end's position: past it, the entry is complete.
    std::vector<std::pair<int64_t, PairEntry *>> expiries_;
    struct Waiting { Request read; int mate_side; int32_t next; };
    std::vector<Waiting> waiting_;   // a pool; resolved slots keep their buffers
    std::vector<int32_t> free_waiting_;
    PairEntry *current_ = nullptr;   // the observed read's pair, if any
    int current_side_ = 0, current_mate_side_ = 0;
    int stream_tid_ = -1;
    int64_t stream_begin_ = 0, stream_end_ = 0;
    bool incomplete_ = false;  // a pair entry was dropped: absence proves nothing
    std::size_t cache_bytes_ = 0;

    std::vector<Deferred> batch_;
    std::size_t batch_bytes_ = 0;
    std::size_t peak_cache_entries_ = 0, peak_cache_bytes_ = 0;
    std::size_t peak_batch_entries_ = 0, peak_batch_bytes_ = 0;
    static void raise_peak(std::atomic<std::size_t> &peak, std::size_t value) {
        std::size_t seen = peak.load();
        while (seen < value && !peak.compare_exchange_weak(seen, value)) {}
    }

    void open_lookup() {
        if (!cram_ && !reads_.file.get()) reads_ = open_reads(path_, P_.reference, true);
    }
    static uint16_t read_end(const bam1_t *rec, bool mate) {
        uint16_t end = rec->core.flag & (SAM_FIRST_IN_PAIR | SAM_SECOND_IN_PAIR);
        if (mate) end = ((end & SAM_FIRST_IN_PAIR) << 1) |
                        ((end & SAM_SECOND_IN_PAIR) >> 1);
        return end;
    }
    // Loose identity is required for a supplementary/secondary request because
    // RNEXT/PNEXT identifies its primary mate, whose reciprocal PNEXT normally
    // points to the requesting read's *primary* alignment rather than this one.
    static void build_key(std::string &result, const bam1_t *rec, bool mate, bool loose) {
        result.clear();
        result.append(bam_get_qname(rec)); result.push_back('\0');
        const uint8_t *rg = bam_aux_get(rec, "RG");
        const char *group = rg ? bam_aux2Z(rg) : nullptr;
        if (group) result.append(group);
        result.push_back('\0');
        const int64_t coords[] = {mate ? rec->core.mtid : rec->core.tid,
                                  mate ? rec->core.mpos : rec->core.pos};
        result.append(reinterpret_cast<const char *>(coords), sizeof(coords));
        const uint16_t end = read_end(rec, mate);
        result.append(reinterpret_cast<const char *>(&end), sizeof(end));
        if (!loose) {
            const int64_t reciprocal[] = {mate ? rec->core.tid : rec->core.mtid,
                                          mate ? rec->core.pos : rec->core.mpos};
            result.append(reinterpret_cast<const char *>(reciprocal), sizeof(reciprocal));
        }
    }
    // Fills `info` in place, so its vectors keep their capacity.
    void summarize(const bam1_t *rec, MateInfo &info) {
        info.span = 0;
        info.touch = 0;
        info.introns.clear();
        info.blocks.clear();
        info.eligible = passes_filters(rec, P_);
        if (!info.eligible) return;
        const Role r = role(rec);
        if (r.placer || (r.partner && (pairs_ || r.dovetail))) {
            info.span = fragment_length_for_filter(rec);
            if (r.partner && P_.skip_introns) extract_introns(rec, info.introns);
            if (r.partner && P_.fragments && (r.dovetail || plain_))
                extract_blocks(rec, info.blocks);
        }
        if (P_.overlap_filter && (P_.has_whitelist || P_.has_blacklist)) {
            info.touch = record_touch(rec, P_, black_, white_, blocks_);
            info.touch |= nonprimary_touch_.touch(rec);
        }
    }
    Need need(const bam1_t *rec, const MateInfo &own) const {
        Need result;
        if (!own.eligible) { result.drop = true; return result; }
        if (P_.overlap_filter && (P_.has_whitelist || P_.has_blacklist)) {
            if (P_.has_blacklist && (own.touch & 2)) { result.drop = true; return result; }
            result.touch = P_.has_blacklist || !(own.touch & 1);
        }
        // A retained fragment is placed at its placer; the partner stands in
        // only when the placer is missing.
        const Role r = role(rec);
        result.once = r.partner;
        result.pair = r.placer && (pairs_ || r.dovetail);
        result.dovetail = r.dovetail;
        if (result.once || result.pair) result.span = fragment_length_for_filter(rec);
        if (result.pair && r.dovetail) {
            result.start = rec->core.pos;
            result.end = bam_endpos(rec);
            if (P_.strand != Strand::None) result.sign = transcript_strand(rec, P_.strand);
        }
        return result;
    }
    // A placer completes its fragment with its partner. Counting reads, both
    // alignments count (weight 2). Counting fragments, plain coverage is the
    // union of both mates' blocks, and a dovetailed pair spans from the
    // leftmost to the rightmost aligned base. --skipIntrons also leaves out
    // the partner's introns.
    void complete(const Need &need, const MateInfo &mate, int &weight, Intervals &shape) const {
        if (!need.pair || mate.span != need.span) return;
        if (!P_.fragments) weight *= 2;
        if (need.dovetail) {
            const int64_t fs = std::min(need.start, mate.blocks.front().first);
            const int64_t fe = std::max(need.end, mate.blocks.back().second);
            if (P_.collapse != Collapse::None) {
                shape.clear();
                collapse_span(fs, fe, need.sign, P_, fe, shape);
                return;
            }
            if (P_.extend) {
                Intervals own_introns;  // the gaps shape_read left in the placer
                for (std::size_t i = 1; i < shape.size(); ++i)
                    own_introns.emplace_back(shape[i - 1].second, shape[i].first);
                shape.assign(1, {fs, fe});
                subtract_gaps(shape, own_introns);
            }
        }
        subtract_gaps(shape, mate.introns);
        if (!P_.extend) join_blocks(shape, mate.blocks);
    }
    bool keeps(uint8_t own_touch, const Need &need, const MateInfo &mate) const {
        const uint8_t touch = own_touch | mate.touch;
        if (need.touch && !((!P_.has_whitelist || (touch & 1)) &&
                            (!P_.has_blacklist || !(touch & 2)))) return false;
        return !need.once || mate.span != need.span;
    }
    template <class Keep>
    void resolve(Request &read, const MateInfo &info, Keep &&keep) {
        if (!keeps(read.touch, read.need, info)) return;
        int weight = read.weight;
        complete(read.need, info, weight, read.iv);
        keep(read.stream, weight, read.iv);
    }
    bool has_mate_coordinate(const bam1_t *rec) const {
        return (rec->core.flag & SAM_READ_PAIRED) && !(rec->core.flag & SAM_MATE_UNMAPPED) &&
               rec->core.mtid >= 0 && rec->core.mpos >= 0;
    }
    // Whether every primary record starting at the mate's coordinate is in
    // the stream (once the stream has passed it).
    bool in_stream(const bam1_t *rec) const {
        return !(rec->core.flag & (SAM_SECONDARY | SAM_SUPPLEMENTARY)) &&
               rec->core.mtid == stream_tid_ && rec->core.mpos >= stream_begin_ &&
               rec->core.mpos < stream_end_;
    }
    // The pair key of a record (its mtid is the stream's): returns its side
    // and its mate's.
    std::pair<int, int> pair_key(std::string &key, const bam1_t *rec) const {
        const char *name = bam_get_qname(rec);
        const uint8_t *rg = bam_aux_get(rec, "RG");
        const char *group = rg ? bam_aux2Z(rg) : nullptr;
        const std::size_t name_size = std::strlen(name), group_size = group ? std::strlen(group) : 0;
        key.resize(name_size + group_size + 2 + 2 * END_BYTES);  // one write, no appends
        char *at = key.data();
        std::memcpy(at, name, name_size + 1); at += name_size + 1;
        if (group) std::memcpy(at, group, group_size);
        at += group_size; *at++ = '\0';
        const std::pair<int64_t, uint16_t> own{rec->core.pos, read_end(rec, false)},
                                           mate{rec->core.mpos, read_end(rec, true)};
        const bool swap = mate < own;
        for (const auto &end : {swap ? mate : own, swap ? own : mate}) {
            std::memcpy(at, &end.first, sizeof(end.first)); at += sizeof(end.first);
            std::memcpy(at, &end.second, sizeof(end.second)); at += sizeof(end.second);
        }
        if (own == mate) return {0, 0};
        return swap ? std::make_pair(1, 0) : std::make_pair(0, 1);
    }
    // A read waiting in `pair` for `mate_side`, batched instead (see build_key).
    Deferred batched(const std::string &pair, int mate_side, Request &&read) const {
        const std::size_t prefix = pair.size() - 2 * END_BYTES;
        auto end = [&](int side) {
            std::pair<int64_t, uint16_t> result;
            const char *at = pair.data() + prefix + side * END_BYTES;
            std::memcpy(&result.first, at, sizeof(result.first));
            std::memcpy(&result.second, at + sizeof(result.first), sizeof(result.second));
            return result;
        };
        const auto mate = end(mate_side), own = end(end(0) == end(1) ? mate_side : 1 - mate_side);
        Deferred deferred{pair.substr(0, prefix), stream_tid_, mate.first, false, std::move(read)};
        const int64_t coords[] = {stream_tid_, mate.first, stream_tid_, own.first};
        deferred.key.append(reinterpret_cast<const char *>(coords), 2 * sizeof(int64_t));
        deferred.key.append(reinterpret_cast<const char *>(&mate.second), sizeof(mate.second));
        deferred.key.append(reinterpret_cast<const char *>(coords + 2), 2 * sizeof(int64_t));
        return deferred;
    }
    void note_cache() {
        const std::size_t entries = pairs_map_.size() + waiting_.size() - free_waiting_.size();
        peak_cache_entries_ = std::max(peak_cache_entries_, entries);
        peak_cache_bytes_ = std::max(peak_cache_bytes_, cache_bytes_);
    }
    bool cache_full(std::size_t bytes) const {
        return pairs_map_.size() + waiting_.size() - free_waiting_.size() >= budget_.cache_entries ||
               cache_bytes_ + bytes > budget_.cache_bytes;
    }
    // The pair entry for `key`, created when budget allows (else null).
    PairEntry *pair_entry(const bam1_t *rec, std::size_t extra_bytes) {
        auto found = pairs_map_.find(key_);
        if (found != pairs_map_.end()) return &*found;
        const std::size_t bytes = key_.size() + sizeof(PairEntry) + 64 + extra_bytes;
        if (cache_full(bytes)) return nullptr;
        PairEntry *entry;
        if (spare_pairs_.empty()) {
            entry = &*pairs_map_.emplace(key_, Pair{}).first;
        } else {
            PairMap::node_type node = std::move(spare_pairs_.back());
            spare_pairs_.pop_back();
            node.key() = key_;
            Pair &pair = node.mapped();
            for (Match &side : pair.side) side.found = side.ambiguous = false;
            pair.waiting = -1;
            entry = &*pairs_map_.insert(std::move(node)).position;
        }
        entry->second.bytes = bytes;
        cache_bytes_ += bytes;
        expiries_.emplace_back(std::max<int64_t>(rec->core.pos, rec->core.mpos), entry);
        std::push_heap(expiries_.begin(), expiries_.end(), Later{});
        note_cache();
        return entry;
    }
    struct Later {
        bool operator()(const std::pair<int64_t, PairEntry *> &a,
                        const std::pair<int64_t, PairEntry *> &b) const { return a.first > b.first; }
    };
    // The stream has passed `pos`: every pair whose later end is before it is
    // complete. Its waiting reads are decided and the entry is dropped.
    template <class Keep>
    void advance(int64_t pos, Keep &&keep) {
        while (!expiries_.empty() && expiries_.front().first < pos) {
            std::pop_heap(expiries_.begin(), expiries_.end(), Later{});
            PairEntry *entry = expiries_.back().second;
            expiries_.pop_back();
            for (int32_t i = entry->second.waiting; i >= 0; i = waiting_[i].next) {
                Waiting &w = waiting_[i];
                const Match &mate = entry->second.side[w.mate_side];
                cache_bytes_ -= sizeof(Waiting) + w.read.iv.size() * sizeof(w.read.iv[0]);
                if (mate.found) resolve(w.read, value(mate), keep);
                else if (!incomplete_) resolve(w.read, MateInfo{}, keep);
                else batch(batched(entry->first, w.mate_side, std::move(w.read)), keep);
                free_waiting_.push_back(i);
            }
            cache_bytes_ -= entry->second.bytes;
            spare_pairs_.push_back(pairs_map_.extract(entry->first));
        }
    }
    template <class Keep>
    void batch(Deferred &&read, Keep &&keep) {
        const std::size_t bytes = read.bytes();
        if (!batch_.empty() && batch_bytes_ + bytes > budget_.batch_bytes) flush(keep);
        batch_.push_back(std::move(read));
        batch_bytes_ += bytes;
        peak_batch_entries_ = std::max(peak_batch_entries_, batch_.size());
        peak_batch_bytes_ = std::max(peak_batch_bytes_, batch_bytes_);
        if (batch_.size() >= budget_.batch_entries || batch_bytes_ >= budget_.batch_bytes) flush(keep);
    }
    // Resolves every deferred mate exactly: same key (name, read group, both
    // coordinates, mate end), primary, and ambiguous duplicates count as no
    // mate. Nearby wanted coordinates share one sequential query.
    template <class Keep>
    void flush(Keep &&keep) {
        if (batch_.empty()) return;
        std::unordered_map<std::string, Match> matches[2];  // full, loose keys
        std::vector<std::pair<int32_t, int64_t>> wanted;
        wanted.reserve(batch_.size());
        for (const auto &read : batch_) {
            matches[read.loose].emplace(read.key, Match{});
            wanted.emplace_back(read.tid, read.pos);
        }
        std::sort(wanted.begin(), wanted.end());
        wanted.erase(std::unique(wanted.begin(), wanted.end()), wanted.end());
        constexpr int64_t cluster_gap = 1 << 16;
        open_lookup();
        for (std::size_t first = 0, last = 0; first < wanted.size(); first = last) {
            for (last = first + 1; last < wanted.size() && wanted[last].first == wanted[first].first &&
                 wanted[last].second - wanted[last - 1].second <= cluster_gap; ++last) {}
            uint64_t scanned = 0;
            for_each_record(cram_, reads_, record_.get(), wanted[first].first, wanted[first].second,
                            wanted[last - 1].second + 1, "could not query BAM mate coordinates",
                            [&](const bam1_t *mate) {
                poll_coverage_work(++scanned);
                if (!(mate->core.flag & SAM_READ_PAIRED) ||
                    (mate->core.flag & (SAM_SECONDARY | SAM_SUPPLEMENTARY)) ||
                    !std::binary_search(wanted.begin() + first, wanted.begin() + last,
                                        std::make_pair(mate->core.tid, int64_t(mate->core.pos))))
                    return;
                for (bool loose : {false, true}) {
                    if (matches[loose].empty()) continue;
                    build_key(key_, mate, false, loose);
                    const auto match = matches[loose].find(key_);
                    if (match == matches[loose].end()) continue;
                    summarize(mate, candidate_);
                    add_candidate(match->second, candidate_);
                }
            });
        }
        std::vector<Deferred> resolving;
        resolving.swap(batch_);
        batch_bytes_ = 0;
        for (auto &read : resolving) resolve(read.read, value(matches[read.loose].at(read.key)), keep);
    }
public:
    enum class Decision { Drop, Keep, Defer };
    MateLookup(const std::string &path, const CovParams &P, MateBudget &budget,
               const std::vector<RegionSet> &black, const std::vector<RegionSet> &white,
               const NonprimaryTouchIndex &nonprimary_touch, bool fragment,
               CramContainers *cram)
        : path_(path), P_(P), budget_(budget), black_(black), white_(white),
          nonprimary_touch_(nonprimary_touch), fragment_(fragment),
          plain_(P.fragments && !P.extend && P.collapse == Collapse::None),
          pairs_(fragment && (!P.fragments || P.skip_introns || plain_)),
          asks_(fragment || (P.overlap_filter && (P.has_whitelist || P.has_blacklist))),
          cram_(cram) {
        if (!record_.get()) throw std::runtime_error("could not allocate a mate record");
    }
    ~MateLookup() {
        raise_peak(budget_.peak_cache_entries, peak_cache_entries_);
        raise_peak(budget_.peak_cache_bytes, peak_cache_bytes_);
        raise_peak(budget_.peak_batch_entries, peak_batch_entries_);
        raise_peak(budget_.peak_batch_bytes, peak_batch_bytes_);
    }
    // Starts a stream: every record overlapping [begin, end) of `tid`, in
    // coordinate order, passes through observe(). End it with finish().
    void begin(int tid, int64_t begin, int64_t end) {
        stream_tid_ = tid; stream_begin_ = begin; stream_end_ = end;
    }
    // The read's own summary (its eligibility and touch); the stream advances
    // to it, and a primary record joins its pair's entry.
    template <class Keep>
    MateInfo observe(const bam1_t *rec, Keep &&keep) {
        summarize(rec, info_);
        // The caller needs no blocks or introns.
        MateInfo info;
        info.eligible = info_.eligible; info.span = info_.span; info.touch = info_.touch;
        current_ = nullptr;
        if (!asks_ || rec->core.tid != stream_tid_) return info;
        advance(rec->core.pos, keep);
        if (!(rec->core.flag & SAM_READ_PAIRED) ||
            (rec->core.flag & (SAM_SECONDARY | SAM_SUPPLEMENTARY)) ||
            rec->core.pos < stream_begin_ || rec->core.mtid != rec->core.tid) return info;
        std::tie(current_side_, current_mate_side_) = pair_key(key_, rec);
        current_ = pair_entry(rec, (info_.introns.size() + info_.blocks.size()) * sizeof(info_.introns[0]));
        if (!current_) { incomplete_ = true; return info; }
        add_candidate(current_->second.side[current_side_], info_);
        return info;
    }
    // Keep/drop from the read and its `mate` seen earlier in the stream (valid
    // until the next observe()). Defer hands the read to defer(), which decides
    // it once its mate is known. A kept read then passes through complete().
    Decision decide(const bam1_t *rec, const MateInfo &own, const MateInfo *&mate) {
        static const MateInfo none;
        mate = &none;
        const Need wanted = need(rec, own);
        if (wanted.drop) return Decision::Drop;
        if (!wanted.touch && !wanted.once && !wanted.pair) return Decision::Keep;
        if (has_mate_coordinate(rec)) {
            // In the stream, an earlier mate's end is complete; the read is in
            // the stream too, so it joined the pair's entry unless dropped.
            if (!in_stream(rec) || rec->core.mpos >= rec->core.pos || !current_)
                return Decision::Defer;
            const Match &found = current_->second.side[current_mate_side_];
            if (found.found) {
                if (!found.ambiguous) mate = &found.info;
            } else if (incomplete_) {
                return Decision::Defer;
            }
        }
        return keeps(own.touch, wanted, *mate) ? Decision::Keep : Decision::Drop;
    }
    void complete(const bam1_t *rec, const MateInfo &own, const MateInfo &mate,
                  int &weight, Intervals &shape) const {
        complete(need(rec, own), mate, weight, shape);
    }
    // `keep(stream, weight, intervals)` receives every deferred read retained.
    template <class Keep>
    void defer(const bam1_t *rec, const MateInfo &own,
               int stream, int weight, const Intervals &iv, Keep &&keep) {
        const std::size_t bytes = sizeof(Waiting) + iv.size() * sizeof(iv[0]);
        if (in_stream(rec) && rec->core.mpos >= rec->core.pos) {
            // A later mate: wait in the pair's entry. A read before the query
            // start has not joined one yet.
            PairEntry *entry = current_;
            int mate_side = current_mate_side_;
            if (!entry && rec->core.pos < stream_begin_) {
                mate_side = pair_key(key_, rec).second;
                entry = pair_entry(rec, 0);
            }
            if (entry && !cache_full(bytes)) {
                int32_t i;
                if (free_waiting_.empty()) { i = static_cast<int32_t>(waiting_.size()); waiting_.emplace_back(); }
                else { i = free_waiting_.back(); free_waiting_.pop_back(); }
                Waiting &w = waiting_[i];
                w.read.touch = own.touch; w.read.need = need(rec, own);
                w.read.stream = stream; w.read.weight = weight;
                w.read.iv.assign(iv.begin(), iv.end());
                w.mate_side = mate_side;
                w.next = entry->second.waiting;
                entry->second.waiting = i;
                cache_bytes_ += bytes;
                note_cache();
                return;
            }
        }
        Deferred read{std::string(), rec->core.mtid, rec->core.mpos,
                      (rec->core.flag & (SAM_SECONDARY | SAM_SUPPLEMENTARY)) != 0,
                      Request{own.touch, need(rec, own), stream, weight, iv}};
        build_key(read.key, rec, true, read.loose);
        batch(std::move(read), keep);
    }
    // Ends the stream: the whole query has been seen, so every pending read
    // is decided, and the batch resolves.
    template <class Keep>
    void finish(Keep &&keep) {
        advance(std::numeric_limits<int64_t>::max(), keep);
        flush(keep);
        incomplete_ = false;
        current_ = nullptr;
        stream_tid_ = -1;
    }
};

// Each worker keeps its bounded accumulators between windows. Reusing these
// buffers avoids allocating/zero-initializing a second set for every query.
struct CoverageScratch {
    std::vector<std::vector<int64_t>> acc, rng;
    // A difference array necessarily contains negative end-boundary deltas.
    // Keep it signed and wide enough that the boundary stage cannot overflow
    // before materialization into the int64 running total below.
    std::vector<std::vector<int64_t>> cdiff;
    std::vector<std::vector<int64_t>> total;  // per-bin totals per lane
    std::vector<std::vector<char>> masked;    // per-bin mask per lane
    bool signal = false;  // the last window had a non-zero unmasked bin (before scaling)
};

// Compute one window's per-bin values (single output stream). `reads` (with its
// own index) and `rec` are per-thread; `cram` is shared (null for BAM/SAM).
inline void compute_window(Alignments &reads, CramContainers *cram, const Window &w,
                           const CovParams &P, uint32_t chrom_len,
                           const RegionSet &blackset,
                           const RegionSet &whiteset,
                           MateLookup &mates,
                           bam1_t *rec,
                           std::vector<std::pair<int64_t, int64_t>> &iv,
                           std::vector<std::vector<float>> &values,
                           CoverageScratch &scratch) {
    const uint32_t nbins = w.bin1 - w.bin0;
    const uint32_t ws_bp = w.bin0 * P.bin_size;
    const uint32_t we_bp = std::min<uint64_t>(
        static_cast<uint64_t>(w.bin1) * P.bin_size, chrom_len);
    int64_t qstart = static_cast<int64_t>(ws_bp) - P.halo;
    if (qstart < 0) qstart = 0;
    int64_t qend = static_cast<int64_t>(we_bp) + P.halo;
    if (qend > chrom_len) qend = chrom_len;
    if (P.fetch_bounds) {
        const auto &bounds = (*P.fetch_bounds)[w.tid][w.bin0 / P.window_bins];
        qstart = std::min(qstart, bounds.begin);
        qend = std::max(qend, bounds.end);
    }

    const int ns = P.strand_lanes ? 2 : P.nstreams;  // lanes
    auto &acc = scratch.acc; auto &rng = scratch.rng; auto &cdiff = scratch.cdiff;
    acc.resize(ns); rng.resize(ns); cdiff.resize(ns);
    for (int s = 0; s < ns; ++s) {
        if (P.agg == Agg::Count) cdiff[s].assign(nbins + 1, 0);
        else { acc[s].assign(nbins, 0); rng[s].assign(nbins + 1, 0); }
    }

    const bool do_overlap = P.overlap_filter &&
                            (P.has_whitelist || P.has_blacklist);
    const bool lookup = do_overlap || P.extend || P.fragments;
    if (lookup) mates.begin(w.tid, qstart, qend);
    auto keep = [&](int s, int weight, const std::vector<std::pair<int64_t, int64_t>> &shape) {
        accumulate_read(shape, P, w.bin0, nbins, ws_bp, we_bp, acc[s], rng[s], cdiff[s], weight);
    };
    uint64_t scanned = 0;
    for_each_record(cram, reads, rec, w.tid, qstart, qend, "could not query BAM/CRAM region",
                    [&](const bam1_t *read) {
        poll_coverage_work(++scanned);
        MateInfo own;
        const MateInfo *mate = nullptr;
        auto decision = MateLookup::Decision::Keep;
        if (lookup) {
            own = mates.observe(read, keep);
            decision = mates.decide(read, own, mate);
            if (decision == MateLookup::Decision::Drop) return;
        }
        int read_weight = 1;
        const int s = shape_read(read, P, chrom_len, iv, read_weight, lookup);
        if (s < 0) return;
        if (decision == MateLookup::Decision::Defer) {
            mates.defer(read, own, s, read_weight, iv, keep);
            return;
        }
        if (lookup) mates.complete(read, own, *mate, read_weight, iv);
        keep(s, read_weight, iv);
    });
    if (lookup) mates.finish(keep);

    // Per-bin totals per lane, and the lane's mask: bins outside the
    // whitelist or overlapping the blacklist (entries of the lane's strand).
    const bool masking = !P.overlap_filter && (P.has_whitelist || P.has_blacklist);
    auto &total = scratch.total; auto &masked = scratch.masked;
    total.resize(ns); masked.resize(ns);
    auto mark = [&](const RegionSet &set, int lane, char value) {
        for (const auto &region : pick_regions(set, P.touch_strand, lane == 0 ? +1 : -1)) {
            int64_t b0 = static_cast<int64_t>(region.first) / P.bin_size - w.bin0;
            int64_t b1 = static_cast<int64_t>(region.second - 1) / P.bin_size - w.bin0;
            if (b0 < 0) b0 = 0;
            if (b1 >= static_cast<int64_t>(nbins)) b1 = nbins - 1;
            for (int64_t i = b0; i <= b1; ++i) masked[lane][i] = value;
        }
    };
    for (int l = 0; l < ns; ++l) {
        total[l].resize(nbins);
        int64_t run = 0;
        for (uint32_t i = 0; i < nbins; ++i) {
            if (P.agg == Agg::Count) total[l][i] = run += cdiff[l][i];
            else total[l][i] = acc[l][i] + (run += rng[l][i]);
        }
        if (!masking) continue;
        masked[l].assign(nbins, P.has_whitelist);
        if (P.has_whitelist) mark(whiteset, l, 0);
        if (P.has_blacklist) mark(blackset, l, 1);
    }

    values.resize(P.nstreams);
    scratch.signal = false;
    for (int s = 0; s < P.nstreams; ++s) {
        const double sc = (s == 1) ? P.scale_minus : P.scale;  // per-strand scale
        std::vector<float> &vals = values[s];
        vals.resize(nbins);
        const int first_lane = P.strand_lanes ? 0 : s, last_lane = P.strand_lanes ? 1 : s;
        for (uint32_t i = 0; i < nbins; ++i) {
            int64_t a = 0;
            bool zero = masking;  // a bin masked in every lane is zero
            for (int l = first_lane; l <= last_lane; ++l) {
                if (masking && masked[l][i]) continue;
                a += total[l][i];
                zero = false;
            }
            if (zero) { vals[i] = 0.0f; continue; }
            if (a != 0) scratch.signal = true;
            const uint32_t abin = w.bin0 + i;
            if (P.agg == Agg::Count) {
                double bin_scale = sc;
                if (P.rpkm) {
                    uint64_t be = (static_cast<uint64_t>(abin) + 1) * P.bin_size;
                    if (be > chrom_len) be = chrom_len;
                    const uint64_t begin = static_cast<uint64_t>(abin) * P.bin_size;
                    bin_scale /= static_cast<double>(be - begin);
                }
                vals[i] = checked_float(a * bin_scale, "coverage value");
            } else if (P.agg == Agg::Sum) {
                vals[i] = checked_float(a * sc, "coverage value");
            } else {
                const uint64_t be = std::min<uint64_t>(
                    (static_cast<uint64_t>(abin) + 1) * P.bin_size, chrom_len);
                vals[i] = checked_float(
                    static_cast<double>(a) / (be - abin * P.bin_size) * sc,
                    "coverage value");
            }
        }
    }
}


// Returns true when a black/whitelist removed all coverage: the output has no
// non-zero bin, yet the same run without the lists would have had one.
bool bam_coverage_bigwig(const std::string &bam_path,
                         const std::string &out_path,
                         const std::string &out_path_reverse,
                         const std::string &out_format,
                         uint32_t bin_size,
                         const std::string &aggregation,
                         int min_mapping_quality,
                         const FlagPatterns &sam_flag_include,
                         uint16_t sam_flag_exclude,
                         const std::string &filter_mode,
                         bool ignore_duplicates,
                         int min_fragment_length,
                         int max_fragment_length,
                         int extend_reads,
                         bool skip_introns,
                         const std::string &strandedness,
                         const std::string &filter_rna_strand,
                         const std::string &collapse_str,
                         int collapsed_length,
                         const std::string &count_unit,
                         py::object blacklist,
                         py::object whitelist,
                         int threads,
                         int window_size,
                         const std::string &normalization,
                         const std::string &normalization_denominator,
                         bool exact_scaling,
                         double scale_factor,
                         double scale_factor_minus,
                         py::object ignore_for_normalization,
                         int max_zooms,
                         int compression_level,
                         bool filter_by_overlap,
                         const std::string &feature_touch_strand,
                         int filter_by_overlap_halo,
                         py::object mate_budget_knob,
                         const std::string &reference_fasta) {
    if (bin_size == 0) throw std::runtime_error("binSize must be > 0");
    if (min_mapping_quality < 0 || min_mapping_quality > 255)
        throw std::runtime_error("min_mapping_quality must be in 0..255");
    if (min_fragment_length < 0 || max_fragment_length < 0)
        throw std::runtime_error("fragment lengths must be >= 0");
    if (max_fragment_length > 0 && min_fragment_length > max_fragment_length)
        throw std::runtime_error(
            "min_fragment_length must be <= max_fragment_length");
    if (extend_reads < 0)
        throw std::runtime_error("extend_reads must be >= 0");
    if (skip_introns && extend_reads == 0)
        throw std::runtime_error("skip_introns requires extend_reads");
    if (collapsed_length < 1)
        throw std::runtime_error("collapsed_length must be >= 1");
    if (threads < 0) throw std::runtime_error("threads must be >= 0");
    if (window_size < 0) throw std::runtime_error("window_size must be >= 0");
    if (!std::isfinite(scale_factor) || !std::isfinite(scale_factor_minus))
        throw std::runtime_error("scale factors must be finite");
    validate_writer_options(max_zooms, compression_level);
    Agg agg;
    if (aggregation == "mean") agg = Agg::Mean;
    else if (aggregation == "sum") agg = Agg::Sum;
    else if (aggregation == "count") agg = Agg::Count;
    else throw std::runtime_error("aggregation must be mean|sum|count");
    if (normalization != "none" && normalization != "cpm" &&
        normalization != "rpkm" && normalization != "bpm")
        throw std::runtime_error(
            "normalization must be none|cpm|rpkm|bpm");
    if (normalization != "none" && agg != Agg::Count)
        throw std::runtime_error(
            "normalized metrics require read-count aggregation");

    FilterMode fmode;
    if (filter_mode == "primary") fmode = FilterMode::Primary;
    else if (filter_mode == "deeptools") fmode = FilterMode::Deeptools;
    else throw std::runtime_error("filter_mode must be primary|deeptools");

    Strand strand;
    if (strandedness == "none") strand = Strand::None;
    else if (strandedness == "forward") strand = Strand::Forward;
    else if (strandedness == "reverse") strand = Strand::Reverse;
    else throw std::runtime_error("strandedness must be none|forward|reverse");

    const bool split = (filter_rna_strand == "split");
    StrandFilter sfilter = StrandFilter::None;
    if (!split) {
        if (filter_rna_strand == "none") sfilter = StrandFilter::None;
        else if (filter_rna_strand == "forward") sfilter = StrandFilter::Forward;
        else if (filter_rna_strand == "reverse") sfilter = StrandFilter::Reverse;
        else throw std::runtime_error(
            "filter_rna_strand must be none|forward|reverse|split");
    }
    if ((sfilter != StrandFilter::None || split) && strand == Strand::None)
        throw std::runtime_error(
            "filter_rna_strand needs strandedness (forward|reverse) to be set");
    if (split && out_path_reverse.empty())
        throw std::runtime_error(
            "filter_rna_strand split needs a second output (out_path_reverse)");

    TouchStrand touch_strand;
    if (feature_touch_strand == "ignore") touch_strand = TouchStrand::Ignore;
    else if (feature_touch_strand == "sense") touch_strand = TouchStrand::Sense;
    else if (feature_touch_strand == "antisense") touch_strand = TouchStrand::Antisense;
    else throw std::runtime_error("feature_touch_strand must be ignore|sense|antisense");
    if (touch_strand != TouchStrand::Ignore && strand == Strand::None)
        throw std::runtime_error(
            "feature_touch_strand needs strandedness (forward|reverse) to be set");
    if (filter_by_overlap_halo < -1)
        throw std::runtime_error("filter_by_overlap_halo must be >= 0 or -1 for auto");

    Collapse collapse;
    if (collapse_str == "none") collapse = Collapse::None;
    else if (collapse_str == "5prime") collapse = Collapse::FivePrime;
    else if (collapse_str == "3prime") collapse = Collapse::ThreePrime;
    else if (collapse_str == "center") collapse = Collapse::Center;
    else throw std::runtime_error("collapse must be none|5prime|3prime|center");
    if (collapse != Collapse::None && extend_reads > 0)
        throw std::runtime_error(
            "extend_reads and collapse cannot be used together");
    if (count_unit != "read" && count_unit != "fragment")
        throw std::runtime_error("count_unit must be read|fragment");
    if (collapse != Collapse::None && collapsed_length < 1)
        throw std::runtime_error("collapsed_length must be >= 1");

    // Effective exclude mask: always drop unmapped; primary mode drops
    // secondary+supplementary; add the caller's exclude bits and, optionally,
    // the duplicate bit.
    uint16_t exclude = sam_flag_exclude | SAM_READ_UNMAPPED;
    if (fmode == FilterMode::Primary) exclude |= (SAM_SECONDARY | SAM_SUPPLEMENTARY);
    if (ignore_duplicates) exclude |= SAM_DUPLICATE;
    const bool extend = extend_reads > 0 && collapse == Collapse::None;

    Alignments reads = open_reads(bam_path, reference_fasta, true);
    check_index(reads, bam_path);
    sam_hdr_t *hdr = reads.hdr.get();
    const int n_targets = sam_hdr_nref(hdr);
    if (n_targets <= 0) throw std::runtime_error("no references in header");

    std::vector<uint32_t> lengths(n_targets);
    std::vector<const char *> names(n_targets);
    for (int t = 0; t < n_targets; ++t) {
        lengths[t] = sam_hdr_tid2len(hdr, t);
        names[t] = sam_hdr_tid2name(hdr, t);
    }

    // Blacklist / whitelist: {chrom: [(start,end[,strand]), ...]}. Strand-split,
    // merged RegionSets used either as bin-masks (default) or fragment-touch filters
    // (--filterByOverlap), optionally strand-oriented (--featureTouchStrand).
    std::vector<RegionSet> blacklist_by_tid = parse_region_sets(blacklist, hdr, n_targets);
    std::vector<RegionSet> whitelist_by_tid = parse_region_sets(whitelist, hdr, n_targets);

    OutFmt fmt;
    if (out_format == "bigwig") fmt = OutFmt::BigWig;
    else if (out_format == "bedgraph") fmt = OutFmt::BedGraph;
    else if (out_format == "bedgraph.gz") fmt = OutFmt::BedGraphGz;
    else throw std::runtime_error("out_format must be bigwig|bedgraph|bedgraph.gz");

    // Pack parsed options.
    CovParams P;
    P.reference = reference_fasta;
    P.bin_size = bin_size; P.agg = agg; P.min_mapq = min_mapping_quality;
    P.include = sam_flag_include; P.exclude = exclude;
    P.min_frag = min_fragment_length; P.max_frag = max_fragment_length;
    P.extend = extend; P.extend_len = extend_reads; P.skip_introns = skip_introns;
    P.strand = strand; P.sfilter = sfilter;
    P.split = split; P.nstreams = split ? 2 : 1;
    P.strand_lanes = !split && !filter_by_overlap && touch_strand != TouchStrand::Ignore &&
                     !(blacklist.is_none() && whitelist.is_none());
    P.overlap_filter = filter_by_overlap;
    P.has_blacklist = !blacklist.is_none();
    P.has_whitelist = !whitelist.is_none();
    P.touch_strand = touch_strand;
    P.collapse = collapse; P.collapsed_len = collapsed_length;
    P.fragments = count_unit == "fragment";
    P.scale = scale_factor;         // refined below (× normalization factor)
    P.scale_minus = scale_factor * scale_factor_minus;
    P.rpkm = false;
    // A configured fragment-length ceiling is also a guaranteed bound on how
    // far a retained alignment can influence this output window. Query at
    // least that far so boundary-crossing fragments are never missed even
    // when the user requests a smaller overlap-prefetch halo.
    P.halo = std::max<int64_t>(
        std::max(0, filter_by_overlap_halo), max_fragment_length);
    P.window_bins = coverage_window_bins(bin_size, window_size);
    // Test-only: {"entries": n[, "bytes": n][, "nonprimary_bytes": n]
    // [, "cram_cache_bytes": n]} shrinks the mate, non-primary index and CRAM
    // container budgets, and the same dict receives their high-water marks
    // (and, for a CRAM, "cram_container_decodes").
    MateBudget mate_budget;
    if (!mate_budget_knob.is_none()) {
        const auto knob = mate_budget_knob.cast<py::dict>();
        if (knob.contains("entries"))
            mate_budget.cache_entries = mate_budget.batch_entries = knob["entries"].cast<std::size_t>();
        if (knob.contains("bytes"))
            mate_budget.cache_bytes = mate_budget.batch_bytes = knob["bytes"].cast<std::size_t>();
        if (knob.contains("nonprimary_bytes"))
            mate_budget.nonprimary_bytes = knob["nonprimary_bytes"].cast<std::size_t>();
        if (!mate_budget.cache_entries || !mate_budget.cache_bytes || !mate_budget.nonprimary_bytes)
            throw std::runtime_error("mate budgets must be positive");
    }
    // A CRAM decodes each container once for all workers (CramContainers).
    // Idle decoded containers are kept up to a budget that scales with the
    // number of windows in flight; the knob's "cram_cache_bytes" shrinks it.
    std::unique_ptr<CramContainers> containers;
    if (reads.file.get()->format.format == cram) {
        std::size_t cache_bytes = (dtp::thread_count(threads, SIZE_MAX) + 2) * (1u << 20);
        if (!mate_budget_knob.is_none() && mate_budget_knob.cast<py::dict>().contains("cram_cache_bytes"))
            cache_bytes = mate_budget_knob.cast<py::dict>()["cram_cache_bytes"].cast<std::size_t>();
        containers = std::make_unique<CramContainers>(bam_path, reference_fasta, cache_bytes);
    }

    const bool admits_nonprimary =
        (P.exclude & (SAM_SECONDARY | SAM_SUPPLEMENTARY)) !=
        (SAM_SECONDARY | SAM_SUPPLEMENTARY);
    NonprimaryStats nonprimary_stats;
    const NonprimaryTouchIndex nonprimary_touch =
        (P.overlap_filter && (P.has_whitelist || P.has_blacklist) &&
         admits_nonprimary)
            ? build_nonprimary_touch_index(
                  bam_path, P, blacklist_by_tid, whitelist_by_tid,
                  mate_budget.nonprimary_bytes, out_path + ".nonprimary", nonprimary_stats)
            : NonprimaryTouchIndex{};

    // Chromosome exclusions have one meaning in read-count and BPM denominators.
    std::vector<char> ignore(n_targets, 0);
    if (normalization != "none" && !ignore_for_normalization.is_none()) {
        for (auto o : ignore_for_normalization.cast<py::list>()) {
            const int tid = resolve_header_tid(hdr, py::str(o).cast<std::string>());
            if (tid >= 0) ignore[tid] = 1;
        }
    }
    const bool read_normalization = normalization != "none" && normalization != "bpm";
    const bool filtered = normalization_denominator == "filtered";
    if (read_normalization && !filtered && normalization_denominator != "library")
        throw std::runtime_error("normalization_denominator must be library|filtered");
    // Counting fragments, the library denominator is always an exact count.
    const bool count_reads = read_normalization && (filtered || exact_scaling || P.fragments);
    const bool fragment = extend || P.fragments;
    // Names what can empty a zero normalization denominator in this run:
    // `lists` when the black/whitelist acts on it, `reads` for the read filters.
    auto empty_after = [&](bool lists, bool reads) {
        std::string causes;
        auto add = [&](const std::string &cause) {
            causes += (causes.empty() ? "" : ", ") + cause;
        };
        if (std::find(ignore.begin(), ignore.end(), 1) != ignore.end())
            add("--ignoreForNormalization");
        if (lists && P.has_blacklist) add("--blackListFileName");
        if (lists && P.has_whitelist) add("--whiteListFileName");
        if (reads) add("read filters");
        return causes.empty() ? std::string("(no mapped reads)")
                              : "(nothing left after filtering: " + causes + ")";
    };
    double counted_denominator = 0;
    std::vector<std::vector<FetchBounds>> fetch_bounds;
    if (fragment || count_reads) {
        const std::vector<Chunk> chunks = coverage_chunks(P, lengths);
        const std::size_t workers = dtp::thread_count(threads, chunks.size());
        // Per worker: its count, and (filtered) its mate lookup streaming the
        // worker's current chunk.
        struct Counter {
            double count = 0;
            std::unique_ptr<MateLookup> mates;
            std::size_t chunk = SIZE_MAX;
        };
        std::vector<Counter> counters(workers);
        if (count_reads && filtered)
            for (auto &counter : counters)
                counter.mates = std::make_unique<MateLookup>(
                    bam_path, P, mate_budget, blacklist_by_tid, whitelist_by_tid, nonprimary_touch,
                    P.fragments, containers.get());
        // Both tasks inspect the same records. Keep their distinct eligibility
        // rules, but perform the BAM decode only once when both are requested.
        fetch_bounds = prepare_coverage(bam_path, P, lengths, chunks, workers,
                                        fragment, [&](std::size_t worker, std::size_t chunk,
                                                     const bam1_t *record) {
            if (!count_reads || ignore[record->core.tid]) return;
            Counter &counter = counters[worker];
            auto count = [&](int, int, const std::vector<std::pair<int64_t, int64_t>> &) {
                counter.count += 1;
            };
            if (filtered) {
                // Unmapped records are mate evidence too.
                if (chunk != counter.chunk) {
                    counter.mates->finish(count);
                    counter.chunk = chunk;
                    counter.mates->begin(chunks[chunk].tid, chunks[chunk].begin, chunks[chunk].end);
                }
                const MateInfo own = counter.mates->observe(record, count);
                if (record->core.flag & SAM_READ_UNMAPPED) return;
                const MateInfo *mate;
                const auto decision = counter.mates->decide(record, own, mate);
                if (decision == MateLookup::Decision::Keep) count(0, 1, {});
                else if (decision == MateLookup::Decision::Defer)
                    counter.mates->defer(record, own, 0, 1, {}, count);
            } else if (!(record->core.flag & SAM_READ_UNMAPPED)) {
                // Counting fragments, a proper pair with both mates mapped
                // counts once, at its first mate.
                const uint16_t flag = record->core.flag;
                const uint16_t pair = SAM_READ_PAIRED | SAM_PROPER_PAIR | SAM_SECOND_IN_PAIR;
                if (!(flag & (SAM_SECONDARY | SAM_SUPPLEMENTARY)) &&
                    !(P.fragments && (flag & pair) == pair && !(flag & SAM_MATE_UNMAPPED)))
                    counter.count += 1;
            }
        });
        for (auto &counter : counters) {
            if (counter.mates) counter.mates->finish([&](int, int, const auto &) { counter.count += 1; });
            counted_denominator += counter.count;
        }
        if (fragment) P.fetch_bounds = &fetch_bounds;
    }

    // Open one output per stream (2 for --filterRNAstrand split).
    std::vector<std::string> out_paths = {out_path};
    if (P.split) out_paths.push_back(out_path_reverse);
    const bool is_bw = (fmt == OutFmt::BigWig);
    dtp::BigWigLibraryScope library(is_bw);
    std::vector<std::unique_ptr<BigWigOutput>> bws(P.nstreams);
    std::vector<FilePtr> fhs(P.nstreams);
    auto finalize = [&]() {
        for (int s = 0; s < P.nstreams; ++s) {
            errno = 0;
            if ((bws[s] && bws[s]->close()) || (fhs[s] && std::fclose(fhs[s].release()) != 0))
                throw dtp::write_error("failed finalizing coverage output", out_paths[s]);
        }
    };
    for (int s = 0; s < P.nstreams; ++s) {
        if (is_bw) {
            bws[s] = std::make_unique<BigWigOutput>(out_paths[s], names, lengths, max_zooms,
                                                    compression_level, threads);
        } else {
            fhs[s].reset(std::fopen(out_paths[s].c_str(), "wb"));
            if (!fhs[s]) throw std::runtime_error("could not open output: " + dtp::display_path(out_paths[s]));
        }
    }
    std::vector<StreamWriter> writers;
    writers.reserve(P.nstreams);
    for (int s = 0; s < P.nstreams; ++s)
        writers.push_back(StreamWriter{fmt, bws[s].get(), fhs[s].get(), compression_level,
                                       names, lengths, bin_size, out_paths[s]});

    // Tile every chromosome into windows (bounded memory per window). The default
    // (256K bins) keeps a window's accumulators small even at binSize 1 (~1-4 MB),
    // so total in-flight memory stays low regardless of thread count.
    const uint32_t window_bins = P.window_bins;
    std::vector<Window> windows;
    for (int t = 0; t < n_targets; ++t) {
        const uint32_t nb = static_cast<uint32_t>(
            (static_cast<uint64_t>(lengths[t]) + bin_size - 1) / bin_size);
        for (uint32_t b = 0; b < nb;) {
            const uint32_t end = static_cast<uint32_t>(
                std::min<uint64_t>(static_cast<uint64_t>(b) + window_bins, nb));
            windows.push_back({t, b, end});
            b = end;
        }
    }
    const std::size_t nwin = windows.size();
    const std::size_t nthreads = dtp::thread_count(threads, std::max<std::size_t>(nwin, 1));
    std::atomic<bool> signal{false};  // some output bin is non-zero before scaling

    {
        // --- normalization: compute the scale factor (the P4 correctness fix) ---
        // The denominator D is the LIBRARY SIZE by default, counted independently of
        // the numerator filters and shared across outputs; `filtered` reproduces the
        // old post-filter behaviour.
        enum { N_NONE, N_CPM, N_RPKM, N_BPM } norm;
        if (normalization == "none") norm = N_NONE;
        else if (normalization == "cpm") norm = N_CPM;
        else if (normalization == "rpkm") norm = N_RPKM;
        else if (normalization == "bpm") norm = N_BPM;
        else throw std::runtime_error("normalization must be none|cpm|rpkm|bpm");

        if (norm == N_BPM) {
            // BPM (TPM-like): scale = 1e6 / (sum of all placed bin values). The total
            // is self-referential, so a first parallel pass computes it (unscaled)
            // before the write pass. For split, the pooled forward+reverse sum is the
            // shared denominator (both tracks comparable).
            CovParams Pt = P; Pt.scale = 1.0; Pt.scale_minus = 1.0;
            std::atomic<std::size_t> next{0};
            std::vector<double> partial(nthreads, 0.0);
            dtp::parallel_rows(nthreads, static_cast<int>(nthreads),
                [&](std::size_t ti) {
                    Alignments worker_reads = open_reads(bam_path, reference_fasta, true);
                    BamRecordPtr record(bam_init1());
                    if (!record.get())
                        throw std::runtime_error(
                            "BPM worker could not allocate a BAM record");
                    std::vector<std::pair<int64_t, int64_t>> iv;
                    std::vector<std::vector<float>> vals;
                    CoverageScratch scratch;
                    MateLookup mates(bam_path, Pt, mate_budget, blacklist_by_tid,
                                     whitelist_by_tid, nonprimary_touch, fragment,
                                     containers.get());
                    dtp::CompensatedSum sum;
                    while (!dtp::interruption_requested()) {
                        if (!dtp::running_in_parallel_worker())
                            dtp::check_python_signals();
                        const std::size_t w = next.fetch_add(
                            1, std::memory_order_relaxed);
                        if (w >= nwin) break;
                        if (ignore[windows[w].tid]) continue;
                        compute_window(worker_reads, containers.get(), windows[w], Pt,
                                       lengths[windows[w].tid],
                                       blacklist_by_tid[windows[w].tid],
                                       whitelist_by_tid[windows[w].tid],
                                       mates,
                                       record.get(), iv, vals, scratch);
                        for (const auto &v : vals)
                            for (float x : v) sum.add(x);
                    }
                    partial[ti] = sum.value();
                });
            dtp::CompensatedSum total_sum;
            for (double p : partial) total_sum.add(p);
            const double total = total_sum.value();
            if (total <= 0.0)
                throw std::runtime_error("BPM denominator is zero " +
                                         empty_after(true, true));
            const double f = 1e6 / total;
            P.scale = scale_factor * f;
            P.scale_minus = scale_factor * scale_factor_minus * f;
        } else if (norm != N_NONE) {
            double D = counted_denominator;
            if (!count_reads) {
                const ReadCounts library = read_counts(reads, bam_path);
                // Library size straight from the index (mapped alignments/target).
                for (int t = 0; t < n_targets; ++t) {
                    if (ignore[t]) continue;
                    D += static_cast<double>(library.mapped[t]);
                }
            }
            if (D <= 0.0)
                throw std::runtime_error(
                    "normalization denominator is zero " +
                    empty_after(filtered && P.overlap_filter, filtered));

            double f = 1.0;
            if (norm == N_CPM) f = 1e6 / D;
            else if (norm == N_RPKM) {
                f = 1e9 / D;
                P.rpkm = true;
            }
            P.scale = scale_factor * f;
            P.scale_minus = scale_factor * scale_factor_minus * f;
        }
        if (!std::isfinite(P.scale) || !std::isfinite(P.scale_minus))
            throw std::runtime_error(
                "normalization produced a non-finite scale factor");

        // Per-thread worker: its own BAM/CRAM handle, index and record buffer.
        struct CovWorker {
            Alignments reads;
            BamRecordPtr rec;
            std::unique_ptr<MateLookup> mates;
            std::vector<std::pair<int64_t, int64_t>> iv;
            CoverageScratch scratch;
        };
        run_windows(
            windows, nthreads, P.nstreams, writers,
            [&]() {
                CovWorker w;
                w.reads = open_reads(bam_path, reference_fasta, true);
                w.rec.reset(bam_init1());
                if (!w.rec) throw std::runtime_error("could not allocate a BAM record");
                w.mates = std::make_unique<MateLookup>(
                    bam_path, P, mate_budget, blacklist_by_tid, whitelist_by_tid,
                    nonprimary_touch, fragment, containers.get());
                return w;
            },
            [&](CovWorker &w, const Window &win, std::vector<std::vector<float>> &out) {
                compute_window(w.reads, containers.get(), win, P, lengths[win.tid],
                               blacklist_by_tid[win.tid], whitelist_by_tid[win.tid],
                               *w.mates, w.rec.get(), w.iv, out, w.scratch);
                if (w.scratch.signal) signal.store(true, std::memory_order_relaxed);
            });

        for (auto &wr : writers) wr.finish();
    }

    finalize();
    // Only an all-zero listed run probes the same windows without the lists,
    // stopping at the first window with signal.
    bool lists_emptied = false;
    if ((P.has_blacklist || P.has_whitelist) && !signal.load()) {
        CovParams Pu = P;
        Pu.has_blacklist = Pu.has_whitelist = Pu.overlap_filter = false;
        const RegionSet none;
        Alignments probe_reads = open_reads(bam_path, reference_fasta, true);
        BamRecordPtr record(bam_init1());
        if (!record) throw std::runtime_error("could not allocate a BAM record");
        MateLookup mates(bam_path, Pu, mate_budget, blacklist_by_tid, whitelist_by_tid,
                         nonprimary_touch, fragment, containers.get());
        std::vector<std::pair<int64_t, int64_t>> iv;
        std::vector<std::vector<float>> vals;
        CoverageScratch scratch;
        for (const auto &win : windows) {
            dtp::check_python_signals();
            compute_window(probe_reads, containers.get(), win, Pu, lengths[win.tid], none, none,
                           mates, record.get(), iv, vals, scratch);
            if ((lists_emptied = scratch.signal)) break;
        }
    }
    if (!mate_budget_knob.is_none()) {
        auto knob = mate_budget_knob.cast<py::dict>();
        knob["peak_cache_entries"] = mate_budget.peak_cache_entries.load();
        knob["peak_cache_bytes"] = mate_budget.peak_cache_bytes.load();
        knob["peak_batch_entries"] = mate_budget.peak_batch_entries.load();
        knob["peak_batch_bytes"] = mate_budget.peak_batch_bytes.load();
        knob["peak_nonprimary_bytes"] = nonprimary_stats.peak_bytes;
        knob["nonprimary_passes"] = nonprimary_stats.passes;
        knob["nonprimary_spilled"] = nonprimary_stats.spilled;
        if (containers) knob["cram_container_decodes"] = containers->decodes();
    }
    return lists_emptied;
}

// --- P9: library-strandedness inference (RSeQC infer_experiment analogue) ------
//
// For reads over each user-supplied known-strand probe gene, tally how often the
// read's read1-implied transcript strand (under the "forward" = read1-sense
// hypothesis) matches the gene strand. A high fraction => forward/secondstrand
// library; low => reverse/firststrand (dUTP); ~0.5 => unstranded. Index-
// accelerated: only the probe regions are read.
std::string strand_record_key(const bam1_t *rec, bool canonical_template) {
    std::string result(bam_get_qname(rec));
    result.push_back('\0');
    const uint8_t *rg = bam_aux_get(rec, "RG");
    const char *group = rg ? bam_aux2Z(rg) : nullptr;
    if (group) result.append(group);
    result.push_back('\0');

    int64_t coordinates[4];
    if (canonical_template && (rec->core.flag & SAM_SECOND_IN_PAIR) &&
        !(rec->core.flag & SAM_FIRST_IN_PAIR)) {
        coordinates[0] = rec->core.mtid;
        coordinates[1] = rec->core.mpos;
        coordinates[2] = rec->core.tid;
        coordinates[3] = rec->core.pos;
    } else {
        coordinates[0] = rec->core.tid;
        coordinates[1] = rec->core.pos;
        coordinates[2] = rec->core.mtid;
        coordinates[3] = rec->core.mpos;
    }
    result.append(reinterpret_cast<const char *>(coordinates), sizeof(coordinates));
    if (!canonical_template) {
        const uint16_t identity_flags = rec->core.flag &
            (SAM_READ_PAIRED | SAM_FIRST_IN_PAIR | SAM_SECOND_IN_PAIR |
             SAM_READ_REVERSE | SAM_MATE_REVERSE);
        result.append(reinterpret_cast<const char *>(&identity_flags),
                      sizeof(identity_flags));
        const uint32_t *cigar = bam_get_cigar(rec);
        result.append(reinterpret_cast<const char *>(cigar),
                      rec->core.n_cigar * sizeof(uint32_t));
    }
    return result;
}

void record_strand_evidence(std::unordered_map<std::string, int8_t> &evidence,
                            std::string key, int8_t observation) {
    auto found = evidence.find(key);
    if (found == evidence.end()) {
        evidence.emplace(std::move(key), observation);
    } else if (found->second != observation) {
        found->second = 0;  // one template/read supports conflicting probe strands
    }
}

py::dict infer_strandedness(const std::string &bam_path, py::object regions,
                            int min_mapping_quality, bool require_proper_pair,
                            bool ignore_duplicates, const FlagPatterns &sam_flag_include,
                            uint16_t sam_flag_exclude, std::size_t sample_size,
                            const std::string &reference_fasta) {
    if (!sample_size || sample_size > 1000000)
        throw std::invalid_argument("strandedness sample_size must be in 1..1000000");
    if (min_mapping_quality < 0 || min_mapping_quality > 255)
        throw std::runtime_error("min_mapping_quality must be in 0..255");
    Alignments reads = open_reads(bam_path, reference_fasta, true);
    check_index(reads, bam_path);
    const auto &file = reads.file;
    const auto &hdr = reads.hdr;
    const auto &idx = reads.idx;

    struct Probe { int tid; int64_t start, end; int gene; };
    std::vector<Probe> probes;
    for (auto r : regions.cast<py::list>()) {
        auto t = r.cast<py::tuple>();
        const int tid = resolve_header_tid(hdr.get(), t[0].cast<std::string>());
        const int64_t start = t[1].cast<int64_t>(), end = t[2].cast<int64_t>();
        const int gene = t[3].cast<int>();
        if (tid >= 0 && start >= 0 && end > start && (gene == 1 || gene == -1))
            probes.push_back({tid, start, end, gene});
    }
    auto order = [](const Probe &a, const Probe &b) {
        return std::tie(a.tid, a.start, a.end, a.gene) < std::tie(b.tid, b.start, b.end, b.gene);
    };
    std::sort(probes.begin(), probes.end(), order);
    probes.erase(std::unique(probes.begin(), probes.end(), [&](const Probe &a, const Probe &b) {
        return !order(a, b) && !order(b, a);
    }), probes.end());

    // This is library-type inference, not a population census. Spread a fixed
    // observation budget over at most 128 index-weighted genomic strata instead of consuming
    // every alignment under every gene. Both accepted identities and examined
    // records are capped; sparse/repetitive/filtered inputs cannot make it scan
    // the whole BAM. Small inputs still visit all probes and all observations.
    const std::size_t strata = std::min<std::size_t>({probes.size(), sample_size, 128});
    const std::size_t base_quota = strata ? (sample_size + strata - 1) / strata : 0;
    std::vector<std::pair<std::size_t, std::size_t>> selection;
    if (probes.size() <= strata) {
        for (std::size_t i = 0; i < probes.size(); ++i) selection.emplace_back(i, base_quota);
    } else if (strata) {
        // Equal quotas per gene overweight low-expression genes. Compressed
        // index spans are a cheap proxy for read mass, used only to select
        // sampling strata; orientation counts still count each sampled template
        // once. No full-file scan is needed to construct these weights.
        std::vector<double> cumulative;
        double mass = 0;
        for (const auto &probe : probes) {
            IteratorPtr it;
            it.reset(sam_itr_queryi(idx.get(), probe.tid, probe.start, probe.end));
            if (!it.get()) throw std::runtime_error("could not query BAM region");
            double weight = 0;
            for (int j = 0; j < it.get()->n_off; ++j) {
                const auto &chunk = it.get()->off[j];
                if (chunk.v > chunk.u) weight += static_cast<double>(chunk.v - chunk.u) / 65536;
            }
            mass += std::max(1.0, weight);
            cumulative.push_back(mass);
        }
        for (std::size_t q = 0; q < strata; ++q) {
            const double target = (q + .5) * mass / strata;
            const auto i = static_cast<std::size_t>(
                std::lower_bound(cumulative.begin(), cumulative.end(), target) - cumulative.begin());
            if (!selection.empty() && selection.back().first == i) selection.back().second += base_quota;
            else selection.emplace_back(i, base_quota);
        }
    }
    std::unordered_map<std::string, int8_t> proper_evidence, single_evidence, alignment_evidence;
    bool saw_proper = false;
    BamRecordPtr record(bam_init1());
    if (!record.get()) throw std::runtime_error("could not allocate a BAM record");
    uint64_t scanned = 0;
    std::size_t stored = 0, sampled_probes = 0;
    std::vector<std::pair<int64_t, int64_t>> aligned_blocks;
    {
    py::gil_scoped_release release;
    for (const auto &stratum : selection) {
        if (stored >= sample_size) break;
        dtp::check_python_signals();
        const auto &probe = probes[stratum.first];
        const auto quota = stratum.second;
        const std::size_t scan_quota = std::max<std::size_t>(quota * 50, 256);
        IteratorPtr iterator;
        iterator.reset(sam_itr_queryi(idx.get(), probe.tid, probe.start, probe.end));
        if (!iterator.get()) throw std::runtime_error("could not query BAM region");
        ++sampled_probes;
        std::size_t accepted = 0, examined = 0;
        while (accepted < quota && examined < scan_quota && stored < sample_size &&
               checked_read(sam_itr_next(file.get(), iterator.get(), record.get())) >= 0) {
            ++examined;
            if ((++scanned & 4095u) == 0) dtp::check_python_signals();
            const uint16_t flag = record.get()->core.flag;
            if (flag & (SAM_READ_UNMAPPED | SAM_SECONDARY | SAM_SUPPLEMENTARY | SAM_QC_FAIL)) continue;
            if (record.get()->core.qual < min_mapping_quality) continue;
            if (ignore_duplicates && (flag & SAM_DUPLICATE)) continue;
            if (flag & sam_flag_exclude) continue;
            if (!flag_included(flag, sam_flag_include)) continue;
            // The BAM iterator retrieves reference spans, including skipped
            // introns. Only aligned bases can supply probe-strand evidence.
            extract_blocks(record.get(), aligned_blocks);
            if (std::none_of(aligned_blocks.begin(), aligned_blocks.end(),
                    [&](const auto &block) {
                        return block.first < probe.end && probe.start < block.second;
                    })) continue;
            const int8_t observation = transcript_strand(record.get(), Strand::Forward) == probe.gene ? 1 : -1;
            auto *evidence = &alignment_evidence;
            bool canonical = false;
            if (require_proper_pair) {
                if (flag & SAM_READ_PAIRED) {
                    if (!is_proper_fragment(record.get())) continue;
                    saw_proper = true;
                    evidence = &proper_evidence;
                    canonical = true;
                } else evidence = &single_evidence;
            }
            std::string key = strand_record_key(record.get(), canonical);
            if (key.size() > 4096) continue;
            const auto before = evidence->size();
            record_strand_evidence(*evidence, std::move(key), observation);
            const auto added = evidence->size() - before;
            stored += added; accepted += added;
        }
    }
    }

    const auto &chosen = require_proper_pair
        ? (saw_proper ? proper_evidence : single_evidence)
        : alignment_evidence;
    uint64_t fwd = 0, rev = 0;
    for (const auto &item : chosen) {
        if (item.second > 0) ++fwd;
        else if (item.second < 0) ++rev;
    }
    const uint64_t total = fwd + rev;

    py::dict out;
    out["forward"] = fwd;
    out["reverse"] = rev;
    out["total"] = total;
    out["sample_strategy"] = "index_weighted_probes";
    out["sample_limit"] = sample_size;
    out["sampled_probes"] = sampled_probes;
    out["alignments_examined"] = scanned;
    out["forward_fraction"] = total ? static_cast<double>(fwd) / total : 0.0;
    return out;
}

inline uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

uint64_t sampling_priority(const bam1_t *rec, uint64_t ordinal, uint64_t salt) {
    uint64_t hash = 1469598103934665603ULL;
    const unsigned char *name = reinterpret_cast<const unsigned char *>(bam_get_qname(rec));
    for (; *name; ++name) {
        hash ^= *name;
        hash *= 1099511628211ULL;
    }
    const uint8_t *rg_tag = bam_aux_get(rec, "RG");
    const unsigned char *group = reinterpret_cast<const unsigned char *>(
        rg_tag ? bam_aux2Z(rg_tag) : nullptr);
    if (group) {
        for (; *group; ++group) {
            hash ^= *group;
            hash *= 1099511628211ULL;
        }
    }
    const uint64_t fields[] = {
        static_cast<uint32_t>(rec->core.tid),
        static_cast<uint32_t>(rec->core.pos),
        static_cast<uint32_t>(rec->core.mtid),
        static_cast<uint32_t>(rec->core.mpos),
        static_cast<uint16_t>(rec->core.flag), ordinal, salt,
    };
    for (const uint64_t field : fields)
        hash = splitmix64(hash ^ splitmix64(field));
    return hash;
}

using LengthSample = std::priority_queue<std::pair<uint64_t, int64_t>>;

void offer_length(LengthSample &values, std::size_t capacity,
                  uint64_t priority, int64_t value) {
    const std::pair<uint64_t, int64_t> candidate{priority, value};
    if (values.size() < capacity) {
        values.push(candidate);
    } else if (candidate < values.top()) {
        values.pop();
        values.push(candidate);
    }
}

// Median fragment/read length from a deterministic bounded sample distributed
// over the complete BAM.  Coordinate-sorted prefixes can be chromosome- or
// library-component-biased, so stopping after the first N alignments is not a
// representative estimator.  A stable hash-priority reservoir remains O(sample)
// memory and produces byte-identical choices across supported platforms.
double estimate_length(const std::string &bam_path, int sample, bool prefer_fragment,
                       const std::string &reference_fasta) {
    if (sample <= 0) throw std::runtime_error("sample must be > 0");
    Alignments reads = open_reads(bam_path, reference_fasta, false);
    const auto &file = reads.file;
    const auto &hdr = reads.hdr;

    LengthSample tlens, rlens;
    BamRecordPtr record(bam_init1());
    if (!record.get()) throw std::runtime_error("could not allocate a BAM record");
    uint64_t scanned = 0;
    while (checked_read(sam_read1(file.get(), hdr.get(), record.get())) >= 0) {
        if ((++scanned & 4095u) == 0)
            dtp::check_python_signals();
        if (record.get()->core.flag & SAM_READ_UNMAPPED) continue;
        if (record.get()->core.flag & (SAM_SECONDARY | SAM_SUPPLEMENTARY)) continue;
        const int64_t rl = inferred_query_length(record.get());
        if (rl > 0) {
            offer_length(rlens, static_cast<std::size_t>(sample),
                         sampling_priority(record.get(), scanned, 0x72656164ULL), rl);
        }
        if (prefer_fragment && is_proper_fragment(record.get()) && record.get()->core.isize > 0)
            offer_length(tlens, static_cast<std::size_t>(sample),
                         sampling_priority(record.get(), scanned, 0x66726167ULL),
                         record.get()->core.isize);
    }

    auto median = [](LengthSample &sampled) -> double {
        std::vector<int64_t> v;
        v.reserve(sampled.size());
        while (!sampled.empty()) {
            v.push_back(sampled.top().second);
            sampled.pop();
        }
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        const size_t n = v.size(), m = n / 2;
        return (n & 1) ? static_cast<double>(v[m]) :
            static_cast<double>(v[m - 1]) / 2 + static_cast<double>(v[m]) / 2;
    };
    if (!tlens.empty()) return median(tlens);
    if (!rlens.empty()) return median(rlens);
    return 0;
}

}  // namespace

PYBIND11_MODULE(_coverage, m) {
    m.doc() = "Native bamCoverage backend.";
    m.def("htslib_version", &htslib_version,
          "Version string of the vendored htslib.");
    m.def("bam_index_stats", &bam_index_stats, py::arg("path"),
          py::arg("reference_fasta") = "",
          "Per-reference mapped/unmapped counts from the BAM index (no scan).");
    m.def("resolve_bam_index_path",
          [](const std::string &path) { return py::bytes(resolve_bam_index_path(path)); },
          py::arg("path"),
          "Path of the local BAM index the selected backend will load, or ''.");
    m.def("coverage_window_bins", &coverage_window_bins,
          py::arg("bin_size"), py::arg("requested") = 0,
          "Production-window bin count after the 262144-bin and 1-Mbp caps.");
    m.def("estimate_fragment_length",
          [](const std::string &path, int sample, const std::string &reference_fasta) {
              return estimate_length(path, sample, true, reference_fasta);
          },
          py::arg("bam_path"), py::arg("sample") = 10000, py::arg("reference_fasta") = "",
          "Median fragment length (proper-pair |TLEN|, else read length) from "
          "a representative bounded sample across the BAM.");
    m.def("infer_strandedness", &infer_strandedness, py::arg("bam_path"),
          py::arg("regions"), py::arg("min_mapping_quality") = 0,
          py::arg("require_proper_pair") = false,
          py::arg("ignore_duplicates") = false,
          py::arg("sam_flag_include") = FlagPatterns{},
          py::arg("sam_flag_exclude") = 0,
          py::arg("sample_size") = 20000,
          py::arg("reference_fasta") = "",
          "Sample independent read/template sense vs antisense evidence over "
          "known-strand probe genes (regions: list of (chrom, start, end, "
          "strand +1/-1)). Proper paired templates are counted once; if none "
          "are sampled, unpaired reads are used. At most sample_size identities "
          "are retained across up to 128 index-weighted genomic probe strata.");
    m.def("bam_coverage_bigwig", &bam_coverage_bigwig,
          py::arg("bam_path"), py::arg("out_path"),
          py::arg("out_path_reverse") = "",
          py::arg("out_format") = "bigwig",
          py::arg("bin_size") = 50,
          py::arg("aggregation") = "mean",
          py::arg("min_mapping_quality") = 0,
          py::arg("sam_flag_include") = FlagPatterns{},
          py::arg("sam_flag_exclude") = 0,
          py::arg("filter_mode") = "primary",
          py::arg("ignore_duplicates") = false,
          py::arg("min_fragment_length") = 0,
          py::arg("max_fragment_length") = 0,
          py::arg("extend_reads") = 0,
          py::arg("skip_introns") = false,
          py::arg("strandedness") = "none",
          py::arg("filter_rna_strand") = "none",
          py::arg("collapse") = "none",
          py::arg("collapsed_length") = 1,
          py::arg("count_unit") = "read",
          py::arg("blacklist") = py::none(),
          py::arg("whitelist") = py::none(),
          py::arg("threads") = 1,
          py::arg("window_size") = 0,
          py::arg("normalization") = "none",
          py::arg("normalization_denominator") = "library",
          py::arg("exact_scaling") = false,
          py::arg("scale_factor") = 1.0,
          py::arg("scale_factor_minus") = 1.0,
          py::arg("ignore_for_normalization") = py::none(),
          py::arg("max_zooms") = 10,
          py::arg("compression_level") = -1,
          py::arg("filter_by_overlap") = false,
          py::arg("feature_touch_strand") = "ignore",
          py::arg("filter_by_overlap_halo") = -1,
          py::arg("mate_budget") = py::none(),
          py::arg("reference_fasta") = "",
          "Binned coverage -> bigWig with inline filters, read extension/collapse, "
          "and strand selection. Multicore via a producer/single-writer streaming "
          "model (threads<=0 = all cores). aggregation: mean|sum|count. "
          "scale_factor_minus is an additional multiplier on scale_factor for the "
          "minus stream. Returns True when a black/whitelist removed all coverage "
          "that the run would have had without the lists.");
}
