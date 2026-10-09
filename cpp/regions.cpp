// BED/GTF region loader for computeMatrixR and computeMatrixOperationsR sort.
//
// Reads region files exactly as deeptoolsintervals 0.1.9 (deepTools 3.5.6)
// does: format detection, '#' group lines, label columns, unique names,
// duplicate transcripts, GTF exons/BED12 blocks and chromosome munging. The
// parser streams plain and gzip files through zlib in large chunks (bzip2 is
// fed by Python), keeps one interned chromosome table, one name arena and flat
// region/exon records (12 bytes per exon), and answers the only queries the
// tools make: every region of a chromosome in findOverlaps order and each
// group's names in file order. It also reads the blacklist/whitelist BED files
// of computeMatrixR and bamCoverageR (`Blacklist`).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <zlib.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "display_path.hpp"

namespace py = pybind11;

namespace {

using Reader = std::function<std::size_t(char *, std::size_t)>;
constexpr std::size_t CHUNK = 1 << 20;
constexpr int64_t COORD_LIMIT = std::numeric_limits<uint32_t>::max();  // tree limit

// Python str.strip() / int() whitespace (ASCII part).
bool is_space(char c) {
    return c == ' ' || (c >= '\t' && c <= '\r') || (c >= '\x1c' && c <= '\x1f');
}

std::string_view strip(std::string_view s) {
    while (!s.empty() && is_space(s.front())) s.remove_prefix(1);
    while (!s.empty() && is_space(s.back())) s.remove_suffix(1);
    return s;
}

bool starts_with(std::string_view s, std::string_view prefix) {
    return s.substr(0, prefix.size()) == prefix;
}

bool iequals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::tolower(static_cast<unsigned char>(a[i])) !=
            std::tolower(static_cast<unsigned char>(b[i])))
            return false;
    return true;
}

void split(std::string_view line, char sep, std::vector<std::string_view> &out) {
    out.clear();
    for (std::size_t pos;;) {
        pos = line.find(sep);
        out.push_back(line.substr(0, pos));
        if (pos == std::string_view::npos) return;
        line.remove_prefix(pos + 1);
    }
}

// A line quoted in a message: at most 200 bytes, as valid UTF-8.
std::string excerpt(std::string_view text) {
    constexpr std::size_t limit = 200;
    if (text.size() <= limit) return dtp::valid_utf8(std::string(text));
    return dtp::valid_utf8(std::string(text.substr(0, limit))) + "\u2026";
}

std::string join_tabs(const std::vector<std::string_view> &cols) {
    std::string out;
    for (std::size_t i = 0; i < cols.size() && out.size() <= 200; ++i) {
        if (i) out += '\t';
        out += cols[i];
    }
    return excerpt(out);
}

// Python int(): surrounding whitespace, a sign, digits with single '_'.
std::optional<int64_t> py_int(std::string_view s) {
    s = strip(s);
    bool negative = false;
    if (!s.empty() && (s[0] == '+' || s[0] == '-')) {
        negative = s[0] == '-';
        s.remove_prefix(1);
    }
    if (s.empty() || s.front() == '_' || s.back() == '_') return std::nullopt;
    int64_t value = 0;
    char previous = 0;
    for (char c : s) {
        if (c == '_') {
            if (previous == '_') return std::nullopt;
        } else if (c >= '0' && c <= '9') {
            if (value < (std::numeric_limits<int64_t>::max() - 9) / 10)
                value = value * 10 + (c - '0');
            else
                value = std::numeric_limits<int64_t>::max();  // beyond every limit used here
        } else {
            return std::nullopt;
        }
        previous = c;
    }
    return negative ? -value : value;
}

int64_t require_int(std::string_view s, const std::string &file) {
    if (auto value = py_int(s)) return *value;
    throw py::value_error(file + ": invalid literal for int() with base 10: '" + excerpt(s) + "'");
}

// Python float() acceptance (decimal forms, inf/nan); no hexadecimal.
bool py_float(std::string_view s) {
    std::string text(strip(s));
    if (text.empty() || text.find_first_of("xX") != std::string::npos) return false;
    text.erase(std::remove(text.begin(), text.end(), '_'), text.end());
    char *end = nullptr;
    std::strtod(text.c_str(), &end);
    return end == text.c_str() + text.size();
}

// csv.reader([field], delimiter=' ') with the default (non-strict) dialect.
// Tokens are written unquoted into one buffer, which never outgrows the field.
class Tokens {
public:
    bool parse(std::string_view field) {
        buffer_.resize(field.size());
        spans_.clear();
        enum { START, FIELD, QUOTED, QUOTE_IN_QUOTED, EAT_CRNL } state = START;
        char *const base = buffer_.data();
        char *out = base, *begin = base;
        auto save = [&] {
            spans_.emplace_back(begin - base, out - begin);
            begin = out;
        };
        bool record_started = false;
        for (const char *p = field.data(), *end = p + field.size(); p < end; ++p) {
            char c = *p;
            bool newline = c == '\n' || c == '\r';
            switch (state) {
            case START:
                if (newline) {
                    if (record_started) save();
                    state = EAT_CRNL;
                } else if (c == '"') {
                    state = QUOTED;
                } else if (c == ' ') {
                    save();
                } else {
                    *out++ = c;
                    state = FIELD;
                }
                record_started = true;
                break;
            case FIELD:
                if (newline) { save(); state = EAT_CRNL; }
                else if (c == ' ') { save(); state = START; }
                else *out++ = c;
                break;
            case QUOTED: {  // copy up to the next quote in one run
                const char *quote = static_cast<const char *>(std::memchr(p, '"', end - p));
                const char *stop = quote ? quote : end;
                std::memcpy(out, p, stop - p);
                out += stop - p;
                if (!quote) p = end - 1;
                else { p = quote; state = QUOTE_IN_QUOTED; }
                break;
            }
            case QUOTE_IN_QUOTED:
                if (c == '"') { *out++ = c; state = QUOTED; }
                else if (c == ' ') { save(); state = START; }
                else if (newline) { save(); state = EAT_CRNL; }
                else { *out++ = c; state = FIELD; }
                break;
            case EAT_CRNL:
                if (!newline) return false;  // new-line character seen in unquoted field
                break;
            }
        }
        if (state != EAT_CRNL && record_started) save();
        return true;
    }
    std::size_t size() const { return spans_.size(); }
    std::string_view operator[](std::size_t i) const {
        return std::string_view(buffer_).substr(spans_[i].first, spans_[i].second);
    }
    std::optional<std::size_t> find(std::string_view key) const {
        for (std::size_t i = 0; i < size(); ++i)
            if ((*this)[i] == key) return i;
        return std::nullopt;
    }
    // `key in s and s[-1] != key`: the value token after the first `key`.
    std::optional<std::string_view> value(std::string_view key) const {
        auto index = find(key);
        if (!index || (*this)[size() - 1] == key) return std::nullopt;
        std::string_view token = (*this)[*index + 1];
        while (!token.empty() && token.back() == ';') token.remove_suffix(1);
        return token;
    }

private:
    std::string buffer_;
    std::vector<std::pair<std::size_t, std::size_t>> spans_;
};

// Lines of a stream without the newline ('\r\n' counts as one newline, as in
// Python's text mode); the final line need not end with a newline.
class LineReader {
public:
    explicit LineReader(Reader read) : read_(std::move(read)), buffer_(CHUNK) {}

    bool next(std::string_view &line) {
        for (;;) {
            const char *newline = static_cast<const char *>(
                std::memchr(buffer_.data() + pos_, '\n', end_ - pos_));
            if (newline) {
                std::size_t at = newline - buffer_.data();
                line = trim_cr(std::string_view(buffer_.data() + pos_, at - pos_));
                pos_ = at + 1;
                return true;
            }
            if (eof_) {
                if (pos_ == end_) return false;
                line = trim_cr(std::string_view(buffer_.data() + pos_, end_ - pos_));
                pos_ = end_;
                return true;
            }
            fill();
        }
    }

private:
    static std::string_view trim_cr(std::string_view s) {
        if (!s.empty() && s.back() == '\r') s.remove_suffix(1);
        return s;
    }
    void fill() {
        std::memmove(buffer_.data(), buffer_.data() + pos_, end_ - pos_);
        end_ -= pos_;
        pos_ = 0;
        if (buffer_.size() - end_ < CHUNK / 2) buffer_.resize(buffer_.size() * 2);
        std::size_t got = read_(buffer_.data() + end_, buffer_.size() - end_);
        end_ += got;
        eof_ = got == 0;
    }
    Reader read_;
    std::vector<char> buffer_;
    std::size_t pos_ = 0, end_ = 0;
    bool eof_ = false;
};

// Stable storage for names: views stay valid while the set lives.
class Arena {
public:
    std::string_view store(std::string_view s) {
        if (s.empty()) return {};
        if (s.size() > capacity_ - used_) {
            capacity_ = std::max<std::size_t>(CHUNK, s.size());
            blocks_.emplace_back(new char[capacity_]);
            used_ = 0;
        }
        char *dest = blocks_.back().get() + used_;
        std::memcpy(dest, s.data(), s.size());
        used_ += s.size();
        return std::string_view(dest, s.size());
    }

private:
    std::vector<std::unique_ptr<char[]>> blocks_;
    std::size_t used_ = 0, capacity_ = 0;
};

struct Region {
    std::string_view name;
    double score;  // valid when has_score; otherwise the score is '.'
    uint32_t start, end;
    int32_t chrom, group, slot;
    char strand;
    bool has_score;
};

struct Exon {
    int32_t slot;
    uint32_t start, end;
};

template <class Contains>
std::string random_label(std::string_view name, Contains contains) {
    std::string label(name);
    for (int i = 1; contains(label); ++i) label = std::string(name) + "_r" + std::to_string(i);
    return label;
}

std::string basename(const std::string &path) {
#ifdef _WIN32
    auto slash = path.find_last_of("/\\:");  // ntpath.basename
#else
    auto slash = path.find_last_of('/');
#endif
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

void warn(const std::string &message) {
    py::module_::import("sys").attr("stderr").attr("write")(message + "\n");
}

// Feed `parse` the lines of a plain or gzip file.
template <class Parse>
void read_gzip(const std::string &path, Parse parse) {
    gzFile file = gzopen(path.c_str(), "rb");
    if (!file) throw std::runtime_error("cannot open " + dtp::display_path(path) + ": " + std::strerror(errno));
    std::unique_ptr<gzFile_s, int (*)(gzFile)> guard(file, gzclose);
    gzbuffer(file, CHUNK);
    parse(LineReader([&](char *dest, std::size_t size) {
        int got = gzread(file, dest, static_cast<unsigned>(std::min<std::size_t>(size, 1u << 30)));
        // zlib >= 1.3.2 reports a truncated stream only alongside the read that
        // hit it (possibly with data) and clears it on the next gzread: check every read.
        int code = Z_OK;
        const char *message = gzerror(file, &code);
        if (got < 0 || code != Z_OK) {
            // zlib prefixes its message with the path it was given
            std::string_view text = message;
            if (starts_with(text, path + ": ")) text.remove_prefix(path.size() + 2);
            throw std::runtime_error(dtp::display_path(path) + ": " +
                                     (code == Z_BUF_ERROR ? "truncated gzip file" : std::string(text)));
        }
        return static_cast<std::size_t>(got);
    }));
}

// Feed `parse` the lines of an already-decompressing stream: `read(size) -> bytes` (bzip2).
template <class Parse>
void read_python(py::object read_bytes, Parse parse) {
    parse(LineReader([&](char *dest, std::size_t size) {
        py::bytes chunk = read_bytes(size);
        std::string_view data(chunk);
        if (data.size() > size) throw std::runtime_error("stream returned too many bytes");
        std::memcpy(dest, data.data(), data.size());
        return data.size();
    }));
}

// mungeChromosome: `chrom`, or its MT/chrM or chr-prefix alias when only that is `known`.
template <class Known>
std::string munge_chrom(const std::string &chrom, Known known) {
    if (known(chrom)) return chrom;
    if (chrom == "MT" && known("chrM")) return "chrM";
    if (chrom == "chrM" && known("MT")) return "MT";
    if (starts_with(chrom, "chr") && chrom.size() > 3 && known(chrom.substr(3))) return chrom.substr(3);
    if (known("chr" + chrom)) return "chr" + chrom;
    return chrom;
}

class RegionSet {
public:
    RegionSet(std::string transcript_id, std::string exon_id, std::string designator,
              bool keep_exons, std::optional<std::string> default_group)
        : transcript_id_(std::move(transcript_id)), exon_id_(std::move(exon_id)),
          designator_(std::move(designator)), keep_exons_(keep_exons),
          default_group_(std::move(default_group)) {}

    void read_file(const std::string &path) {
        read_gzip(path, [&](LineReader lines) { read(std::move(lines), path); });
    }
    void read_stream(py::object read_bytes, const std::string &path) {
        read_python(std::move(read_bytes), [&](LineReader lines) { read(std::move(lines), path); });
    }

    void finish() {
        if (regions_.empty())
            throw std::runtime_error("None of the input BED/GTF files had valid regions");
        // findOverlaps sorts its result tuples: start, end, name, group index.
        order_.resize(regions_.size());
        for (std::size_t i = 0; i < order_.size(); ++i) order_[i] = static_cast<int32_t>(i);
        std::sort(order_.begin(), order_.end(), [&](int32_t a, int32_t b) {
            const Region &x = regions_[a], &y = regions_[b];
            if (x.chrom != y.chrom) return x.chrom < y.chrom;
            if (x.start != y.start) return x.start < y.start;
            if (x.end != y.end) return x.end < y.end;
            if (x.name != y.name) return x.name < y.name;
            return x.group < y.group;
        });
        chrom_offsets_.assign(chroms_.size() + 1, 0);
        for (const Region &region : regions_) ++chrom_offsets_[region.chrom + 1];
        for (std::size_t i = 1; i < chrom_offsets_.size(); ++i) chrom_offsets_[i] += chrom_offsets_[i - 1];
        std::sort(exons_.begin(), exons_.end(), [](const Exon &a, const Exon &b) {
            return std::tie(a.slot, a.start, a.end) < std::tie(b.slot, b.start, b.end);
        });
        exon_offsets_.assign(slots_ + 1, 0);
        for (const Exon &exon : exons_) ++exon_offsets_[exon.slot + 1];
        for (std::size_t i = 1; i < exon_offsets_.size(); ++i) exon_offsets_[i] += exon_offsets_[i - 1];
    }

    // mungeChromosome(chrom, append=False)
    std::string munge(const std::string &chrom) const {
        return munge_chrom(chrom, [&](const std::string &name) { return chrom_index_.count(name) > 0; });
    }

    // Regions of `chrom` starting before `end`, in findOverlaps order, as
    // computeMatrixR rows [chrom, exons, name, group, strand, score].
    py::list rows(const std::string &chrom, int64_t end) const {
        py::list out;
        auto found = chrom_index_.find(munge(chrom));
        if (found == chrom_index_.end()) return out;
        py::str chrom_name(chrom);
        for (int32_t o = chrom_offsets_[found->second]; o < chrom_offsets_[found->second + 1]; ++o) {
            if (regions_[order_[o]].start >= end) break;
            out.append(row(chrom_name, order_[o]));
        }
        return out;
    }

    // Each group's region names in file order.
    std::vector<std::vector<std::string_view>> group_names() const {
        std::vector<std::vector<std::string_view>> names(labels_.size());
        for (const Region &region : regions_) names[region.group].push_back(region.name);
        return names;
    }

    // A GTF read as a blacklist: the same transcript entries, from the
    // first data line `line` on, passed to `add(chrom, start, end, strand)`.
    template <class Add>
    void read_gtf_entries(LineReader &lines, std::string_view line, const std::string &path, Add add) {
        filename_ = dtp::display_path(path);
        parse_gtf(lines, line);
        for (const Region &region : regions_) add(chroms_[region.chrom], region.start, region.end, region.strand);
    }
    bool is_gtf(const std::vector<std::string_view> &cols) { return cols.size() > 6 && seems_like_gtf(cols); }

    const std::vector<std::string> &labels() const { return labels_; }
    const std::vector<std::string> &chroms() const { return chroms_; }
    std::size_t size() const { return regions_.size(); }

private:
    using Names = std::unordered_map<std::string_view, int32_t>;

    py::tuple row(const py::str &chrom, int32_t index) const {
        const Region &region = regions_[index];
        py::list exons;
        int32_t first = exon_offsets_[region.slot], last = exon_offsets_[region.slot + 1];
        if (first == last) exons.append(py::make_tuple(region.start, region.end));
        for (int32_t e = first; e < last; ++e) exons.append(py::make_tuple(exons_[e].start, exons_[e].end));
        py::object score = region.has_score ? py::object(py::float_(region.score)) : py::object(py::str("."));
        return py::make_tuple(chrom, exons, py::str(region.name.data(), region.name.size()),
                              region.group, py::str(std::string(1, region.strand)), score);
    }

    void read(LineReader lines, const std::string &path) {
        filename_ = dtp::display_path(path);
        std::string_view line;
        std::optional<std::size_t> label_column;
        bool have_line;
        // Skip the header, which may name a deepTools_group column.
        while ((have_line = lines.next(line)) &&
               (starts_with(line, "#") || starts_with(line, "track") || starts_with(line, "browser"))) {
            if (!label_column) {
                split(strip(line), '\t', cols_);
                auto at = std::find(cols_.begin(), cols_.end(), "deepTools_group");
                if (at != cols_.end()) label_column = at - cols_.begin();
            }
        }
        line = strip(have_line ? line : std::string_view());
        split(line, '\t', cols_);
        std::size_t ncols = cols_.size() - (label_column ? 1 : 0);
        if (line.empty() || ncols < 3)
            throw std::runtime_error(filename_ + " does not seem to be a recognized file type!");
        if (ncols > 6 && seems_like_gtf(cols_))
            parse_gtf(lines, line);
        else
            parse_bed(lines, line, ncols < 6 ? 3 : ncols == 6 ? 6 : ncols <= 11 ? 6 : 12, label_column);
        label_idx_ = static_cast<int32_t>(labels_.size());
    }

    bool seems_like_gtf(const std::vector<std::string_view> &cols) {
        if (cols.size() < 9 || !py_int(cols[3]) || !py_int(cols[4])) return false;
        if (cols[5] != "." && !py_float(cols[5])) return false;
        if (cols[7] != "." && !py_int(cols[7])) return false;
        if (!tokens_.parse(cols[8])) return false;
        return tokens_.find("gene_id") && tokens_[tokens_.size() - 1] != "gene_id";
    }

    // --- labels, names, chromosomes -------------------------------------
    std::optional<int32_t> label_index(std::string_view label) const {
        auto at = std::find(labels_.begin(), labels_.end(), label);
        if (at == labels_.end()) return std::nullopt;
        return static_cast<int32_t>(at - labels_.begin());
    }
    std::string random_group_label(std::string_view label) const {
        return random_label(label, [&](const std::string &l) { return label_index(l).has_value(); });
    }
    Names &names(int32_t group) {
        if (static_cast<std::size_t>(group) >= names_.size()) names_.resize(group + 1);
        return names_[group];
    }
    int32_t new_slot(int32_t group, std::string_view name) {
        int32_t slot = slots_++;
        names(group).emplace(arena_.store(name), slot);
        return slot;
    }
    // mungeChromosome(chrom, append=True)
    int32_t intern_chrom(std::string_view chrom) {
        std::string name = munge(std::string(chrom));
        auto found = chrom_index_.find(name);
        if (found != chrom_index_.end()) return found->second;
        chroms_.push_back(name);
        return chrom_index_[name] = static_cast<int32_t>(chroms_.size() - 1);
    }
    void add_region(std::string_view chrom, int64_t start, int64_t end, std::string_view name,
                    std::string_view strand, std::string_view score) {
        if (start >= COORD_LIMIT || end >= COORD_LIMIT)
            throw std::runtime_error("Coordinates beyond 2^32 - 1 in " + filename_);
        int32_t slot = new_slot(label_idx_, name);
        Region region{};
        region.name = names(label_idx_).find(name)->first;
        region.start = static_cast<uint32_t>(start);
        region.end = static_cast<uint32_t>(end);
        region.chrom = intern_chrom(chrom);
        region.group = label_idx_;
        region.slot = slot;
        region.strand = strand == "+" ? '+' : strand == "-" ? '-' : '.';
        region.has_score = score != ".";
        if (region.has_score) region.score = std::strtod(std::string(score).c_str(), nullptr);
        regions_.push_back(region);
    }
    void add_exon(int32_t slot, int64_t start, int64_t end) {
        if (start < 0 || end < 0 || start >= COORD_LIMIT || end >= COORD_LIMIT)
            throw std::runtime_error("Exon coordinates outside [0, 2^32 - 1) in " + filename_);
        exons_.push_back(Exon{slot, static_cast<uint32_t>(start), static_cast<uint32_t>(end)});
    }

    // --- BED --------------------------------------------------------------
    void use_label_column(std::size_t column) {
        if (column >= cols_.size()) throw std::runtime_error("Missing deepTools_group column in " + filename_);
        std::string label(cols_[column]);
        cols_.erase(cols_.begin() + column);
        auto index = label_index(label);
        if (!index) {
            labels_.push_back(label);
            index = static_cast<int32_t>(labels_.size() - 1);
        }
        label_idx_ = *index;
    }

    void parse_bed(LineReader &lines, std::string_view line, std::size_t ncols,
                   std::optional<std::size_t> label_column) {
        bool group_open = false;  // a '#' line closed a group
        std::size_t group_entries = 0;
        if (label_column) use_label_column(*label_column);
        parse_bed_core(ncols);
        group_entries = 1;
        while (lines.next(line)) {
            line = strip(line);
            if (line.empty()) continue;
            if (line[0] == '#') {
                if (label_column) continue;
                // A '#' line names the entries since the previous one.
                if (group_open && group_entries == 0) {
                    warn("Warning, the '" + std::string(strip(line.substr(1))) +
                         "' group had no valid entries! Removing it.");
                    continue;
                }
                std::string_view label = strip(line.substr(1));
                labels_.push_back(random_group_label(label.empty() ? basename(filename_) : std::string(label)));
                ++label_idx_;
                group_open = true;
                group_entries = 0;
                continue;
            }
            split(line, '\t', cols_);
            if (label_column) use_label_column(*label_column);
            parse_bed_core(ncols);
            if (!label_column) ++group_entries;
        }
        if (group_entries > 0 && !label_column)
            labels_.push_back(random_group_label(default_group_ ? *default_group_ : basename(filename_)));
    }

    void parse_bed_core(std::size_t ncols) {
        if (ncols <= 3 && cols_.size() < 3)
            throw std::runtime_error("BED line with fewer than 3 columns in " + filename_ + ": " +
                                     join_tabs(cols_));
        if (ncols > 3 && cols_.size() < 6)
            throw std::runtime_error("BED line with " + std::to_string(cols_.size()) + " columns in " +
                                     filename_ + ", whose first region line has " + std::to_string(ncols) +
                                     " (6 or more needed): " + join_tabs(cols_));
        std::string name = std::string(cols_[0]) + ":" + std::string(cols_[1]) + "-" + std::string(cols_[2]);
        int64_t start = require_int(cols_[1], filename_);
        std::string start_text(cols_[1]);
        if (start < 0) {
            start = 0;
            start_text = "0";
        }
        int64_t end = require_int(cols_[2], filename_);
        if (start >= end) {
            warn("Warning: " + std::string(cols_[0]) + ":" + start_text + "-" + std::string(cols_[2]) +
                 " is an invalid BED interval! Ignoring it.");
            return;
        }
        std::string_view strand = ".", score = ".";
        if (ncols > 3) {
            name = std::string(cols_[3]);
            strand = cols_[5];
            score = cols_[4];
        }
        Names &group = names(label_idx_);
        name = random_label(name, [&](const std::string &n) { return group.count(n) > 0; });
        add_region(cols_[0], start, end, name, strand, score);
        if (ncols == 12 && keep_exons_) {
            if (cols_.size() != 12) throw std::runtime_error("BED12 line without 12 columns in " + filename_);
            add_blocks(regions_.back().slot, start, end, require_int(cols_[9], filename_), cols_[10], cols_[11]);
        }
    }

    // parseExonBounds: invalid or too few blocks fall back to the entry bounds.
    void add_blocks(int32_t slot, int64_t start, int64_t end, int64_t count,
                    std::string_view sizes_text, std::string_view offsets_text) {
        auto list = [&](std::string_view text) {
            while (!text.empty() && text.front() == ',') text.remove_prefix(1);
            while (!text.empty() && text.back() == ',') text.remove_suffix(1);
            std::vector<std::string_view> items;
            split(text, ',', items);
            int64_t n = static_cast<int64_t>(items.size());
            int64_t keep = count < 0 ? std::max<int64_t>(0, n + count) : std::min(count, n);
            items.resize(static_cast<std::size_t>(keep));
            return items;
        };
        auto repr = [](const std::vector<std::string_view> &items) {
            std::string out = "[";
            for (std::size_t i = 0; i < items.size(); ++i)
                out += (i ? ", '" : "'") + std::string(items[i]) + "'";
            return out + "]";
        };
        std::vector<std::string_view> offsets = list(offsets_text), sizes = list(sizes_text);
        std::vector<std::pair<int64_t, int64_t>> blocks;
        bool valid = std::all_of(offsets.begin(), offsets.end(), [](auto x) { return py_int(x).has_value(); });
        for (std::size_t i = 0; valid && i < std::min(offsets.size(), sizes.size()); ++i) {
            auto size = py_int(sizes[i]);
            int64_t block_start = start + *py_int(offsets[i]);
            if (size) blocks.emplace_back(block_start, block_start + *size);
            valid = size.has_value();
        }
        if (!valid) {
            warn("Warning: Received an invalid exon offset (" + repr(offsets) + ") or size (" + repr(sizes) +
                 "), using the entry bounds instead (" + std::to_string(start) + "-" + std::to_string(end) + ")");
            return;
        }
        if (static_cast<int64_t>(offsets.size()) < count || static_cast<int64_t>(sizes.size()) < count) {
            warn("Warning: There were too few exon start/end offsets (" + repr(offsets) + ") or sizes (" +
                 repr(sizes) + "), using the entry bounds instead (" + std::to_string(start) + "-" +
                 std::to_string(end) + ")");
            return;
        }
        for (auto [s, e] : blocks) add_exon(slot, s, e);
    }

    // --- GTF --------------------------------------------------------------
    void parse_gtf(LineReader &lines, std::string_view line) {
        std::string file_label = random_group_label(basename(filename_));
        split(line, '\t', cols_);
        if (iequals(cols_[2], transcript_id_))
            parse_transcript(file_label);
        else if (iequals(cols_[2], exon_id_))
            parse_exon();
        while (lines.next(line)) {
            if (starts_with(line, "#")) continue;
            split(line, '\t', cols_);
            if (cols_.size() < 3) throw std::runtime_error("Malformed GTF line in " + filename_ + ": " + excerpt(line));
            if (iequals(cols_[2], transcript_id_))
                parse_transcript(file_label);
            else if (keep_exons_ && iequals(cols_[2], exon_id_))
                parse_exon();
        }
    }

    bool valid_start(int64_t start) {
        if (start - 1 >= 0) return true;
        warn("Warning: Invalid start in '" + join_tabs(cols_) + "', skipping");
        return false;
    }
    void parse_attributes() {
        if (!tokens_.parse(cols_[8]))
            throw std::runtime_error("Malformed GTF attributes in " + filename_ + ": " + join_tabs(cols_));
    }

    void parse_transcript(const std::string &file_label) {
        if (cols_.size() < 4) return;  // like deeptoolsintervals: no coordinates, silently skipped
        int64_t start = require_int(cols_[3], filename_);
        if (!valid_start(start)) return;
        if (cols_.size() < 9) {
            warn("Warning: non-GTF line encountered! " + join_tabs(cols_));
            return;
        }
        parse_attributes();
        std::string label = file_label;
        if (auto group = tokens_.value("deepTools_group"))
            label = *group;
        else if (default_group_)
            label = *default_group_;
        auto name = tokens_.value(designator_);
        if (!name) {
            warn("Warning: " + join_tabs(cols_) + " is malformed!");
            return;
        }
        int64_t end = require_int(cols_[4], filename_);
        if (start > end || start < 1) {
            warn("Warning: " + std::string(cols_[0]) + ":" + std::string(cols_[3]) + "-" + std::string(cols_[4]) +
                 " is an invalid GTF interval! Ignoring it.");
            return;
        }
        auto index = label_index(label);
        if (!index) {
            labels_.push_back(label);
            index = static_cast<int32_t>(labels_.size() - 1);
        }
        label_idx_ = *index;
        if (names(label_idx_).count(*name)) {
            warn("Warning: " + std::string(*name) + " occurs more than once! Only using the first instance.");
            if (!duplicated_.count(*name)) duplicated_.emplace(arena_.store(*name));
            return;
        }
        add_region(cols_[0], start - 1, end, *name, cols_[6], cols_[5]);
    }

    void parse_exon() {
        if (cols_.size() < 4) throw std::runtime_error("Malformed GTF line in " + filename_ + ": " + join_tabs(cols_));
        int64_t start = require_int(cols_[3], filename_);
        if (!valid_start(start)) return;
        if (cols_.size() < 9) throw std::runtime_error("Malformed GTF line in " + filename_ + ": " + join_tabs(cols_));
        parse_attributes();
        auto name = tokens_.value(designator_);
        if (!name) {
            warn("Warning: " + join_tabs(cols_) + " is malformed!");
            return;
        }
        if (duplicated_.count(*name)) return;
        Names &group = names(label_idx_);
        auto found = group.find(*name);
        int32_t slot = found != group.end() ? found->second : new_slot(label_idx_, *name);
        add_exon(slot, start - 1, require_int(cols_[4], filename_));
    }

    std::string transcript_id_, exon_id_, designator_;
    bool keep_exons_;
    std::optional<std::string> default_group_;

    std::string filename_;
    std::vector<std::string_view> cols_;
    Tokens tokens_;
    Arena arena_;

    std::vector<std::string> labels_;
    int32_t label_idx_ = 0;
    std::vector<Names> names_;               // per group: name -> slot
    int32_t slots_ = 0;                      // (group, name) keys; exons attach to a slot
    std::unordered_set<std::string_view> duplicated_;
    std::vector<std::string> chroms_;
    std::unordered_map<std::string, int32_t> chrom_index_;
    std::vector<Region> regions_;
    std::vector<Exon> exons_;

    std::vector<int32_t> order_, chrom_offsets_, exon_offsets_;
};

// A blacklist or whitelist: a BED (chromosome, start, end and the optional
// strand, column 6) or a GTF (its transcripts, as the regions read them). Fields are tab-separated, or whitespace-separated on a line
// without a tab. Per chromosome name as written, the intervals are merged
// into three sets: every entry ('.'), and the entries
// that apply to each strand ('+'/'-': that strand's entries plus those
// without a +/- strand).
class Blacklist {
public:
    using Intervals = std::vector<std::pair<int64_t, int64_t>>;

    // merge_aliases: chr1/1 and chrM/MT entries join the first-seen name, as
    // the region loader munges chromosomes (computeMatrixR); otherwise every
    // name is kept as written (bamCoverageR resolves them against the BAM).
    explicit Blacklist(bool merge_aliases) : merge_aliases_(merge_aliases) {}

    void read_file(const std::string &path) {
        read_gzip(path, [&](LineReader lines) { read(lines, path); });
    }
    void read_stream(py::object read_bytes, const std::string &path) {
        read_python(std::move(read_bytes), [&](LineReader lines) { read(lines, path); });
    }

    void finish() {
        for (auto &sets : sets_)
            for (Intervals &set : sets) {
                std::sort(set.begin(), set.end());
                Intervals merged;
                for (const auto &interval : set) {
                    if (!merged.empty() && interval.first <= merged.back().second)
                        merged.back().second = std::max(merged.back().second, interval.second);
                    else
                        merged.push_back(interval);
                }
                set.swap(merged);
            }
    }

    // The merged intervals of `chrom` (or its alias, as the regions munge it)
    // applying to `strand` ('.' = all).
    Intervals intervals(const std::string &chrom, char strand) const {
        const Intervals *set = find(chrom, strand);
        return set ? *set : Intervals();
    }

    // The merged intervals of `chrom` applying to `strand` that share a base
    // with one of `segments`.
    Intervals overlaps(const std::string &chrom, const Intervals &segments, char strand) const {
        Intervals out;
        const Intervals *set = find(chrom, strand);
        if (!set) return out;
        for (auto [start, end] : segments) {
            auto it = std::upper_bound(set->begin(), set->end(), start,
                                       [](int64_t value, const auto &interval) { return value < interval.second; });
            for (; it != set->end() && it->first < end; ++it) out.push_back(*it);
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    const std::vector<std::string> &chroms() const { return chroms_; }

private:
    const Intervals *find(const std::string &chrom, char strand) const {
        auto found = index_.find(munge_chrom(chrom, [&](const std::string &name) { return index_.count(name) > 0; }));
        if (found == index_.end()) return nullptr;
        return &sets_[found->second][strand == '+' ? 1 : strand == '-' ? 2 : 0];
    }

    void read(LineReader &lines, const std::string &path) {
        const std::string file = dtp::display_path(path);
        std::string_view line;
        std::vector<std::string_view> cols;
        bool first = true;
        while (lines.next(line)) {
            line = strip(line);
            if (line.empty() || starts_with(line, "#") || starts_with(line, "track") || starts_with(line, "browser"))
                continue;
            if (line.find('\t') != std::string_view::npos) {
                split(line, '\t', cols);
            } else {
                cols.clear();
                for (std::size_t pos = 0; pos < line.size();) {
                    std::size_t end = pos;
                    while (end < line.size() && !is_space(line[end])) ++end;
                    cols.push_back(line.substr(pos, end - pos));
                    for (pos = end; pos < line.size() && is_space(line[pos]);) ++pos;
                }
            }
            if (std::exchange(first, false)) {
                // A GTF (detected as the region loader does) gives its transcripts.
                RegionSet gtf("transcript", "exon", "transcript_id", false, std::nullopt);
                if (gtf.is_gtf(cols)) {
                    gtf.read_gtf_entries(lines, line, path, [&](const std::string &chrom, int64_t start,
                                                                int64_t end, char strand) {
                        add(chrom, start, end, std::string_view(&strand, 1));
                    });
                    return;
                }
            }
            if (cols.size() < 3)
                throw std::runtime_error("BED line with fewer than 3 columns in " + file +
                                         ": " + excerpt(line));
            int64_t start = std::max<int64_t>(0, require_int(cols[1], file)), end = require_int(cols[2], file);
            add(std::string(cols[0]), start, end, cols.size() >= 6 ? strip(cols[5]) : std::string_view("."));
        }
    }

    void add(const std::string &written, int64_t start, int64_t end, std::string_view strand) {
        if (end <= start) return;
        const std::string chrom =
            merge_aliases_ ? munge_chrom(written, [&](const std::string &name) { return index_.count(name) > 0; })
                           : written;
        auto [at, added] = index_.emplace(chrom, static_cast<int32_t>(sets_.size()));
        if (added) {
            chroms_.push_back(chrom);
            sets_.emplace_back();
        }
        auto &sets = sets_[at->second];
        sets[0].emplace_back(start, end);
        if (strand != "-") sets[1].emplace_back(start, end);
        if (strand != "+") sets[2].emplace_back(start, end);
    }

    bool merge_aliases_;
    std::vector<std::string> chroms_;
    std::unordered_map<std::string, int32_t> index_;
    std::vector<std::array<Intervals, 3>> sets_;  // per chromosome: all, +, -
};

}  // namespace

PYBIND11_MODULE(_regions, module) {
    module.doc() = "BED/GTF region loader (deeptoolsintervals-compatible)";
    py::class_<RegionSet>(module, "RegionSet")
        .def(py::init<std::string, std::string, std::string, bool, std::optional<std::string>>(),
             py::arg("transcript_id") = "transcript", py::arg("exon_id") = "exon",
             py::arg("designator") = "transcript_id", py::arg("keep_exons") = false,
             py::arg("default_group") = py::none())
        .def("read_file", &RegionSet::read_file, py::arg("path"))
        .def("read_stream", &RegionSet::read_stream, py::arg("read"), py::arg("path"))
        .def("finish", &RegionSet::finish)
        .def("munge", &RegionSet::munge, py::arg("chrom"))
        .def("rows", &RegionSet::rows, py::arg("chrom"), py::arg("end"))
        .def("group_names", &RegionSet::group_names)
        .def_property_readonly("labels", &RegionSet::labels)
        .def_property_readonly("chroms", &RegionSet::chroms)
        .def("__len__", &RegionSet::size);
    py::class_<Blacklist>(module, "Blacklist")
        .def(py::init<bool>(), py::arg("merge_aliases") = false)
        .def("read_file", &Blacklist::read_file, py::arg("path"))
        .def("read_stream", &Blacklist::read_stream, py::arg("read"), py::arg("path"))
        .def("finish", &Blacklist::finish)
        .def("intervals", &Blacklist::intervals, py::arg("chrom"), py::arg("strand") = '.')
        .def("overlaps", &Blacklist::overlaps, py::arg("chrom"), py::arg("segments"), py::arg("strand") = '.')
        .def_property_readonly("chroms", &Blacklist::chroms);
}
