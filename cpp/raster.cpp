#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize2.h"

#include "parallel.hpp"
#include "matrix_view.hpp"

namespace py = pybind11;

using dtp::parallel_rows;

namespace {

stbir_filter parse_filter(const std::string &name) {
    if (name == "triangle" || name == "bilinear") return STBIR_FILTER_TRIANGLE;
    if (name == "mitchell") return STBIR_FILTER_MITCHELL;
    if (name == "catmullrom") return STBIR_FILTER_CATMULLROM;
    if (name == "cubic" || name == "cubicbspline") return STBIR_FILTER_CUBICBSPLINE;
    if (name == "box") return STBIR_FILTER_BOX;
    if (name == "point" || name == "nearest") return STBIR_FILTER_POINT_SAMPLE;
    if (name == "default") return STBIR_FILTER_DEFAULT;
    throw std::runtime_error("unknown resize filter: " + name);
}

// Build a full-resolution RGBA raster of scalar sample type T (uint8 or uint16)
// then downsample it to (out_h, out_w) with `filter`, writing an 8-bit RGBA
// result. A uint16 intermediate averages neighbours at higher precision (closer
// to matplotlib's float resample) at twice the transient raster size.
template <typename T>
void render_impl(const dtp::MatrixView &view,
                 const std::uint8_t *lut_data, const std::array<std::uint8_t, 4> &bad,
                 std::size_t src_h, std::size_t src_w,
                 double vmin, double inv_span,
                 int out_h, int out_w, stbir_filter filter,
                 int num_threads, std::uint8_t *out_data) {
    constexpr int scale = sizeof(T) == 2 ? 257 : 1;  // 255 -> 65535 exactly
    auto promote = [](std::uint8_t v) -> T { return static_cast<T>(v * scale); };

    std::vector<T> raw(src_h * src_w * 4);
    {
        py::gil_scoped_release release;
        parallel_rows(src_h, num_threads, [&](std::size_t rr) {
            T *out_row = raw.data() + rr * src_w * 4;
            for (std::size_t cc = 0; cc < src_w; ++cc) {
                const double fv = view.at(rr, cc);
                T *px = out_row + cc * 4;
                if (!std::isfinite(fv)) {
                    for (int k = 0; k < 4; ++k) px[k] = promote(bad[k]);
                    continue;
                }
                double t = (static_cast<double>(fv) - vmin) * inv_span;
                t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
                const int idx = static_cast<int>(std::lround(t * 255.0));
                const std::uint8_t *entry = lut_data + static_cast<std::size_t>(idx) * 4;
                for (int k = 0; k < 4; ++k) px[k] = promote(entry[k]);
            }
        });
    }

    const bool resize = !(static_cast<std::size_t>(out_h) == src_h &&
                          static_cast<std::size_t>(out_w) == src_w);
    std::vector<T> resized;
    const T *final_ptr = raw.data();
    if (resize) {
        resized.resize(static_cast<std::size_t>(out_h) * out_w * 4);
        py::gil_scoped_release release;
        const stbir_datatype dt = sizeof(T) == 2 ? STBIR_TYPE_UINT16 : STBIR_TYPE_UINT8;
        const void *ok = stbir_resize(
            raw.data(), static_cast<int>(src_w), static_cast<int>(src_h), 0,
            resized.data(), out_w, out_h, 0,
            STBIR_RGBA, dt, STBIR_EDGE_CLAMP, filter);
        if (ok == nullptr) throw std::runtime_error("stbir_resize failed");
        final_ptr = resized.data();
    }

    const std::size_t n = static_cast<std::size_t>(out_h) * out_w * 4;
    if (sizeof(T) == 2) {
        for (std::size_t i = 0; i < n; ++i)
            out_data[i] = static_cast<std::uint8_t>(
                (static_cast<unsigned>(final_ptr[i]) + 128u) / 257u);
    } else {
        for (std::size_t i = 0; i < n; ++i)
            out_data[i] = static_cast<std::uint8_t>(final_ptr[i]);
    }
}

// Render a (possibly strided) projected matrix to an antialiased uint8 RGBA
// image at (out_h, out_w). The colour of a finite value v is looked up in a
// 256-entry RGBA LUT at round(clamp((v-vmin)/(vmax-vmin), 0, 1) * 255); missing
// non-finite values take `bad`. Colour-mapping and downsampling
// happen in C++ so matplotlib never builds a full-resolution float RGBA buffer.
// `filter` selects the stb resize kernel; `bit_depth` (8 or 16) sets the
// precision of the intermediate raster used for blending.
py::array render_heatmap(const py::buffer &values,
                         const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> &lut,
                         std::array<std::uint8_t, 4> bad,
                         double vmin, double vmax,
                         int out_h, int out_w,
                         const std::string &filter, int bit_depth,
                         int num_threads,
                         const py::object &rows_object, const py::object &row_range,
                         const py::object &cols_object, const py::object &col_range) {
    const dtp::MatrixView view(values, rows_object, row_range, cols_object, col_range);

    const auto lut_info = lut.request();
    if (lut_info.ndim != 2 || lut_info.shape[0] != 256 || lut_info.shape[1] != 4)
        throw std::runtime_error("lut must be a (256, 4) uint8 array");
    const std::uint8_t *lut_data = static_cast<const std::uint8_t *>(lut_info.ptr);

    const std::size_t src_h = view.rows;
    const std::size_t src_w = view.cols;
    if (src_h == 0 || src_w == 0) throw std::runtime_error("empty sub-matrix");
    if (out_h <= 0) out_h = static_cast<int>(src_h);
    if (out_w <= 0) out_w = static_cast<int>(src_w);

    const double span = vmax - vmin;
    const double inv_span = span != 0.0 ? 1.0 / span : 0.0;
    const stbir_filter stb_filter = parse_filter(filter);
    if (bit_depth != 8 && bit_depth != 16)
        throw std::runtime_error("bit_depth must be 8 or 16");

    py::array_t<std::uint8_t> result({static_cast<py::ssize_t>(out_h),
                                      static_cast<py::ssize_t>(out_w),
                                      static_cast<py::ssize_t>(4)});
    std::uint8_t *out_data = result.mutable_data();

    if (bit_depth == 16) {
        render_impl<std::uint16_t>(view, lut_data,
                                   bad, src_h, src_w, vmin, inv_span,
                                   out_h, out_w, stb_filter, num_threads, out_data);
    } else {
        render_impl<std::uint8_t>(view, lut_data,
                                  bad, src_h, src_w, vmin, inv_span,
                                  out_h, out_w, stb_filter, num_threads, out_data);
    }
    return result;
}

}  // namespace

PYBIND11_MODULE(_raster, module) {
    module.doc() = "float32 -> antialiased uint8 RGBA heatmap rasteriser";
    module.attr("supports_multicore") = true;
    module.def("render_heatmap", &render_heatmap, py::arg("matrix"),
               py::arg("lut"), py::arg("bad"),
               py::arg("vmin"), py::arg("vmax"),
               py::arg("out_h") = 0, py::arg("out_w") = 0,
               py::arg("filter") = "triangle", py::arg("bit_depth") = 8,
               py::arg("num_threads") = 1, DTP_PROJECTION_ARGS);
}
