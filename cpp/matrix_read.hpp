#pragma once

#include "matrix_text.hpp"
#include <pybind11/pybind11.h>

namespace dtp {

struct MatrixSchema {
    std::size_t rows = 0;
    std::size_t bins = 0;
};

// Header metadata uses the same Python validator as CLI preflight. The body
// stays native, and its count is checked on the existing read pass, before the
// caller can commit any output. The GIL is held only for the one JSON header.
template <class HeaderFn, class LinesFn>
void read_matrix_chunks(const std::string &path, std::size_t buffer_size,
                        HeaderFn on_header, LinesFn on_lines) {
    MatrixSchema schema;
    std::size_t rows_read = 0;
    read_gzip_matrix(path, buffer_size,
        [&](const char *begin, const char *end) {
            {
                namespace py = pybind11;
                py::gil_scoped_acquire acquire;
                const auto parameters = py::module_::import("json").attr("loads")(
                    py::str(begin, static_cast<std::size_t>(end - begin)));
                const auto shape = py::module_::import("deeptoolsr.matrix_validation")
                    .attr("validate_matrix_header")(parameters).cast<py::tuple>();
                schema.rows = shape[0].cast<std::size_t>();
                schema.bins = shape[1].cast<std::size_t>();
            }
            on_header(begin, end, schema);
        },
        [&](const char *base, const auto &lines) {
            if (lines.size() > schema.rows - rows_read)
                throw std::runtime_error("matrix contains more regions than its header");
            on_lines(base, lines, schema);
            rows_read += lines.size();
        });
    if (rows_read != schema.rows)
        throw std::runtime_error("expected " + std::to_string(schema.rows) +
            " regions but read " + std::to_string(rows_read));
}

}  // namespace dtp
