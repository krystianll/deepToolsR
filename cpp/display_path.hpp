#pragma once

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace dtp {

// Text that reaches Python (messages, labels, names) must be valid UTF-8.
// Length of the valid UTF-8 sequence at s[i] (of n bytes), 0 if invalid.
inline std::size_t utf8_sequence(const unsigned char *s, std::size_t i, std::size_t n) {
    const unsigned char c = s[i];
    const std::size_t len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3
                          : (c >> 3) == 30 ? 4 : 0;
    bool valid = len && i + len <= n && c != 0xC0 && c != 0xC1 && c <= 0xF4;
    for (std::size_t k = 1; valid && k < len; ++k) valid = (s[i + k] & 0xC0) == 0x80;
    if (valid && len == 3) {  // no overlong forms or surrogates
        const unsigned v = ((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6);
        valid = v >= 0x800 && (v < 0xD800 || v > 0xDFFF);
    }
    if (valid && len == 4) {
        const unsigned v = ((c & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12);
        valid = v >= 0x10000 && v <= 0x10FFFF;
    }
    return valid ? len : 0;
}

inline bool is_valid_utf8(const char *begin, const char *end) {
    const auto *s = reinterpret_cast<const unsigned char *>(begin);
    const std::size_t n = static_cast<std::size_t>(end - begin);
    for (std::size_t i = 0; i < n;) {
        if (s[i] < 0x80) { ++i; continue; }
        const std::size_t len = utf8_sequence(s, i, n);
        if (!len) return false;
        i += len;
    }
    return true;
}

// valid_utf8 replaces each byte that does not start a valid UTF-8 sequence
// with U+FFFD.
inline std::string valid_utf8(const std::string &text) {
    std::string out;
    out.reserve(text.size());
    const auto *s = reinterpret_cast<const unsigned char *>(text.data());
    for (std::size_t i = 0, n = text.size(); i < n;) {
        const std::size_t len = utf8_sequence(s, i, n);
        if (len) out.append(text, i, len), i += len;
        else out += "\xEF\xBF\xBD", ++i;
    }
    return out;
}

// Native code receives paths as the bytes the C file APIs open: UTF-8 on
// POSIX, the ANSI code page on Windows (deeptoolsr.native_path). Every path
// echoed in a message or label goes through display_path, which converts the
// ANSI bytes back to UTF-8 on Windows and is valid_utf8 elsewhere.
inline std::string display_path(const std::string &path) {
#ifdef _WIN32
    const int wide = MultiByteToWideChar(CP_ACP, 0, path.data(), int(path.size()), nullptr, 0);
    if (wide > 0) {
        std::wstring text(wide, L'\0');
        MultiByteToWideChar(CP_ACP, 0, path.data(), int(path.size()), &text[0], wide);
        const int bytes = WideCharToMultiByte(CP_UTF8, 0, text.data(), wide, nullptr, 0,
                                              nullptr, nullptr);
        std::string out(bytes, '\0');
        WideCharToMultiByte(CP_UTF8, 0, text.data(), wide, &out[0], bytes, nullptr, nullptr);
        return out;
    }
#endif
    return valid_utf8(path);
}

// "WHAT: PATH: reason" for a failed write; the reason is errno when set, so
// callers clear errno before the operation that may fail.
inline std::runtime_error write_error(const std::string &what, const std::string &path) {
    const int code = errno;
    std::string message = what + ": " + display_path(path);
    if (code) message += std::string(": ") + std::strerror(code);
    return std::runtime_error(message);
}

}  // namespace dtp
