cmake_minimum_required(VERSION 3.15)
project(deeptoolsr_native LANGUAGES C CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

# --- vendored libdeflate (fast deflate for bigWig blocks, matrix gzip, BGZF) ---
set(LIBDEFLATE_BUILD_STATIC_LIB ON  CACHE BOOL "" FORCE)
set(LIBDEFLATE_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(LIBDEFLATE_BUILD_GZIP       OFF CACHE BOOL "" FORCE)
set(LIBDEFLATE_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
# Position-independent so it can link into the Python extension modules.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(cpp/libdeflate)
set(LIBDEFLATE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/libdeflate)

pybind11_add_module(_compute_matrix_io cpp/compute_matrix_io.cpp)
target_compile_features(_compute_matrix_io PRIVATE cxx_std_17)
target_include_directories(_compute_matrix_io PRIVATE cpp cpp/include ${LIBDEFLATE_INCLUDE_DIR})
target_link_libraries(_compute_matrix_io PRIVATE ZLIB::ZLIB libdeflate_static)

pybind11_add_module(_compute_matrix_stream cpp/compute_matrix_stream.cpp)
target_compile_features(_compute_matrix_stream PRIVATE cxx_std_17)
target_include_directories(_compute_matrix_stream PRIVATE cpp cpp/include ${LIBDEFLATE_INCLUDE_DIR})
target_link_libraries(_compute_matrix_stream PRIVATE ZLIB::ZLIB Threads::Threads libdeflate_static)

pybind11_add_module(_statistics cpp/statistics.cpp)
target_compile_features(_statistics PRIVATE cxx_std_17)
target_include_directories(_statistics PRIVATE cpp cpp/include)
target_link_libraries(_statistics PRIVATE Threads::Threads)

pybind11_add_module(_cluster cpp/cluster.cpp)
target_compile_features(_cluster PRIVATE cxx_std_17)
target_include_directories(_cluster PRIVATE cpp cpp/include)
target_link_libraries(_cluster PRIVATE Threads::Threads)
# No fused multiply-add: GCC contracts by default on aarch64, and separately
# inlined copies of one kernel (serial vs pooled) could round differently,
# breaking identical results for every -p. x86-64 baseline builds have no FMA.
target_compile_options(_cluster PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>)

pybind11_add_module(_raster cpp/raster.cpp)
target_compile_features(_raster PRIVATE cxx_std_17)
target_include_directories(_raster PRIVATE cpp cpp/include)
target_link_libraries(_raster PRIVATE Threads::Threads)

pybind11_add_module(_transform cpp/transform.cpp)
target_compile_features(_transform PRIVATE cxx_std_17)
target_include_directories(_transform PRIVATE cpp)
target_link_libraries(_transform PRIVATE Threads::Threads)

set(LIBBIGWIG_SOURCES
    cpp/libBigWig/bwRead.c
    cpp/libBigWig/bwStats.c
    cpp/libBigWig/bwValues.c
    cpp/libBigWig/bwWrite.c
    cpp/libBigWig/io.c)
pybind11_add_module(_compute_matrix_native
    cpp/compute_matrix_native.cpp
    ${LIBBIGWIG_SOURCES})
target_compile_features(_compute_matrix_native PRIVATE cxx_std_17)
target_include_directories(_compute_matrix_native PRIVATE cpp ${LIBDEFLATE_INCLUDE_DIR})
target_compile_definitions(_compute_matrix_native PRIVATE NOCURL)
target_link_libraries(_compute_matrix_native PRIVATE ZLIB::ZLIB Threads::Threads libdeflate_static)

# --- bamCoverageR native backend (_coverage) -------------------------------
# Vendored htslib is built BAM-minimal (no bz2/lzma/curl/plugins) against zlib +
# the vendored libdeflate, producing a static libhts that links into _coverage.
# Linux/macOS drive htslib's own configure+make (keeps the vendored tree drop-in
# upgradeable); Windows compiles the same sources with CMake directly.
set(HTSLIB_VENDORED ${CMAKE_CURRENT_SOURCE_DIR}/cpp/htslib)
if(WIN32)
    # Windows (MSYS2 MinGW): htslib's autotools configure/make is impractical, so
    # compile its LIBHTS_OBJS (Makefile) + bundled htscodecs here, with a
    # config.h equivalent to what configure produces for this feature set.
    # CRAM stays in: hts.c/sam.c reference it unconditionally.
    # SIMD extras (SSSE3/AVX/NEON rANS kernels) are left disabled; their files
    # compile to nothing without the HAVE_* macros.
    file(STRINGS ${HTSLIB_VENDORED}/version.sh _hts_version_line REGEX "^VERSION=")
    string(REGEX REPLACE "^VERSION=" "" HTSLIB_VERSION "${_hts_version_line}")
    set(HTSLIB_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/htslib-gen)
    file(WRITE ${HTSLIB_GEN_DIR}/config.h
"#define HAVE_ATTRIBUTE_CONSTRUCTOR 1
#define HAVE_LIBZ 1
#define HAVE_LIBDEFLATE 1
#define HAVE___BUILTIN_CLZ 1
#define HAVE___BUILTIN_PREFETCH 1
#define _FILE_OFFSET_BITS 64
#define _XOPEN_SOURCE 700
")
    file(WRITE ${HTSLIB_GEN_DIR}/version.h
        "#define HTS_VERSION_TEXT \"${HTSLIB_VERSION}\"\n")
    file(WRITE ${HTSLIB_GEN_DIR}/config_vars.h
"#define HTS_CC \"${CMAKE_C_COMPILER_ID}\"
#define HTS_CPPFLAGS \"\"
#define HTS_CFLAGS \"-O2\"
#define HTS_LDFLAGS \"\"
#define HTS_LIBS \"-lz -ldeflate -lregex -lws2_32\"
")
    set(_hts_c
        kfunc kstring bcf_sr_sort bgzf errmod faidx header hfile hts hts_expr
        hts_os md5 multipart probaln realn regidx region sam sam_mods simd
        synced_bcf_reader vcf_sweep tbx textutils thread_pool vcf vcfutils
        cram/cram_codecs cram/cram_decode cram/cram_encode cram/cram_external
        cram/cram_index cram/cram_io cram/cram_stats cram/mFILE
        cram/open_trace_file cram/pooled_alloc cram/string_alloc
        htscodecs/htscodecs/arith_dynamic htscodecs/htscodecs/fqzcomp_qual
        htscodecs/htscodecs/htscodecs htscodecs/htscodecs/pack
        htscodecs/htscodecs/rANS_static4x16pr
        htscodecs/htscodecs/rANS_static32x16pr_avx2
        htscodecs/htscodecs/rANS_static32x16pr_avx512
        htscodecs/htscodecs/rANS_static32x16pr_sse4
        htscodecs/htscodecs/rANS_static32x16pr_neon
        htscodecs/htscodecs/rANS_static32x16pr htscodecs/htscodecs/rANS_static
        htscodecs/htscodecs/rle htscodecs/htscodecs/tokenise_name3
        htscodecs/htscodecs/utils)
    list(TRANSFORM _hts_c PREPEND ${HTSLIB_VENDORED}/)
    list(TRANSFORM _hts_c APPEND .c)
    add_library(htslib_static STATIC ${_hts_c})
    # Generated headers first so they win over any stale configure output.
    target_include_directories(htslib_static PRIVATE
        ${HTSLIB_GEN_DIR} ${HTSLIB_VENDORED}
        ${LIBDEFLATE_INCLUDE_DIR})
    target_compile_options(htslib_static PRIVATE -O2)
    # POSIX regex (sam.c, hts_expr.c) from MSYS2 libsystre/TRE; Winsock from
    # hfile.c's network backend.
    target_link_libraries(htslib_static PUBLIC
        libdeflate_static ZLIB::ZLIB Threads::Threads regex tre intl iconv ws2_32)
    set(HTSLIB_INCLUDE_DIR ${HTSLIB_VENDORED})
    set(HTSLIB_STATIC htslib_static)
    add_custom_target(htslib_ep)
    add_dependencies(htslib_ep htslib_static)
else()
    include(ExternalProject)
    ExternalProject_Add(htslib_ep
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/htslib-ep
        # Copy the vendored source into the build tree so configure/make never dirty
        # the checked-in sources (htslib builds in-source).
        DOWNLOAD_COMMAND ${CMAKE_COMMAND} -E copy_directory ${HTSLIB_VENDORED} <SOURCE_DIR>
        BUILD_IN_SOURCE 1
        CONFIGURE_COMMAND <SOURCE_DIR>/configure
            --disable-bz2 --disable-lzma --disable-libcurl --disable-plugins
            CC=${CMAKE_C_COMPILER}
            "CFLAGS=-fPIC -O2 -I${LIBDEFLATE_INCLUDE_DIR}"
            "LDFLAGS=-L${CMAKE_CURRENT_BINARY_DIR}/cpp/libdeflate"
        BUILD_COMMAND make libhts.a
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS <SOURCE_DIR>/libhts.a
        LOG_DOWNLOAD 1
        LOG_CONFIGURE 1
        LOG_BUILD 1)
    # libdeflate (built by add_subdirectory) must exist before htslib configures.
    add_dependencies(htslib_ep libdeflate_static)
    ExternalProject_Get_Property(htslib_ep SOURCE_DIR)
    set(HTSLIB_INCLUDE_DIR ${SOURCE_DIR})
    set(HTSLIB_STATIC ${SOURCE_DIR}/libhts.a)
endif()

# bigWigOperationsR backend: bigWig-only (libBigWig + libdeflate + zlib), no htslib.
pybind11_add_module(_bigwig cpp/bigwig_ops.cpp ${LIBBIGWIG_SOURCES})
target_compile_features(_bigwig PRIVATE cxx_std_17)
target_include_directories(_bigwig PRIVATE cpp ${LIBDEFLATE_INCLUDE_DIR})
target_compile_definitions(_bigwig PRIVATE NOCURL)
target_link_libraries(_bigwig PRIVATE libdeflate_static ZLIB::ZLIB Threads::Threads)

pybind11_add_module(_coverage cpp/coverage.cpp ${LIBBIGWIG_SOURCES})
add_dependencies(_coverage htslib_ep)
target_compile_features(_coverage PRIVATE cxx_std_17)
target_include_directories(_coverage PRIVATE cpp ${HTSLIB_INCLUDE_DIR} ${LIBDEFLATE_INCLUDE_DIR})
target_compile_definitions(_coverage PRIVATE NOCURL)
target_link_libraries(_coverage PRIVATE
    ${HTSLIB_STATIC} libdeflate_static ZLIB::ZLIB Threads::Threads m)

if(MSVC)
    target_compile_options(_compute_matrix_io PRIVATE /O2)
else()
    target_compile_options(_compute_matrix_io PRIVATE -O3)
    target_compile_options(_compute_matrix_stream PRIVATE -O3)
    target_compile_options(_transform PRIVATE -O3)
    target_compile_options(_compute_matrix_native PRIVATE -O3)
    target_compile_options(_coverage PRIVATE -O3)
    target_compile_options(_bigwig PRIVATE -O3)
endif()

install(TARGETS _compute_matrix_io LIBRARY DESTINATION deeptoolsr)
install(TARGETS _compute_matrix_stream LIBRARY DESTINATION deeptoolsr)
install(TARGETS _transform LIBRARY DESTINATION deeptoolsr)
install(TARGETS _statistics LIBRARY DESTINATION deeptoolsr)
install(TARGETS _cluster LIBRARY DESTINATION deeptoolsr)
install(TARGETS _raster LIBRARY DESTINATION deeptoolsr)
install(TARGETS _compute_matrix_native LIBRARY DESTINATION deeptoolsr)
install(TARGETS _coverage LIBRARY DESTINATION deeptoolsr)
install(TARGETS _bigwig LIBRARY DESTINATION deeptoolsr)
