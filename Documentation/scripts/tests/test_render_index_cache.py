"""Index bounds, cache invalidation through the real Lock, and repeated-draw CPU cost."""
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    out = tmp_path_factory.mktemp("index-range")
    resources = (ROOT / "host/d3d9gpu/d9g_resources.cpp").read_text()
    start = resources.index("template <class Stub, class Com, int I, class Desc>\nHRESULT STDMETHODCALLTYPE BufferBase<Stub, Com, I, Desc>::Lock(")
    end = resources.index("\n}\n", start) + 3
    header = (ROOT / "host/d3d9gpu/d9g.h").read_text()
    method = header.index("    bool index_range(")
    method_end = header.index("\n    }", method) + 6
    source = HARNESS.replace("// LOCK", resources[start:end]).replace("// RANGE", header[method:method_end])
    (out / "test.cpp").write_text(source)
    exe = out / "test"
    done = subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                           f"-I{ROOT / 'host/d3d9gpu'}", str(out / "test.cpp"), "-o", str(exe)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return exe


def test_index_bounds_and_lock_invalidation(binary):
    done = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr


def test_repeated_draw_benchmark_keeps_the_same_bounds(binary):
    done = subprocess.run([str(binary), "--benchmark"], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert result["baseline_checksum"] == result["cached_checksum"]
    out = ROOT / "private/generated/ci"
    out.mkdir(parents=True, exist_ok=True)
    (out / "render-benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(done.stdout)


HARNESS = r'''
#include "d9g_index_range.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
using UINT = uint32_t;
using DWORD = uint32_t;
using HRESULT = int;
#define STDMETHODCALLTYPE
constexpr int D3D_OK = 0, D3DERR_INVALIDCALL = -1;
constexpr uint32_t D3DLOCK_READONLY = 16, D3DPOOL_MANAGED = 1;
int answer(int, int, int result) { return result; }
void not_implemented_value(const char*) {}
template<class Stub, class Com, int I, class Desc> struct BufferBase {
    uint32_t len_ = 4096, pool_ = D3DPOOL_MANAGED;
    uint32_t dirty_lo_ = 0, dirty_hi_ = 0, lock_count_ = 0;
    std::vector<uint8_t> store_ = std::vector<uint8_t>(len_, 0);
    d9g::IndexRangeCache index_ranges_;
    bool locked() const { return lock_count_ != 0; }
    HRESULT Lock(UINT offset, UINT size, void** data, DWORD flags);
    // RANGE
};
// LOCK

__attribute__((noinline)) uint64_t scan(const std::vector<uint8_t>& bytes, uint32_t first, uint32_t count, uint32_t width) {
    uint32_t low = UINT32_MAX, high = 0;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t value = 0;
        std::memcpy(&value, bytes.data() + uint64_t(first + i) * width, width);
        low = std::min(low, value); high = std::max(high, value);
    }
    return uint64_t(low) << 32 | high;
}
__attribute__((noinline)) uint64_t cached(d9g::IndexRangeCache& cache, const std::vector<uint8_t>& bytes,
                                        uint32_t first, uint32_t count, uint32_t width) {
    uint32_t low = 0, high = 0;
    assert(cache.get(bytes, first, count, width, low, high));
    return uint64_t(low) << 32 | high;
}
int main(int argc, char**) {
    std::mt19937 rng(45123);
    std::vector<uint8_t> bytes(131072);
    for (auto& b : bytes) b = uint8_t(rng());
    d9g::IndexRangeCache cache;
    if (argc > 1) {
        uint64_t baseline_sum = 0, cached_sum = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 10000; ++i) {
            asm volatile("" : : "r"(bytes.data()) : "memory");
            baseline_sum += scan(bytes, 0, 16384, 2);
        }
        const auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < 10000; ++i) {
            asm volatile("" : : "r"(bytes.data()) : "memory");
            cached_sum += cached(cache, bytes, 0, 16384, 2);
        }
        const auto t2 = std::chrono::steady_clock::now();
        assert(baseline_sum == cached_sum);
        std::cout << "{\"scope\":\"repeated index bounds only; not gameplay FPS\",\"draws\":10000,"
                  << "\"baseline_ms\":" << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << ",\"cached_ms\":" << std::chrono::duration<double, std::milli>(t2 - t1).count()
                  << ",\"baseline_checksum\":" << baseline_sum << ",\"cached_checksum\":" << cached_sum << "}\n";
        return 0;
    }
    // Mixed 16/32-bit slices, small draws, direct-map collisions, and repeated hits.
    for (int i = 0; i < 12000; ++i) {
        const uint32_t width = (rng() & 1) ? 2 : 4;
        const uint32_t first = rng() % 1024, count = 1 + rng() % 4096;
        const auto expected = scan(bytes, first, count, width);
        assert(cached(cache, bytes, first, count, width) == expected);
        assert(cached(cache, bytes, first, count, width) == expected);
        if (i % 113 == 0) { bytes[rng() % bytes.size()] ^= 255; cache.invalidate(); }
    }
    uint32_t low = 7, high = 9;
    assert(!cache.get(bytes, UINT32_MAX, UINT32_MAX, 4, low, high));
    assert(!cache.get(bytes, 0, 0, 2, low, high));
    assert(!cache.get(bytes, 0, 128, 3, low, high));
    assert(!cache.get(bytes, uint32_t(bytes.size() / 2), 1, 2, low, high));
    assert(low == 7 && high == 9);

    // The production BufferBase::Lock invalidates a populated cache before the
    // caller writes. Reads during nested locks must always see the latest bytes.
    BufferBase<int, int, 5, int> buffer;
    assert(buffer.index_range(0, 128, 2, low, high) && high == 0);
    void* data = nullptr;
    assert(buffer.Lock(0, 0, &data, 0) == D3D_OK);
    const uint16_t first = 1234, second = 54321;
    std::memcpy(data, &first, 2);
    assert(buffer.index_range(0, 128, 2, low, high) && high == first);
    assert(buffer.Lock(0, 0, &data, 0) == D3D_OK);
    std::memcpy(data, &second, 2);
    assert(buffer.index_range(0, 128, 2, low, high) && high == second);
    --buffer.lock_count_;
    std::memcpy(data, &first, 2);
    assert(buffer.index_range(0, 128, 2, low, high) && high == first);
    --buffer.lock_count_;
    assert(buffer.index_range(0, 128, 2, low, high) && high == first);
    // A managed READONLY lock also invalidates, conservatively retaining the
    // old bounds check's behavior even if a caller writes through that pointer.
    assert(buffer.Lock(0, 0, &data, D3DLOCK_READONLY) == D3D_OK);
    std::memcpy(data, &second, 2); --buffer.lock_count_;
    assert(buffer.index_range(0, 128, 2, low, high) && high == second);
}
'''
