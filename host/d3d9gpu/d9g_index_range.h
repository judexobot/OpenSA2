// Bounds of an index-buffer slice, reused until the buffer is locked again.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace d9g {

class IndexRangeCache {
public:
    // Invalidate before exposing writable bytes. Queries while a lock is held
    // must use cacheable=false: the caller can write through its pointer again.
    void invalidate() {
        if (entries_) for (auto& entry : *entries_) entry.count = 0;
    }

    bool get(const std::vector<uint8_t>& bytes, uint32_t first, uint32_t count, uint32_t width,
             uint32_t& low, uint32_t& high, bool cacheable = true) {
        if (!count || (width != 2 && width != 4) || (uint64_t(first) + count) * width > bytes.size())
            return false;
        Entry* entry = nullptr;
        // Tiny draws are cheaper to scan. Allocate no cache for vertex buffers
        // or small index buffers, and bound each large buffer's cache to 64 slots.
        if (cacheable && count >= 64) {
            if (!entries_) entries_ = std::make_unique<std::array<Entry, 64>>();
            uint32_t hash = first * 0x9e3779b1u ^ count * 0x85ebca6bu ^ width;
            hash ^= hash >> 16; // Aligned slices should not all share the low bits.
            entry = &(*entries_)[hash & 63];
            if (entry->count == count && entry->first == first && entry->width == width) {
                low = entry->low;
                high = entry->high;
                return true;
            }
        }
        low = UINT32_MAX;
        high = 0;
        const uint8_t* data = bytes.data() + uint64_t(first) * width;
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t index = 0;
            std::memcpy(&index, data + uint64_t(i) * width, width);
            low = std::min(low, index);
            high = std::max(high, index);
        }
        if (entry) *entry = {first, count, width, low, high};
        return true;
    }

private:
    struct Entry { uint32_t first = 0, count = 0, width = 0, low = 0, high = 0; };
    std::unique_ptr<std::array<Entry, 64>> entries_;
};

} // namespace d9g
