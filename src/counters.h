#pragma once
#include <cstdint>

// Per-frame submission counters, incremented by whoever issues GL work.
struct FrameCounters {
    uint64_t drawCalls = 0;
    uint64_t instances = 0;
    uint64_t triangles = 0;
    uint64_t bytesUploaded = 0;
};

inline FrameCounters g_counters;

// Bytes currently held in GPU buffers allocated by the app.
inline uint64_t g_residentBytes = 0;
