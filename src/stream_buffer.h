#pragma once
#include <GL/glew.h>

#include <cstddef>
#include <cstdint>
#include <vector>

// How CPU-written per-frame data reaches the GPU.
enum class UploadMethod {
    BufferSubData,  // one buffer, glBufferSubData each frame; the driver must stall or copy
    Orphaning,      // glBufferData(nullptr) first, so the driver hands out fresh storage
    PersistentRing, // persistently mapped, 3 regions guarded by fences; the app owns the sync
    kCount
};

// A GPU buffer the CPU rewrites every frame.
//   void* dst = stream.beginWrite();   // may wait on a fence (PersistentRing)
//   ...write bytes into dst...
//   GLintptr offset = stream.endWrite(bytes);
//   ...bind buffer() at offset, issue draws...
//   stream.endFrame();                 // fences this frame's region
class StreamBuffer {
public:
    static constexpr int kRingRegions = 3;

    // (Re)allocates for `bytes` per frame. Returns false if the GPU allocation failed.
    bool reserve(uint64_t bytes, UploadMethod method);
    void release();

    void* beginWrite();
    GLintptr endWrite(uint64_t bytes);
    void endFrame();

    GLuint buffer() const { return buffer_; }
    double lastWaitMs() const { return waitMs_; }

private:
    UploadMethod method_ = UploadMethod::BufferSubData;
    GLuint buffer_ = 0;
    uint64_t regionBytes_ = 0;
    uint64_t allocatedBytes_ = 0;
    std::vector<std::byte> staging_;
    std::byte* mapped_ = nullptr;
    GLsync fences_[kRingRegions] = {};
    int region_ = 0;
    double waitMs_ = 0.0;
};
