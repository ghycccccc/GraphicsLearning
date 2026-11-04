#include "stream_buffer.h"

#include "counters.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

bool StreamBuffer::reserve(uint64_t bytes, UploadMethod method) {
    // Regions must start at an offset that is legal for SSBO binding.
    GLint ssboAlignment = 256;
    glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &ssboAlignment);
    const uint64_t alignment = uint64_t(std::max(ssboAlignment, 256));
    const uint64_t regionBytes = (std::max<uint64_t>(bytes, 1) + alignment - 1) / alignment * alignment;
    if (buffer_ && method == method_ && regionBytes == regionBytes_) return true;
    release();

    method_ = method;
    regionBytes_ = regionBytes;
    while (glGetError() != GL_NO_ERROR) {
    }
    glCreateBuffers(1, &buffer_);
    switch (method) {
    case UploadMethod::BufferSubData:
        allocatedBytes_ = regionBytes;
        glNamedBufferStorage(buffer_, GLsizeiptr(allocatedBytes_), nullptr, GL_DYNAMIC_STORAGE_BIT);
        break;
    case UploadMethod::Orphaning:
        allocatedBytes_ = regionBytes;
        glNamedBufferData(buffer_, GLsizeiptr(allocatedBytes_), nullptr, GL_STREAM_DRAW);
        break;
    case UploadMethod::PersistentRing: {
        allocatedBytes_ = regionBytes * kRingRegions;
        const GLbitfield flags = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
        glNamedBufferStorage(buffer_, GLsizeiptr(allocatedBytes_), nullptr, flags);
        if (glGetError() == GL_NO_ERROR) {
            mapped_ = static_cast<std::byte*>(
                glMapNamedBufferRange(buffer_, 0, GLsizeiptr(allocatedBytes_), flags));
        }
        break;
    }
    default:
        break;
    }

    if (glGetError() == GL_OUT_OF_MEMORY ||
        (method == UploadMethod::PersistentRing && mapped_ == nullptr)) {
        std::fprintf(stderr, "Stream buffer allocation of %.0f MB failed\n",
                     double(allocatedBytes_) / (1024.0 * 1024.0));
        glDeleteBuffers(1, &buffer_);
        buffer_ = 0;
        allocatedBytes_ = 0;
        regionBytes_ = 0;
        mapped_ = nullptr;
        return false;
    }
    if (method != UploadMethod::PersistentRing) staging_.resize(size_t(regionBytes));
    g_residentBytes += allocatedBytes_;
    return true;
}

void StreamBuffer::release() {
    for (GLsync& fence : fences_) {
        if (fence) glDeleteSync(fence);
        fence = nullptr;
    }
    if (mapped_) glUnmapNamedBuffer(buffer_);
    glDeleteBuffers(1, &buffer_);
    buffer_ = 0;
    mapped_ = nullptr;
    g_residentBytes -= allocatedBytes_;
    allocatedBytes_ = 0;
    regionBytes_ = 0;
    staging_.clear();
    staging_.shrink_to_fit();
}

void* StreamBuffer::beginWrite() {
    waitMs_ = 0.0;
    if (method_ != UploadMethod::PersistentRing) return staging_.data();

    // The region we are about to overwrite was last used kRingRegions frames ago; wait
    // until the GPU has finished reading it.
    region_ = (region_ + 1) % kRingRegions;
    if (GLsync fence = fences_[region_]) {
        const auto start = std::chrono::steady_clock::now();
        while (glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 1'000'000'000) == GL_TIMEOUT_EXPIRED) {
        }
        waitMs_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        glDeleteSync(fence);
        fences_[region_] = nullptr;
    }
    return mapped_ + region_ * regionBytes_;
}

GLintptr StreamBuffer::endWrite(uint64_t bytes) {
    g_counters.bytesUploaded += bytes;
    switch (method_) {
    case UploadMethod::BufferSubData:
        glNamedBufferSubData(buffer_, 0, GLsizeiptr(bytes), staging_.data());
        return 0;
    case UploadMethod::Orphaning:
        glNamedBufferData(buffer_, GLsizeiptr(allocatedBytes_), nullptr, GL_STREAM_DRAW);
        glNamedBufferSubData(buffer_, 0, GLsizeiptr(bytes), staging_.data());
        return 0;
    default: // coherent mapping: writes are already visible, nothing to submit
        return GLintptr(region_ * regionBytes_);
    }
}

void StreamBuffer::endFrame() {
    if (method_ != UploadMethod::PersistentRing) return;
    fences_[region_] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}
