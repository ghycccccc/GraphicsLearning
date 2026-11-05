#include "grass_sim.h"

#include "counters.h"
#include "gl_util.h"
#include "profiler.h"

#include <glm/gtc/packing.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <execution>
#include <numeric>

namespace {
constexpr int kChunkSize = 16384;     // blades per parallel task
constexpr float kMaxBend = 1.5f;      // must match grass_sim.comp
constexpr float kFlattenedThreshold = 0.6f;

std::vector<std::byte> encodeBends(const std::vector<glm::vec2>& bends, InstanceLayout layout) {
    std::vector<std::byte> bytes(bends.size() * bendStride(layout));
    for (size_t i = 0; i < bends.size(); ++i) storeBend(bytes.data(), i, bends[i], layout);
    return bytes;
}
} // namespace

bool GrassSim::Program::load(const char* defines) {
    id = loadProgram({{GL_COMPUTE_SHADER, "grass_sim.comp"}}, defines);
    if (!id) return false;
    count = glGetUniformLocation(id, "uCount");
    time = glGetUniformLocation(id, "uTime");
    dt = glGetUniformLocation(id, "uDt");
    windDir = glGetUniformLocation(id, "uWindDir");
    windStrength = glGetUniformLocation(id, "uWindStrength");
    stiffness = glGetUniformLocation(id, "uStiffness");
    damping = glGetUniformLocation(id, "uDamping");
    trampleCenter = glGetUniformLocation(id, "uTrampleCenter");
    trampleRadius = glGetUniformLocation(id, "uTrampleRadius");
    trampleStrength = glGetUniformLocation(id, "uTrampleStrength");
    return true;
}

bool GrassSim::init() {
    if (!floatProgram_.load("") || !packedProgram_.load("#define PACKED_LAYOUT\n")) return false;

    glCreateBuffers(1, &syncResult_);
    glNamedBufferStorage(syncResult_, sizeof(uint32_t), nullptr, GL_DYNAMIC_STORAGE_BIT);

    const GLbitfield flags = GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
    glCreateBuffers(1, &asyncRing_);
    glNamedBufferStorage(asyncRing_, kRingSlots * kRingSlotBytes, nullptr, flags);
    asyncMapped_ = static_cast<const std::byte*>(glMapNamedBufferRange(asyncRing_, 0, kRingSlots * kRingSlotBytes, flags));
    return asyncMapped_ != nullptr;
}

void GrassSim::destroy() {
    reset();
    for (GLsync& fence : asyncFences_) {
        if (fence) glDeleteSync(fence);
        fence = nullptr;
    }
    glDeleteProgram(floatProgram_.id);
    glDeleteProgram(packedProgram_.id);
    glUnmapNamedBuffer(asyncRing_);
    glDeleteBuffers(1, &asyncRing_);
    glDeleteBuffers(1, &syncResult_);
}

void GrassSim::reset() {
    owner_ = Owner::Rest;
    fieldVersion_ = ~0ull;
    bends_.clear();
    bends_.shrink_to_fit();
    velocities_.clear();
    velocities_.shrink_to_fit();
    releaseGpuState();
    flattened_ = 0;
    flattenedAge_ = -1;
    readbackMs_ = 0.0;
}

void GrassSim::releaseGpuState() {
    glDeleteBuffers(1, &bendBuffer_);
    glDeleteBuffers(1, &velocityBuffer_);
    bendBuffer_ = velocityBuffer_ = 0;
    g_residentBytes -= gpuStateBytes_;
    gpuStateBytes_ = 0;
}

void GrassSim::track(const GrassField& field) {
    // New blades, a new layout or a new mesh: start over from the rest pose.
    if (field.version() != fieldVersion_) {
        reset();
        fieldVersion_ = field.version();
    }
}

void GrassSim::moveToCpu(const GrassField& field) {
    const size_t n = size_t(field.count());
    if (owner_ == Owner::Cpu) return;
    bends_.resize(n);
    velocities_.resize(n);
    if (owner_ == Owner::Gpu) {
        // One-off transfer when switching backends; a blocking read is fine here.
        const InstanceLayout layout = field.layout();
        std::vector<std::byte> raw(n * bendStride(layout));
        glGetNamedBufferSubData(bendBuffer_, 0, GLsizeiptr(raw.size()), raw.data());
        for (size_t i = 0; i < n; ++i) {
            if (layout == InstanceLayout::Float32) {
                std::memcpy(&bends_[i], &raw[i * 8], 8);
            } else {
                uint32_t packed;
                std::memcpy(&packed, &raw[i * 4], 4);
                bends_[i] = glm::unpackHalf2x16(packed);
            }
        }
        glGetNamedBufferSubData(velocityBuffer_, 0, GLsizeiptr(n * sizeof(glm::vec2)), velocities_.data());
        releaseGpuState();
    } else {
        for (size_t i = 0; i < n; ++i) {
            const GrassInstance& blade = field.instances()[i];
            bends_[i] = {blade.b.z, blade.b.w};
            velocities_[i] = glm::vec2(0.0f);
        }
    }
    owner_ = Owner::Cpu;
}

void GrassSim::moveToGpu(const GrassField& field) {
    if (owner_ == Owner::Gpu) return;
    const size_t n = size_t(field.count());
    if (owner_ == Owner::Rest) {
        bends_.resize(n);
        velocities_.assign(n, glm::vec2(0.0f));
        for (size_t i = 0; i < n; ++i) bends_[i] = {field.instances()[i].b.z, field.instances()[i].b.w};
    }
    const std::vector<std::byte> bendBytes = encodeBends(bends_, field.layout());
    const uint64_t velocityBytes = n * sizeof(glm::vec2);

    releaseGpuState();
    glCreateBuffers(1, &bendBuffer_);
    glNamedBufferStorage(bendBuffer_, GLsizeiptr(std::max<size_t>(bendBytes.size(), 4)), bendBytes.data(), 0);
    glCreateBuffers(1, &velocityBuffer_);
    glNamedBufferStorage(velocityBuffer_, GLsizeiptr(std::max<uint64_t>(velocityBytes, 4)), velocities_.data(), 0);
    gpuStateBytes_ = bendBytes.size() + velocityBytes;
    g_residentBytes += gpuStateBytes_;
    g_counters.bytesUploaded += gpuStateBytes_;

    bends_.clear();
    bends_.shrink_to_fit();
    velocities_.clear();
    velocities_.shrink_to_fit();
    owner_ = Owner::Gpu;
}

void GrassSim::stepCpu(const GrassField& field, const SimParams& p, bool parallel, void* bendsOut,
                       CpuProfiler& cpu) {
    track(field);
    moveToCpu(field);
    CpuScope scope(cpu, "Simulate (CPU)");

    const int count = field.count();
    const InstanceLayout layout = field.layout();
    const GrassInstance* blades = field.instances().data();
    glm::vec2* bends = bends_.data();
    glm::vec2* velocities = velocities_.data();
    std::atomic<int> flattened{0};

    // Same math as grass_sim.comp, on exact (unquantized) blade data.
    auto stepRange = [&](int begin, int end) {
        int localFlattened = 0;
        for (int i = begin; i < end; ++i) {
            const GrassInstance& blade = blades[i];
            const glm::vec2 root(blade.a.x, blade.a.y);
            const glm::vec2 rest(blade.b.z, blade.b.w);
            glm::vec2 bend = bends[i];
            glm::vec2 velocity = velocities[i];

            const glm::vec2 target = rest + p.windDir * windSway(root, p.time, p.windStrength);
            glm::vec2 force = p.stiffness * (target - bend) - p.damping * velocity;
            const glm::vec2 away = root - p.trampleCenter;
            const float dist = glm::length(away);
            if (dist < p.trampleRadius && dist > 1e-4f) {
                force += away / dist * (p.trampleStrength * (1.0f - dist / p.trampleRadius));
            }
            velocity += force * p.dt;
            bend += velocity * p.dt;
            const float len = glm::length(bend);
            if (len > kMaxBend) {
                bend *= kMaxBend / len;
                velocity *= 0.5f;
            }

            bends[i] = bend;
            velocities[i] = velocity;
            storeBend(bendsOut, size_t(i), bend, layout); // bendsOut may be write-combined memory
            if (glm::length(bend - rest) > kFlattenedThreshold) ++localFlattened;
        }
        flattened += localFlattened;
    };

    if (parallel) {
        const int chunkCount = (count + kChunkSize - 1) / kChunkSize;
        if (int(chunks_.size()) != chunkCount) {
            chunks_.resize(size_t(chunkCount));
            std::iota(chunks_.begin(), chunks_.end(), 0);
        }
        std::for_each(std::execution::par, chunks_.begin(), chunks_.end(), [&](int chunk) {
            stepRange(chunk * kChunkSize, std::min(count, (chunk + 1) * kChunkSize));
        });
    } else {
        stepRange(0, count);
    }
    flattened_ = flattened.load();
    flattenedAge_ = 0;
    readbackMs_ = 0.0;
}

void GrassSim::stepGpu(const GrassField& field, const SimParams& p, Readback readback, CpuProfiler& cpu,
                       GpuProfiler& gpu) {
    track(field);
    moveToGpu(field);
    const GLuint count = GLuint(field.count());
    if (count == 0) return;
    const Program& prog = field.layout() == InstanceLayout::Float32 ? floatProgram_ : packedProgram_;
    const uint32_t zero = 0;
    const int slot = int(frame_ % kRingSlots);
    readbackMs_ = 0.0;

    if (readback == Readback::Async) {
        // Collect the newest result that is already finished (normally from 2 frames ago);
        // never wait for one that is still in flight.
        const int readySlot = int((frame_ + kRingSlots - 2) % kRingSlots);
        if (GLsync fence = asyncFences_[readySlot]) {
            const GLenum status = glClientWaitSync(fence, 0, 0);
            if (status == GL_ALREADY_SIGNALED || status == GL_CONDITION_SATISFIED) {
                uint32_t value;
                std::memcpy(&value, asyncMapped_ + readySlot * kRingSlotBytes, sizeof(value));
                flattened_ = int(value);
                flattenedAge_ = int(frame_ - asyncFrame_[readySlot]);
                glDeleteSync(fence);
                asyncFences_[readySlot] = nullptr;
            }
        }
        // This frame's slot was last used kRingSlots frames ago; it must be done before reuse.
        if (GLsync fence = asyncFences_[slot]) {
            glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, GLuint64(1'000'000'000));
            glDeleteSync(fence);
            asyncFences_[slot] = nullptr;
        }
        glClearNamedBufferSubData(asyncRing_, GL_R32UI, slot * kRingSlotBytes, sizeof(uint32_t), GL_RED_INTEGER,
                                  GL_UNSIGNED_INT, &zero);
        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 3, asyncRing_, slot * kRingSlotBytes, kRingSlotBytes);
    } else {
        glClearNamedBufferData(syncResult_, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, syncResult_);
        if (readback == Readback::None) flattenedAge_ = -1;
    }

    {
        GpuScope gpuScope(gpu, "Simulate");
        CpuScope cpuScope(cpu, "Simulate dispatch");
        glUseProgram(prog.id);
        glProgramUniform1ui(prog.id, prog.count, count);
        glProgramUniform1f(prog.id, prog.time, p.time);
        glProgramUniform1f(prog.id, prog.dt, p.dt);
        glProgramUniform2fv(prog.id, prog.windDir, 1, &p.windDir[0]);
        glProgramUniform1f(prog.id, prog.windStrength, p.windStrength);
        glProgramUniform1f(prog.id, prog.stiffness, p.stiffness);
        glProgramUniform1f(prog.id, prog.damping, p.damping);
        glProgramUniform2fv(prog.id, prog.trampleCenter, 1, &p.trampleCenter[0]);
        glProgramUniform1f(prog.id, prog.trampleRadius, p.trampleRadius);
        glProgramUniform1f(prog.id, prog.trampleStrength, p.trampleStrength);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, field.staticBuffer());
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bendBuffer_);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, velocityBuffer_);
        // 256 blades per group; 16M blades -> 62,500 groups, under the 65,535 limit.
        glDispatchCompute((count + 255) / 256, 1, 1);
        // Make the new bends visible to the vertex stage (SSBO reads and attribute fetch).
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
        g_counters.uniformUpdates += 10;
        g_counters.stateChanges += 4;
    }

    if (readback == Readback::Sync) {
        // Looks harmless, but the driver must finish all queued GPU work, including this
        // dispatch, before it can return 4 bytes. The GPU then idles until we submit again.
        CpuScope scope(cpu, "Readback (sync)");
        const auto start = std::chrono::steady_clock::now();
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        uint32_t value = 0;
        glGetNamedBufferSubData(syncResult_, 0, sizeof(value), &value);
        readbackMs_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        flattened_ = int(value);
        flattenedAge_ = 0;
    } else if (readback == Readback::Async) {
        glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
        asyncFences_[slot] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        asyncFrame_[slot] = frame_;
    }
    ++frame_;
}
