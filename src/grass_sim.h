#pragma once
#include "grass.h"

#include <GL/glew.h>
#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

class CpuProfiler;
class GpuProfiler;

// How the CPU learns the flattened-blade count when the simulation runs on the GPU.
enum class Readback {
    None,  // the CPU never looks at GPU results
    Sync,  // glGetBufferSubData right after the dispatch: the CPU waits for the GPU to drain
    Async, // results land in a persistently mapped ring, read 2 frames later behind a fence
    kCount
};
inline constexpr const char* kReadbackNames[] = {"None", "Sync glGetBufferSubData", "Async (fence, 2 frames late)"};
inline constexpr const char* kReadbackKeys[] = {"none", "sync", "async"};

struct SimParams {
    float time = 0.0f;
    float dt = 1.0f / 60.0f; // fixed step so CPU and GPU runs are comparable
    float stiffness = 40.0f;
    float damping = 4.0f;
    glm::vec2 windDir{1.0f, 0.0f};
    float windStrength = 0.35f;
    glm::vec2 trampleCenter{0.0f};
    float trampleRadius = 0.0f;
    float trampleStrength = 300.0f;
};

// Per-blade spring-damper state (bend + velocity). The state lives where the current backend
// needs it and is moved when the backend changes, so switching CPU <-> GPU keeps the motion.
class GrassSim {
public:
    bool init();
    void destroy();

    // Drops all state; the next step starts from the rest pose.
    void reset();

    // Steps on the CPU and writes the new bends into `bendsOut` using the field's layout.
    void stepCpu(const GrassField& field, const SimParams& params, bool parallel, void* bendsOut,
                 CpuProfiler& cpu);
    // Steps in a compute shader; the bends stay in bendBuffer().
    void stepGpu(const GrassField& field, const SimParams& params, Readback readback, CpuProfiler& cpu,
                 GpuProfiler& gpu);

    GLuint bendBuffer() const { return bendBuffer_; }
    int flattenedBlades() const { return flattened_; }
    int flattenedAgeFrames() const { return flattenedAge_; } // -1 when unknown
    double readbackStallMs() const { return readbackMs_; }

private:
    enum class Owner { Rest, Cpu, Gpu };
    struct Program {
        GLuint id = 0;
        GLint count = -1, time = -1, dt = -1, windDir = -1, windStrength = -1, stiffness = -1, damping = -1;
        GLint trampleCenter = -1, trampleRadius = -1, trampleStrength = -1;
        bool load(const char* defines);
    };
    static constexpr int kRingSlots = 3;
    static constexpr GLsizeiptr kRingSlotBytes = 256;

    void track(const GrassField& field);
    void moveToCpu(const GrassField& field);
    void moveToGpu(const GrassField& field);
    void releaseGpuState();

    Program floatProgram_, packedProgram_;
    Owner owner_ = Owner::Rest;
    uint64_t fieldVersion_ = ~0ull;

    std::vector<glm::vec2> bends_, velocities_;
    std::vector<int> chunks_;

    GLuint bendBuffer_ = 0, velocityBuffer_ = 0;
    uint64_t gpuStateBytes_ = 0;

    GLuint syncResult_ = 0;  // plain buffer read with glGetBufferSubData
    GLuint asyncRing_ = 0;   // persistently mapped, one slot per frame
    const std::byte* asyncMapped_ = nullptr;
    GLsync asyncFences_[kRingSlots] = {};
    uint64_t asyncFrame_[kRingSlots] = {};
    uint64_t frame_ = 0;

    int flattened_ = 0;
    int flattenedAge_ = -1;
    double readbackMs_ = 0.0;
};
