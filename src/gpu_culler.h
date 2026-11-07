#pragma once
#include "culling.h"
#include "grass.h"

#include <GL/glew.h>
#include <glm/glm.hpp>

#include <cstdint>
#include <map>
#include <string>

class CpuProfiler;
class GpuProfiler;

enum class Compaction { GlobalAtomic, WorkgroupAtomic, PrefixSum, kCount };
inline constexpr const char* kCompactionNames[] = {"Global atomics", "Workgroup-aggregated atomics",
                                                   "Prefix sum (3 passes, ordered)"};
inline constexpr const char* kCompactionKeys[] = {"global", "workgroup", "prefix"};

struct GpuCullSettings {
    Compaction compaction = Compaction::WorkgroupAtomic;
    bool lod = false;
    glm::vec2 lodDistances{15.0f, 40.0f};
    bool readbackCount = false; // anti-pattern: read instanceCount back and draw directly

    bool operator==(const GpuCullSettings&) const = default;
};

// Frustum/distance culling and LOD selection in compute. Output stays on the GPU:
//   commandBuffer(): GrassField::kLodCount indirect draw commands, one per LOD
//   visibleBuffer(): visible blade ids, LOD l starting at l * region()
class GpuCuller {
public:
    bool init();
    void destroy();
    void release(); // frees the per-blade buffers while culling is unused

    void run(const GrassField& field, const CullView& view, const GpuCullSettings& settings, CpuProfiler& cpu,
             GpuProfiler& gpu);

    GLuint commandBuffer() const { return commands_; }
    GLuint visibleBuffer() const { return visible_; }
    // Commands as read back this frame (only valid with readbackCount).
    const DrawElementsIndirectCommand* readbackCommands() const { return readback_; }
    double readbackStallMs() const { return readbackMs_; }

    // Visible blades per LOD, read back asynchronously; statsAgeFrames() frames old, -1 if none yet.
    uint32_t visibleBlades(int lod) const { return stats_[lod]; }
    int statsAgeFrames() const { return statsAge_; }

private:
    struct Program {
        GLuint id = 0;
        GLint count = -1, region = -1, planes = -1, cameraPos = -1, maxDistance = -1, lodEnabled = -1,
              lodDistances = -1, blockCount = -1;
    };
    const Program& cullProgram(const std::string& defines);
    void ensureCapacity(uint32_t blades, bool lod);
    void collectStats();

    static constexpr int kStatsSlots = 3;
    static constexpr GLsizeiptr kStatsSlotBytes = 256;

    std::map<std::string, Program> programs_;
    Program scanProgram_;
    GLuint commands_ = 0, visible_ = 0, blockSums_ = 0, blockOffsets_ = 0;
    uint64_t visibleBytes_ = 0, blockBytes_ = 0;

    GLuint statsRing_ = 0;
    const std::byte* statsMapped_ = nullptr;
    GLsync statsFences_[kStatsSlots] = {};
    uint64_t statsFrame_[kStatsSlots] = {};
    uint64_t frame_ = 0;
    uint32_t stats_[GrassField::kLodCount] = {};
    int statsAge_ = -1;

    DrawElementsIndirectCommand readback_[GrassField::kLodCount] = {};
    double readbackMs_ = 0.0;
};
