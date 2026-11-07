#include "gpu_culler.h"

#include "counters.h"
#include "gl_util.h"
#include "profiler.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace {
constexpr uint32_t kGroupSize = 256; // local_size_x of grass_cull.comp
}

const GpuCuller::Program& GpuCuller::cullProgram(const std::string& defines) {
    auto it = programs_.find(defines);
    if (it != programs_.end()) return it->second;
    Program p;
    p.id = loadProgram({{GL_COMPUTE_SHADER, "grass_cull.comp"}}, defines);
    if (p.id) {
        p.count = glGetUniformLocation(p.id, "uCount");
        p.region = glGetUniformLocation(p.id, "uRegion");
        p.planes = glGetUniformLocation(p.id, "uPlanes");
        p.cameraPos = glGetUniformLocation(p.id, "uCameraPos");
        p.maxDistance = glGetUniformLocation(p.id, "uMaxDistance");
        p.lodEnabled = glGetUniformLocation(p.id, "uLodEnabled");
        p.lodDistances = glGetUniformLocation(p.id, "uLodDistances");
    }
    return programs_.emplace(defines, p).first->second;
}

bool GpuCuller::init() {
    scanProgram_.id = loadProgram({{GL_COMPUTE_SHADER, "grass_scan.comp"}});
    if (!scanProgram_.id || !cullProgram("#define COMPACTION_WORKGROUP\n").id) return false;
    scanProgram_.blockCount = glGetUniformLocation(scanProgram_.id, "uBlockCount");

    glCreateBuffers(1, &commands_);
    glNamedBufferStorage(commands_, sizeof(readback_), nullptr, GL_DYNAMIC_STORAGE_BIT);

    const GLbitfield flags = GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
    glCreateBuffers(1, &statsRing_);
    glNamedBufferStorage(statsRing_, kStatsSlots * kStatsSlotBytes, nullptr, flags);
    statsMapped_ = static_cast<const std::byte*>(glMapNamedBufferRange(statsRing_, 0, kStatsSlots * kStatsSlotBytes, flags));
    return statsMapped_ != nullptr;
}

void GpuCuller::destroy() {
    release();
    for (auto& [defines, program] : programs_) glDeleteProgram(program.id);
    glDeleteProgram(scanProgram_.id);
    glDeleteBuffers(1, &commands_);
    glUnmapNamedBuffer(statsRing_);
    glDeleteBuffers(1, &statsRing_);
}

void GpuCuller::release() {
    glDeleteBuffers(1, &visible_);
    glDeleteBuffers(1, &blockSums_);
    glDeleteBuffers(1, &blockOffsets_);
    visible_ = blockSums_ = blockOffsets_ = 0;
    g_residentBytes -= visibleBytes_ + 2 * blockBytes_;
    visibleBytes_ = blockBytes_ = 0;
    for (GLsync& fence : statsFences_) {
        if (fence) glDeleteSync(fence);
        fence = nullptr;
    }
    statsAge_ = -1;
}

void GpuCuller::ensureCapacity(uint32_t blades, bool lod) {
    // Each LOD gets a region big enough for every blade: wasteful, but no draw can overflow.
    const uint64_t visibleBytes = uint64_t(blades) * (lod ? GrassField::kLodCount : 1) * sizeof(uint32_t);
    const uint64_t blockBytes = uint64_t((blades + kGroupSize - 1) / kGroupSize) * 16; // uvec4 per workgroup
    if (visible_ && visibleBytes == visibleBytes_ && blockBytes == blockBytes_) return;
    release();
    visibleBytes_ = std::max<uint64_t>(visibleBytes, 4);
    blockBytes_ = std::max<uint64_t>(blockBytes, 16);
    glCreateBuffers(1, &visible_);
    glNamedBufferStorage(visible_, GLsizeiptr(visibleBytes_), nullptr, 0);
    glCreateBuffers(1, &blockSums_);
    glNamedBufferStorage(blockSums_, GLsizeiptr(blockBytes_), nullptr, 0);
    glCreateBuffers(1, &blockOffsets_);
    glNamedBufferStorage(blockOffsets_, GLsizeiptr(blockBytes_), nullptr, 0);
    g_residentBytes += visibleBytes_ + 2 * blockBytes_;
}

void GpuCuller::collectStats() {
    // The commands of 2 frames ago have been copied into the ring; read them if the GPU is done.
    const int slot = int((frame_ + kStatsSlots - 2) % kStatsSlots);
    GLsync fence = statsFences_[slot];
    if (!fence) return;
    const GLenum status = glClientWaitSync(fence, 0, 0);
    if (status != GL_ALREADY_SIGNALED && status != GL_CONDITION_SATISFIED) return;
    DrawElementsIndirectCommand commands[GrassField::kLodCount];
    std::memcpy(commands, statsMapped_ + slot * kStatsSlotBytes, sizeof(commands));
    for (int lod = 0; lod < GrassField::kLodCount; ++lod) stats_[lod] = commands[lod].instanceCount;
    statsAge_ = int(frame_ - statsFrame_[slot]);
    glDeleteSync(fence);
    statsFences_[slot] = nullptr;
}

void GpuCuller::run(const GrassField& field, const CullView& view, const GpuCullSettings& settings,
                    CpuProfiler& cpu, GpuProfiler& gpu) {
    const uint32_t count = uint32_t(field.count());
    collectStats();
    ensureCapacity(count, settings.lod);
    readbackMs_ = 0.0;

    // ===== MUST READ (M4 #5): the indirect draw arguments are reset by the CPU ============
    // The CPU fills in everything it knows: which mesh each LOD draws (count / firstIndex /
    // baseVertex) and where its ids start (baseInstance). Only instanceCount, the number the
    // CPU does NOT know, starts at 0 and is filled in by the compute shader.
    DrawElementsIndirectCommand commands[GrassField::kLodCount];
    for (int lod = 0; lod < GrassField::kLodCount; ++lod) {
        const GrassField::LodMesh& mesh = field.lodMesh(lod);
        commands[lod] = {mesh.indexCount, 0, mesh.firstIndex, mesh.baseVertex, uint32_t(lod) * count};
    }
    glNamedBufferSubData(commands_, 0, sizeof(commands), commands);

    const Frustum frustum = Frustum::fromViewProj(view.viewProj);
    std::string layout = field.layout() == InstanceLayout::Packed16 ? "#define PACKED_LAYOUT\n" : "";
    auto setUniforms = [&](const Program& p) {
        glProgramUniform1ui(p.id, p.count, count);
        glProgramUniform1ui(p.id, p.region, count);
        glProgramUniform4fv(p.id, p.planes, 6, &frustum.planes[0][0]);
        glProgramUniform3fv(p.id, p.cameraPos, 1, &view.cameraPos[0]);
        glProgramUniform1f(p.id, p.maxDistance, view.maxDistance);
        glProgramUniform1i(p.id, p.lodEnabled, settings.lod ? 1 : 0);
        glProgramUniform2fv(p.id, p.lodDistances, 1, &settings.lodDistances[0]);
        g_counters.uniformUpdates += 7;
    };
    const GLuint groups = (count + kGroupSize - 1) / kGroupSize; // 16M blades -> 62,500 groups

    {
        GpuScope gpuScope(gpu, "Cull");
        CpuScope cpuScope(cpu, "Cull dispatch");
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, field.staticBuffer());
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, commands_);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, visible_);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, blockSums_);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, blockOffsets_);

        if (settings.compaction == Compaction::PrefixSum) {
            const Program& countPass = cullProgram(layout + "#define PREFIX_COUNT\n");
            const Program& scatterPass = cullProgram(layout + "#define PREFIX_SCATTER\n");
            glUseProgram(countPass.id);
            setUniforms(countPass);
            glDispatchCompute(groups, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            glUseProgram(scanProgram_.id);
            glProgramUniform1ui(scanProgram_.id, scanProgram_.blockCount, groups);
            glDispatchCompute(1, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            glUseProgram(scatterPass.id);
            setUniforms(scatterPass);
            glDispatchCompute(groups, 1, 1);
        } else {
            const char* variant = settings.compaction == Compaction::GlobalAtomic ? "#define COMPACTION_GLOBAL\n"
                                                                                  : "#define COMPACTION_WORKGROUP\n";
            const Program& p = cullProgram(layout + variant);
            glUseProgram(p.id);
            setUniforms(p);
            glDispatchCompute(groups, 1, 1);
        }
        // The next draw reads the commands as indirect arguments (COMMAND bit) and the ids from
        // an SSBO (SHADER_STORAGE bit); copies/readbacks of the commands need BUFFER_UPDATE.
        glMemoryBarrier(GL_COMMAND_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
        g_counters.stateChanges += 6;
    }
    // ======================================================================================

    if (settings.readbackCount) {
        // ===== MUST READ (M4 #6): the anti-pattern this whole design avoids ===============
        // Reading instanceCount back to call glDrawElementsInstanced ourselves looks natural,
        // but the driver must finish the culling dispatch (and everything queued before it)
        // before it can return 60 bytes. The CPU stalls, then the GPU starves. See the M3
        // readback results: costs nothing on an idle CPU, CPU + GPU time with real game logic.
        CpuScope scope(cpu, "Readback cull count (sync)");
        const auto start = std::chrono::steady_clock::now();
        glGetNamedBufferSubData(commands_, 0, sizeof(readback_), readback_);
        readbackMs_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        // ==================================================================================
    }

    // Stats for the UI only: copy the commands into a mapped ring and read them 2 frames later.
    const int slot = int(frame_ % kStatsSlots);
    if (GLsync fence = statsFences_[slot]) {
        glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, GLuint64(1'000'000'000));
        glDeleteSync(fence);
    }
    glCopyNamedBufferSubData(commands_, statsRing_, 0, slot * kStatsSlotBytes, sizeof(readback_));
    statsFences_[slot] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    statsFrame_[slot] = frame_;
    ++frame_;
}
