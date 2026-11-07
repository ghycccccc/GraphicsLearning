#include "culling.h"

#include "counters.h"

#include <algorithm>
#include <cmath>
#include <execution>
#include <numeric>

// ===== MUST READ (M4 #1): frustum planes from a view-projection matrix ====================
// Gribb & Hartmann: a clip-space point is inside when -w <= x, y, z <= w. Writing clip = M * p
// with row i of M as r_i, "x >= -w" becomes dot(r3 + r0, p) >= 0, i.e. a plane (r3 + r0) in
// world space. The six planes are r3 +/- r0 (left/right), r3 +/- r1 (bottom/top) and
// r3 +/- r2 (near/far). Normalizing makes dot(n, p) + w a true distance, which the sphere test
// needs. glm is column-major, so row i is (m[0][i], m[1][i], m[2][i], m[3][i]).
Frustum Frustum::fromViewProj(const glm::mat4& m) {
    auto row = [&](int i) { return glm::vec4(m[0][i], m[1][i], m[2][i], m[3][i]); };
    const glm::vec4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
    Frustum f;
    f.planes[0] = r3 + r0;
    f.planes[1] = r3 - r0;
    f.planes[2] = r3 + r1;
    f.planes[3] = r3 - r1;
    f.planes[4] = r3 + r2;
    f.planes[5] = r3 - r2;
    for (glm::vec4& p : f.planes) p /= glm::length(glm::vec3(p));
    return f;
}

// A sphere is outside if it lies entirely behind any single plane. This is conservative: a
// sphere near a frustum corner can be outside while touching no single plane's back side, and
// it is then drawn anyway. That costs a little work, never a missing blade.
bool Frustum::sphereOutside(const glm::vec3& c, float r) const {
    for (const glm::vec4& p : planes) {
        if (glm::dot(glm::vec3(p), c) + p.w < -r) return true;
    }
    return false;
}

// Box version: test only the corner furthest along each plane normal (the "positive vertex").
// If even that corner is behind the plane, the whole box is.
bool Frustum::boxOutside(const glm::vec3& lo, const glm::vec3& hi) const {
    for (const glm::vec4& p : planes) {
        const glm::vec3 positive(p.x >= 0 ? hi.x : lo.x, p.y >= 0 ? hi.y : lo.y, p.z >= 0 ? hi.z : lo.z);
        if (glm::dot(glm::vec3(p), positive) + p.w < 0.0f) return true;
    }
    return false;
}
// ==========================================================================================

namespace {
constexpr int kCullBlock = 65536; // blades per parallel task

bool bladeVisible(const GrassInstance& blade, const Frustum& frustum, const CullView& view) {
    glm::vec3 center;
    float radius;
    bladeBounds(blade, center, radius);
    if (frustum.sphereOutside(center, radius)) return false;
    return view.maxDistance <= 0.0f || glm::distance(view.cameraPos, center) - radius <= view.maxDistance;
}
} // namespace

// ===== MUST READ (M4 #2): CPU culling and parallel stream compaction ======================
// Single-threaded culling is a filter: test each blade, append the visible ones.
// In parallel the problem is the output: threads cannot all append to one list without a lock
// or an atomic per blade. The standard answer is two passes:
//   1. each block of blades counts its visible blades (no writes, fully parallel);
//   2. an exclusive prefix sum over the counts gives each block its first output slot;
//   3. each block runs the test again and writes its ids from that slot on.
// The output keeps blade order and no thread ever waits on another. This is exactly the shape
// of the GPU prefix-sum compaction in grass_cull.comp, with blocks instead of workgroups.
// Note `out` is usually persistently mapped (write-combined) memory: write it, never read it.
uint32_t cullBladesCpu(const GrassField& field, const Frustum& frustum, const CullView& view, bool parallel,
                       uint32_t* out) {
    const GrassInstance* blades = field.instances().data();
    const uint32_t count = uint32_t(field.count());
    if (!parallel) {
        uint32_t visible = 0;
        for (uint32_t i = 0; i < count; ++i) {
            if (bladeVisible(blades[i], frustum, view)) out[visible++] = i;
        }
        return visible;
    }

    const uint32_t blockCount = (count + kCullBlock - 1) / kCullBlock;
    std::vector<uint32_t> blocks(blockCount), counts(blockCount), offsets(blockCount);
    std::iota(blocks.begin(), blocks.end(), 0u);
    auto range = [&](uint32_t block) {
        return std::pair<uint32_t, uint32_t>(block * kCullBlock, std::min(count, (block + 1) * kCullBlock));
    };
    std::for_each(std::execution::par, blocks.begin(), blocks.end(), [&](uint32_t block) {
        auto [begin, end] = range(block);
        uint32_t visible = 0;
        for (uint32_t i = begin; i < end; ++i) visible += bladeVisible(blades[i], frustum, view) ? 1 : 0;
        counts[block] = visible;
    });
    std::exclusive_scan(counts.begin(), counts.end(), offsets.begin(), 0u);
    std::for_each(std::execution::par, blocks.begin(), blocks.end(), [&](uint32_t block) {
        auto [begin, end] = range(block);
        uint32_t* dst = out + offsets[block];
        for (uint32_t i = begin; i < end; ++i) {
            if (bladeVisible(blades[i], frustum, view)) *dst++ = i;
        }
    });
    return blockCount ? offsets.back() + counts.back() : 0;
}
// ==========================================================================================

void ChunkGrid::update(const GrassField& field) {
    if (sortedIds_ && version_ == field.version()) return;
    release();
    version_ = field.version();

    const float half = field.fieldSize() * 0.5f;
    const int cells = std::max(1, int(std::ceil(field.fieldSize() / kChunkSize)));
    auto cellOf = [&](const GrassInstance& blade) {
        const int cx = std::clamp(int((blade.a.x + half) / kChunkSize), 0, cells - 1);
        const int cz = std::clamp(int((blade.a.y + half) / kChunkSize), 0, cells - 1);
        return cz * cells + cx;
    };

    // Counting sort of blade ids by chunk.
    const std::vector<GrassInstance>& blades = field.instances();
    std::vector<uint32_t> cellCounts(size_t(cells) * size_t(cells), 0);
    for (const GrassInstance& blade : blades) ++cellCounts[size_t(cellOf(blade))];
    std::vector<uint32_t> cellStart(cellCounts.size());
    std::exclusive_scan(cellCounts.begin(), cellCounts.end(), cellStart.begin(), 0u);

    chunks_.resize(cellCounts.size());
    for (size_t c = 0; c < chunks_.size(); ++c) {
        chunks_[c] = {glm::vec3(1e30f), glm::vec3(-1e30f), cellStart[c], cellCounts[c]};
    }
    std::vector<uint32_t> sorted(blades.size());
    std::vector<uint32_t> cursor = cellStart;
    for (uint32_t i = 0; i < uint32_t(blades.size()); ++i) {
        const int cell = cellOf(blades[i]);
        sorted[cursor[size_t(cell)]++] = i;
        glm::vec3 center;
        float radius;
        bladeBounds(blades[i], center, radius);
        Chunk& chunk = chunks_[size_t(cell)];
        chunk.lo = glm::min(chunk.lo, center - radius);
        chunk.hi = glm::max(chunk.hi, center + radius);
    }
    chunks_.erase(std::remove_if(chunks_.begin(), chunks_.end(), [](const Chunk& c) { return c.count == 0; }),
                  chunks_.end());

    bytes_ = sorted.size() * sizeof(uint32_t);
    glCreateBuffers(1, &sortedIds_);
    glNamedBufferStorage(sortedIds_, GLsizeiptr(std::max<uint64_t>(bytes_, 4)), sorted.data(), 0);
    g_residentBytes += bytes_;
    g_counters.bytesUploaded += bytes_;
}

void ChunkGrid::release() {
    glDeleteBuffers(1, &sortedIds_);
    sortedIds_ = 0;
    g_residentBytes -= bytes_;
    bytes_ = 0;
    chunks_.clear();
    version_ = ~0ull;
}

// ===== MUST READ (M4 #3): chunk culling feeding multi-draw indirect =======================
// Instead of testing millions of blades, test a few thousand chunk boxes. Each visible chunk is
// already a contiguous range of the chunk-sorted id list, so it becomes one indirect draw
// command: draw `count` instances of the blade mesh, with ids starting at `baseInstance`.
// The vertex shader reads visibleIds[gl_BaseInstance + gl_InstanceID]. All commands go out
// in ONE glMultiDrawElementsIndirect call: thousands of draws for the price of one API call.
// The trade-off: a chunk touching the frustum draws all its blades, so culling is coarser
// (more blades drawn than per-blade culling) in exchange for almost no CPU time.
uint32_t ChunkGrid::cull(const Frustum& frustum, const CullView& view, const GrassField::LodMesh& mesh,
                         DrawElementsIndirectCommand* out, uint32_t& visibleBlades) const {
    uint32_t commands = 0;
    visibleBlades = 0;
    for (const Chunk& chunk : chunks_) {
        if (frustum.boxOutside(chunk.lo, chunk.hi)) continue;
        if (view.maxDistance > 0.0f) {
            const glm::vec3 closest = glm::clamp(view.cameraPos, chunk.lo, chunk.hi);
            if (glm::distance(closest, view.cameraPos) > view.maxDistance) continue;
        }
        out[commands++] = {mesh.indexCount, chunk.count, mesh.firstIndex, mesh.baseVertex, chunk.first};
        visibleBlades += chunk.count;
    }
    return commands;
}
// ==========================================================================================
