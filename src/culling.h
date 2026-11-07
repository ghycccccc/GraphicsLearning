#pragma once
#include "grass.h"

#include <GL/glew.h>
#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

// Layout of one indirect draw, as glDrawElementsIndirect / glMultiDrawElementsIndirect read it
// from GL_DRAW_INDIRECT_BUFFER. Written by the CPU (chunk culling) or a compute shader (GPU culling).
struct DrawElementsIndirectCommand {
    uint32_t count;         // indices per instance (the blade mesh)
    uint32_t instanceCount; // how many blades to draw
    uint32_t firstIndex;    // which LOD mesh inside the shared index buffer
    int32_t baseVertex;
    uint32_t baseInstance;  // where this draw's ids start in the visible-id list (gl_BaseInstance)
};
static_assert(sizeof(DrawElementsIndirectCommand) == 20);

// The camera culling is done for. Usually the render camera; differs while culling is frozen.
struct CullView {
    glm::mat4 viewProj{1.0f};
    glm::vec3 cameraPos{0.0f};
    float maxDistance = 0.0f; // 0 = no distance culling
};

// Conservative bounds: a blade bent by up to kCullBendReach x its height still fits.
inline constexpr float kCullBendReach = 2.0f;

inline void bladeBounds(const GrassInstance& blade, glm::vec3& center, float& radius) {
    const float h = blade.b.x;
    center = {blade.a.x, 0.5f * h, blade.a.y};
    radius = h * (0.5f + kCullBendReach);
}

struct Frustum {
    glm::vec4 planes[6]; // xyz = inward normal (unit length), w = offset; inside when dot(n, p) + w >= 0

    static Frustum fromViewProj(const glm::mat4& viewProj);
    bool sphereOutside(const glm::vec3& center, float radius) const;
    bool boxOutside(const glm::vec3& lo, const glm::vec3& hi) const;
};

// Per-blade CPU culling. Writes the ids of visible blades to `out` (room for field.count()) and
// returns how many there are. Same test as grass_cull.comp, without LOD.
uint32_t cullBladesCpu(const GrassField& field, const Frustum& frustum, const CullView& view, bool parallel,
                       uint32_t* out);

// A uniform grid of square chunks over the field. Blades keep their order in the field; the grid
// owns a chunk-sorted copy of the ids so every chunk is one contiguous range of that list.
class ChunkGrid {
public:
    static constexpr float kChunkSize = 8.0f; // metres

    void update(const GrassField& field); // rebuilds when the field changed
    void release();

    // Writes one indirect draw per visible chunk (drawing `mesh`) and returns how many.
    // `visibleBlades` receives the number of blades those draws cover.
    uint32_t cull(const Frustum& frustum, const CullView& view, const GrassField::LodMesh& mesh,
                  DrawElementsIndirectCommand* out, uint32_t& visibleBlades) const;

    GLuint sortedIdBuffer() const { return sortedIds_; }
    int chunkCount() const { return int(chunks_.size()); }

private:
    struct Chunk {
        glm::vec3 lo, hi; // bounds of every blade sphere in the chunk
        uint32_t first, count;
    };
    std::vector<Chunk> chunks_;
    GLuint sortedIds_ = 0;
    uint64_t bytes_ = 0;
    uint64_t version_ = ~0ull;
};
