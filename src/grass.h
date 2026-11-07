#pragma once
#include <GL/glew.h>
#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

// Shared per-frame inputs for scene shaders.
struct SceneParams {
    glm::mat4 viewProj{1.0f};
    glm::vec3 cameraPos{0.0f};
    float time = 0.0f;
    glm::vec2 windDir{1.0f, 0.0f};
    float windStrength = 0.35f;
    glm::vec3 sunDir{0.0f, 1.0f, 0.0f};
    glm::vec3 fogColor{0.6f, 0.75f, 0.9f};
    float fogDensity = 0.01f;
    glm::vec2 trampleCenter{0.0f}; // xz of the trample sphere
    float trampleRadius = 0.0f;    // 0 = no trampling
    bool colorByDraw = false;      // debug view: tint blades by gl_DrawID
};

// CPU-side per-blade record. The per-blade and batch paths read it directly; the instanced
// paths get it re-encoded into a static GPU buffer (see InstanceLayout) plus a bend buffer.
struct GrassInstance {
    glm::vec4 a; // root.x, root.z, yaw (radians), color variation
    glm::vec4 b; // height, width, bend.x, bend.z (world-space tip offset as a fraction of height)
};
static_assert(sizeof(GrassInstance) == 32);

// Encoding of the static per-blade data the instanced paths read.
//   Float32:  vec4(root.x, root.z, yaw/2pi, variation), vec4(height, width, restBend/0.5, 0)  32 B
//   Packed16: float root.x, float root.z, u16 yaw/2pi | u8 variation | u8 restBend/0.5,
//             half2(height, width)                                                         16 B
// The current bend (when animated outside the vertex shader) lives in a separate buffer:
// vec2 (8 B) or half2 (4 B).
enum class InstanceLayout { Float32, Packed16, kCount };
inline constexpr const char* kLayoutNames[] = {"32 B floats", "16 B packed"};
inline constexpr const char* kLayoutKeys[] = {"float", "packed"};
inline uint32_t staticStride(InstanceLayout layout) { return layout == InstanceLayout::Float32 ? 32 : 16; }
inline uint32_t bendStride(InstanceLayout layout) { return layout == InstanceLayout::Float32 ? 8 : 4; }

// Writes `bend` into element `index` of a bend buffer with the given layout.
void storeBend(void* bends, size_t index, glm::vec2 bend, InstanceLayout layout);

// Wind sway amount for a blade rooted at `rootXZ`. Must match windSway() in the shaders.
float windSway(glm::vec2 rootXZ, float time, float strength);

// A square field of grass blades: CPU records, the static GPU encoding of them, and the
// single blade mesh every path draws from.
class GrassField {
public:
    static constexpr int kMaxSegments = 16;
    static constexpr int kLodCount = 3;

    // One blade mesh inside the shared GPU vertex/index buffers.
    struct LodMesh {
        uint32_t indexCount;
        uint32_t firstIndex;
        int32_t baseVertex;
        uint32_t triangles;
    };

    bool init();
    void destroy();

    // Rebuilds the blade mesh with `segments` levels (2*segments-1 triangles).
    void setSegments(int segments);
    int segments() const { return segments_; }

    // Places `count` blades at `density` blades per m^2 and uploads them.
    // Returns false if the GPU could not allocate the static buffer.
    bool regenerate(int count, float density, uint32_t seed);

    // Re-encodes the static buffer. Returns false if the GPU allocation failed.
    bool setLayout(InstanceLayout layout);
    InstanceLayout layout() const { return layout_; }

    int count() const { return count_; }
    float fieldSize() const { return fieldSize_; }
    const std::vector<GrassInstance>& instances() const { return instances_; }

    const std::vector<glm::vec2>& bladeVertices() const { return bladeVertices_; }
    const std::vector<uint16_t>& bladeIndices() const { return bladeIndices_; }
    int verticesPerBlade() const { return int(bladeVertices_.size()); }
    int indicesPerBlade() const { return int(bladeIndices_.size()); }
    int trianglesPerBlade() const { return indicesPerBlade() / 3; }

    // LOD 0 is the full blade (same as bladeVertices()/bladeIndices()), at offset 0 in the GPU buffers.
    const LodMesh& lodMesh(int lod) const { return lodMeshes_[lod]; }

    GLuint bladeVertexBuffer() const { return vbo_; }
    GLuint bladeIndexBuffer() const { return ibo_; }
    GLuint staticBuffer() const { return staticBuffer_; }
    // CPU copy of the static buffer, for paths that re-upload everything each frame.
    const std::vector<std::byte>& staticBytes() const { return staticBytes_; }

    // Bumped whenever blades, layout or the blade mesh change, so derived data can rebuild.
    uint64_t version() const { return version_; }

private:
    bool rebuildStatic();

    std::vector<GrassInstance> instances_;
    std::vector<std::byte> staticBytes_;
    std::vector<glm::vec2> bladeVertices_;
    std::vector<uint16_t> bladeIndices_;
    LodMesh lodMeshes_[kLodCount] = {};
    GLuint vbo_ = 0, ibo_ = 0;
    GLuint staticBuffer_ = 0;
    uint64_t staticBufferBytes_ = 0;
    uint64_t meshBytes_ = 0;
    uint64_t version_ = 0;
    InstanceLayout layout_ = InstanceLayout::Float32;
    int segments_ = 0;
    int count_ = 0;
    float fieldSize_ = 0.0f;
};

// Flat ground quad generated in the vertex shader.
class Ground {
public:
    bool init();
    void destroy();
    void draw(const SceneParams& scene, float halfSize) const;

private:
    GLuint program_ = 0;
    GLuint vao_ = 0;
    GLint locViewProj_ = -1, locHalfSize_ = -1, locSunDir_ = -1, locCameraPos_ = -1;
    GLint locFogColor_ = -1, locFogDensity_ = -1;
};
