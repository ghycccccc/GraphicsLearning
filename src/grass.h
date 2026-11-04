#pragma once
#include <GL/glew.h>
#include <glm/glm.hpp>

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
};

// Per-blade data as stored on the GPU (std430 / vertex attributes, 32 bytes).
struct GrassInstance {
    glm::vec4 a; // root.x, root.z, yaw (radians), color variation
    glm::vec4 b; // height, width, bend.x, bend.z (world-space tip offset as a fraction of height)
};
static_assert(sizeof(GrassInstance) == 32);

// Wind sway amount for a blade rooted at `rootXZ`. Must match windSway() in the shaders.
float windSway(glm::vec2 rootXZ, float time, float strength);

// A square field of grass blades: CPU copy of the per-blade data, a static GPU copy,
// and the single blade mesh every path draws from.
class GrassField {
public:
    static constexpr int kMaxSegments = 16;

    bool init();
    void destroy();

    // Rebuilds the blade mesh with `segments` levels (2*segments-1 triangles).
    void setSegments(int segments);
    int segments() const { return segments_; }

    // Places `count` blades at `density` blades per m^2 and uploads them.
    // Returns false if the GPU could not allocate the instance buffer.
    bool regenerate(int count, float density, uint32_t seed);

    int count() const { return count_; }
    float fieldSize() const { return fieldSize_; }
    const std::vector<GrassInstance>& instances() const { return instances_; }

    const std::vector<glm::vec2>& bladeVertices() const { return bladeVertices_; }
    const std::vector<uint16_t>& bladeIndices() const { return bladeIndices_; }
    int verticesPerBlade() const { return int(bladeVertices_.size()); }
    int indicesPerBlade() const { return int(bladeIndices_.size()); }
    int trianglesPerBlade() const { return indicesPerBlade() / 3; }

    GLuint bladeVertexBuffer() const { return vbo_; }
    GLuint bladeIndexBuffer() const { return ibo_; }
    GLuint instanceBuffer() const { return instanceBuffer_; }

    // Bumped whenever blades or the blade mesh change, so derived data can rebuild.
    uint64_t version() const { return version_; }

private:
    std::vector<GrassInstance> instances_;
    std::vector<glm::vec2> bladeVertices_;
    std::vector<uint16_t> bladeIndices_;
    GLuint vbo_ = 0, ibo_ = 0;
    GLuint instanceBuffer_ = 0;
    uint64_t instanceBytes_ = 0;
    uint64_t meshBytes_ = 0;
    uint64_t version_ = 0;
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
