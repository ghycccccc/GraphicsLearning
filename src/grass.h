#pragma once
#include <GL/glew.h>
#include <glm/glm.hpp>

#include <cstdint>

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

// Per-blade data as stored on the GPU (std430, 32 bytes).
struct GrassInstance {
    glm::vec4 posYaw; // xyz root position, w yaw (radians)
    glm::vec4 params; // x height, y width, z forward bend, w color variation
};

// A square field of grass blades, all drawn as instances of one blade mesh.
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
    void draw(const SceneParams& scene) const;

    int count() const { return count_; }
    float fieldSize() const { return fieldSize_; }
    int trianglesPerBlade() const { return indexCount_ / 3; }
    int verticesPerBlade() const { return vertexCount_; }

private:
    GLuint program_ = 0;
    GLuint vao_ = 0, vbo_ = 0, ibo_ = 0;
    GLuint instanceBuffer_ = 0;
    uint64_t instanceBytes_ = 0;
    uint64_t meshBytes_ = 0;
    int segments_ = 0;
    int vertexCount_ = 0;
    int indexCount_ = 0;
    int count_ = 0;
    float fieldSize_ = 0.0f;
    GLint locViewProj_ = -1, locTime_ = -1, locWindDir_ = -1, locWindStrength_ = -1;
    GLint locSunDir_ = -1, locCameraPos_ = -1, locFogColor_ = -1, locFogDensity_ = -1;
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
