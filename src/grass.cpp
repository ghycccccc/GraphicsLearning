#include "grass.h"

#include "counters.h"
#include "gl_util.h"

#include <cmath>
#include <cstdio>
#include <random>

float windSway(glm::vec2 rootXZ, float time, float strength) {
    const float phase = time * 1.7f + rootXZ.x * 0.13f + rootXZ.y * 0.09f;
    return (0.5f + 0.35f * std::sin(phase) + 0.15f * std::sin(phase * 2.3f + 1.0f)) * strength;
}

// --- GrassField --------------------------------------------------------------

bool GrassField::init() {
    setSegments(4);
    return true;
}

void GrassField::destroy() {
    glDeleteBuffers(1, &vbo_);
    glDeleteBuffers(1, &ibo_);
    glDeleteBuffers(1, &instanceBuffer_);
}

void GrassField::setSegments(int segments) {
    segments_ = segments < 1 ? 1 : (segments > kMaxSegments ? kMaxSegments : segments);

    // Blade in a local (x across, t up) space: pairs of vertices per level, one tip vertex.
    bladeVertices_.clear();
    for (int i = 0; i < segments_; ++i) {
        const float t = float(i) / float(segments_);
        bladeVertices_.push_back({-0.5f, t});
        bladeVertices_.push_back({0.5f, t});
    }
    bladeVertices_.push_back({0.0f, 1.0f});

    bladeIndices_.clear();
    for (int i = 0; i < segments_ - 1; ++i) {
        const uint16_t v = uint16_t(i * 2);
        bladeIndices_.insert(bladeIndices_.end(), {v, uint16_t(v + 1), uint16_t(v + 2)});
        bladeIndices_.insert(bladeIndices_.end(), {uint16_t(v + 1), uint16_t(v + 3), uint16_t(v + 2)});
    }
    const uint16_t top = uint16_t((segments_ - 1) * 2);
    bladeIndices_.insert(bladeIndices_.end(), {top, uint16_t(top + 1), uint16_t(top + 2)});

    const auto vboBytes = GLsizeiptr(bladeVertices_.size() * sizeof(glm::vec2));
    const auto iboBytes = GLsizeiptr(bladeIndices_.size() * sizeof(uint16_t));

    glDeleteBuffers(1, &vbo_);
    glDeleteBuffers(1, &ibo_);
    g_residentBytes -= meshBytes_;
    glCreateBuffers(1, &vbo_);
    glNamedBufferStorage(vbo_, vboBytes, bladeVertices_.data(), 0);
    glCreateBuffers(1, &ibo_);
    glNamedBufferStorage(ibo_, iboBytes, bladeIndices_.data(), 0);
    meshBytes_ = uint64_t(vboBytes + iboBytes);
    g_residentBytes += meshBytes_;
    g_counters.bytesUploaded += meshBytes_;
    ++version_;
}

bool GrassField::regenerate(int count, float density, uint32_t seed) {
    fieldSize_ = std::sqrt(float(count) / density);
    ++version_;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    instances_.resize(size_t(count));
    for (GrassInstance& blade : instances_) {
        const float x = (u01(rng) - 0.5f) * fieldSize_;
        const float z = (u01(rng) - 0.5f) * fieldSize_;
        const float yaw = u01(rng) * 6.2831853f;
        const float height = 0.35f + 0.5f * u01(rng);
        const float width = 0.035f + 0.03f * u01(rng);
        const float bend = 0.1f + 0.35f * u01(rng); // rest lean along the blade's facing
        blade.a = {x, z, yaw, u01(rng)};
        blade.b = {height, width, std::sin(yaw) * bend, std::cos(yaw) * bend};
    }

    if (instanceBuffer_) {
        glDeleteBuffers(1, &instanceBuffer_);
        g_residentBytes -= instanceBytes_;
        instanceBuffer_ = 0;
        instanceBytes_ = 0;
        count_ = 0;
    }
    while (glGetError() != GL_NO_ERROR) {
    }
    const uint64_t bytes = uint64_t(count) * sizeof(GrassInstance);
    glCreateBuffers(1, &instanceBuffer_);
    glNamedBufferStorage(instanceBuffer_, GLsizeiptr(bytes), instances_.data(), 0);
    if (glGetError() == GL_OUT_OF_MEMORY) {
        std::fprintf(stderr, "Out of GPU memory allocating %.0f MB of instances\n",
                     double(bytes) / (1024.0 * 1024.0));
        glDeleteBuffers(1, &instanceBuffer_);
        instanceBuffer_ = 0;
        instances_.clear();
        return false;
    }
    instanceBytes_ = bytes;
    g_residentBytes += instanceBytes_;
    g_counters.bytesUploaded += instanceBytes_;
    count_ = count;
    return true;
}

// --- Ground ------------------------------------------------------------------

bool Ground::init() {
    program_ = loadProgram({{GL_VERTEX_SHADER, "ground.vert"}, {GL_FRAGMENT_SHADER, "ground.frag"}});
    if (!program_) return false;
    glCreateVertexArrays(1, &vao_); // attribute-less; corners come from gl_VertexID
    locViewProj_ = glGetUniformLocation(program_, "uViewProj");
    locHalfSize_ = glGetUniformLocation(program_, "uHalfSize");
    locSunDir_ = glGetUniformLocation(program_, "uSunDir");
    locCameraPos_ = glGetUniformLocation(program_, "uCameraPos");
    locFogColor_ = glGetUniformLocation(program_, "uFogColor");
    locFogDensity_ = glGetUniformLocation(program_, "uFogDensity");
    return true;
}

void Ground::destroy() {
    glDeleteProgram(program_);
    glDeleteVertexArrays(1, &vao_);
}

void Ground::draw(const SceneParams& scene, float halfSize) const {
    glUseProgram(program_);
    glProgramUniformMatrix4fv(program_, locViewProj_, 1, GL_FALSE, &scene.viewProj[0][0]);
    glProgramUniform1f(program_, locHalfSize_, halfSize);
    glProgramUniform3fv(program_, locSunDir_, 1, &scene.sunDir[0]);
    glProgramUniform3fv(program_, locCameraPos_, 1, &scene.cameraPos[0]);
    glProgramUniform3fv(program_, locFogColor_, 1, &scene.fogColor[0]);
    glProgramUniform1f(program_, locFogDensity_, scene.fogDensity);
    glBindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    g_counters.drawCalls += 1;
    g_counters.triangles += 2;
    g_counters.stateChanges += 2;
    g_counters.uniformUpdates += 6;
}
