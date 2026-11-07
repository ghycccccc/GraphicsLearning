#include "grass.h"

#include "counters.h"
#include "gl_util.h"

#include <glm/gtc/packing.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

void storeBend(void* bends, size_t index, glm::vec2 bend, InstanceLayout layout) {
    if (layout == InstanceLayout::Float32) static_cast<glm::vec2*>(bends)[index] = bend;
    else static_cast<uint32_t*>(bends)[index] = glm::packHalf2x16(bend);
}

namespace {
void encodeStatic(const GrassInstance& blade, InstanceLayout layout, std::byte* out) {
    const float yaw01 = blade.a.z / 6.2831853f;
    const float restBend01 = glm::length(glm::vec2(blade.b.z, blade.b.w)) / 0.5f;
    if (layout == InstanceLayout::Float32) {
        const glm::vec4 words[2] = {{blade.a.x, blade.a.y, yaw01, blade.a.w},
                                    {blade.b.x, blade.b.y, restBend01, 0.0f}};
        std::memcpy(out, words, sizeof(words));
        return;
    }
    auto unorm = [](float v, float maxValue) {
        return uint32_t(std::lround(std::clamp(v, 0.0f, 1.0f) * maxValue));
    };
    const uint32_t words[4] = {
        glm::floatBitsToUint(blade.a.x),
        glm::floatBitsToUint(blade.a.y),
        unorm(yaw01, 65535.0f) | unorm(blade.a.w, 255.0f) << 16 | unorm(restBend01, 255.0f) << 24,
        glm::packHalf2x16({blade.b.x, blade.b.y}),
    };
    std::memcpy(out, words, sizeof(words));
}
} // namespace

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
    glDeleteBuffers(1, &staticBuffer_);
}

namespace {
// Blade in a local (x across, t up) space: pairs of vertices per level, one tip vertex.
void buildBlade(int segments, std::vector<glm::vec2>& vertices, std::vector<uint16_t>& indices) {
    vertices.clear();
    for (int i = 0; i < segments; ++i) {
        const float t = float(i) / float(segments);
        vertices.push_back({-0.5f, t});
        vertices.push_back({0.5f, t});
    }
    vertices.push_back({0.0f, 1.0f});

    indices.clear();
    for (int i = 0; i < segments - 1; ++i) {
        const uint16_t v = uint16_t(i * 2);
        indices.insert(indices.end(), {v, uint16_t(v + 1), uint16_t(v + 2)});
        indices.insert(indices.end(), {uint16_t(v + 1), uint16_t(v + 3), uint16_t(v + 2)});
    }
    const uint16_t top = uint16_t((segments - 1) * 2);
    indices.insert(indices.end(), {top, uint16_t(top + 1), uint16_t(top + 2)});
}
} // namespace

void GrassField::setSegments(int segments) {
    segments_ = segments < 1 ? 1 : (segments > kMaxSegments ? kMaxSegments : segments);
    buildBlade(segments_, bladeVertices_, bladeIndices_);

    // The GPU buffers hold all LOD meshes back to back: LOD 0 first (so draws that ignore LOD
    // use offset 0), then half the segments, then a single triangle. A draw selects one with
    // firstIndex + baseVertex; one multi-draw-indirect call can draw all three.
    const int lodSegments[kLodCount] = {segments_, std::max(1, segments_ / 2), 1};
    std::vector<glm::vec2> gpuVertices;
    std::vector<uint16_t> gpuIndices;
    for (int lod = 0; lod < kLodCount; ++lod) {
        std::vector<glm::vec2> vertices;
        std::vector<uint16_t> indices;
        buildBlade(lodSegments[lod], vertices, indices);
        lodMeshes_[lod] = {uint32_t(indices.size()), uint32_t(gpuIndices.size()), int32_t(gpuVertices.size()),
                           uint32_t(indices.size() / 3)};
        gpuVertices.insert(gpuVertices.end(), vertices.begin(), vertices.end());
        gpuIndices.insert(gpuIndices.end(), indices.begin(), indices.end());
    }

    const auto vboBytes = GLsizeiptr(gpuVertices.size() * sizeof(glm::vec2));
    const auto iboBytes = GLsizeiptr(gpuIndices.size() * sizeof(uint16_t));

    glDeleteBuffers(1, &vbo_);
    glDeleteBuffers(1, &ibo_);
    g_residentBytes -= meshBytes_;
    glCreateBuffers(1, &vbo_);
    glNamedBufferStorage(vbo_, vboBytes, gpuVertices.data(), 0);
    glCreateBuffers(1, &ibo_);
    glNamedBufferStorage(ibo_, iboBytes, gpuIndices.data(), 0);
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

    count_ = count;
    if (!rebuildStatic()) {
        instances_.clear();
        count_ = 0;
        return false;
    }
    return true;
}

bool GrassField::setLayout(InstanceLayout layout) {
    if (layout == layout_) return true;
    layout_ = layout;
    ++version_;
    return rebuildStatic();
}

bool GrassField::rebuildStatic() {
    if (staticBuffer_) {
        glDeleteBuffers(1, &staticBuffer_);
        g_residentBytes -= staticBufferBytes_;
        staticBuffer_ = 0;
        staticBufferBytes_ = 0;
    }
    const uint32_t stride = staticStride(layout_);
    staticBytes_.resize(instances_.size() * stride);
    for (size_t i = 0; i < instances_.size(); ++i) encodeStatic(instances_[i], layout_, &staticBytes_[i * stride]);
    if (staticBytes_.empty()) return true;

    while (glGetError() != GL_NO_ERROR) {
    }
    glCreateBuffers(1, &staticBuffer_);
    glNamedBufferStorage(staticBuffer_, GLsizeiptr(staticBytes_.size()), staticBytes_.data(), 0);
    if (glGetError() == GL_OUT_OF_MEMORY) {
        std::fprintf(stderr, "Out of GPU memory allocating %.0f MB of blade data\n",
                     double(staticBytes_.size()) / (1024.0 * 1024.0));
        glDeleteBuffers(1, &staticBuffer_);
        staticBuffer_ = 0;
        return false;
    }
    staticBufferBytes_ = staticBytes_.size();
    g_residentBytes += staticBufferBytes_;
    g_counters.bytesUploaded += staticBufferBytes_;
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
