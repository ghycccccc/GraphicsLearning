#include "grass.h"

#include "counters.h"
#include "gl_util.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// --- GrassField --------------------------------------------------------------

bool GrassField::init() {
    program_ = loadProgram({{GL_VERTEX_SHADER, "grass.vert"}, {GL_FRAGMENT_SHADER, "grass.frag"}});
    if (!program_) return false;

    glCreateVertexArrays(1, &vao_);
    glEnableVertexArrayAttrib(vao_, 0);
    glVertexArrayAttribFormat(vao_, 0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(vao_, 0, 0);
    setSegments(4);

    locViewProj_ = glGetUniformLocation(program_, "uViewProj");
    locTime_ = glGetUniformLocation(program_, "uTime");
    locWindDir_ = glGetUniformLocation(program_, "uWindDir");
    locWindStrength_ = glGetUniformLocation(program_, "uWindStrength");
    locSunDir_ = glGetUniformLocation(program_, "uSunDir");
    locCameraPos_ = glGetUniformLocation(program_, "uCameraPos");
    locFogColor_ = glGetUniformLocation(program_, "uFogColor");
    locFogDensity_ = glGetUniformLocation(program_, "uFogDensity");
    return true;
}

void GrassField::destroy() {
    glDeleteProgram(program_);
    glDeleteVertexArrays(1, &vao_);
    glDeleteBuffers(1, &vbo_);
    glDeleteBuffers(1, &ibo_);
    glDeleteBuffers(1, &instanceBuffer_);
}

void GrassField::setSegments(int segments) {
    segments_ = segments < 1 ? 1 : (segments > kMaxSegments ? kMaxSegments : segments);

    // Blade in a local (x across, t up) space: pairs of vertices per level, one tip vertex.
    std::vector<glm::vec2> vertices;
    for (int i = 0; i < segments_; ++i) {
        const float t = float(i) / float(segments_);
        vertices.push_back({-0.5f, t});
        vertices.push_back({0.5f, t});
    }
    vertices.push_back({0.0f, 1.0f});

    std::vector<uint16_t> indices;
    for (int i = 0; i < segments_ - 1; ++i) {
        const uint16_t v = uint16_t(i * 2);
        indices.insert(indices.end(), {v, uint16_t(v + 1), uint16_t(v + 2)});
        indices.insert(indices.end(), {uint16_t(v + 1), uint16_t(v + 3), uint16_t(v + 2)});
    }
    const uint16_t top = uint16_t((segments_ - 1) * 2);
    indices.insert(indices.end(), {top, uint16_t(top + 1), uint16_t(top + 2)});

    vertexCount_ = int(vertices.size());
    indexCount_ = int(indices.size());
    const auto vboBytes = GLsizeiptr(vertices.size() * sizeof(glm::vec2));
    const auto iboBytes = GLsizeiptr(indices.size() * sizeof(uint16_t));

    glDeleteBuffers(1, &vbo_);
    glDeleteBuffers(1, &ibo_);
    g_residentBytes -= meshBytes_;
    glCreateBuffers(1, &vbo_);
    glNamedBufferStorage(vbo_, vboBytes, vertices.data(), 0);
    glCreateBuffers(1, &ibo_);
    glNamedBufferStorage(ibo_, iboBytes, indices.data(), 0);
    meshBytes_ = uint64_t(vboBytes + iboBytes);
    g_residentBytes += meshBytes_;
    g_counters.bytesUploaded += meshBytes_;

    glVertexArrayVertexBuffer(vao_, 0, vbo_, 0, sizeof(glm::vec2));
    glVertexArrayElementBuffer(vao_, ibo_);
}

bool GrassField::regenerate(int count, float density, uint32_t seed) {
    fieldSize_ = std::sqrt(float(count) / density);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::vector<GrassInstance> instances(static_cast<size_t>(count));
    for (GrassInstance& blade : instances) {
        const float x = (u01(rng) - 0.5f) * fieldSize_;
        const float z = (u01(rng) - 0.5f) * fieldSize_;
        const float yaw = u01(rng) * 6.2831853f;
        const float height = 0.35f + 0.5f * u01(rng);
        const float width = 0.035f + 0.03f * u01(rng);
        const float bend = 0.1f + 0.35f * u01(rng);
        blade.posYaw = {x, 0.0f, z, yaw};
        blade.params = {height, width, bend, u01(rng)};
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
    glNamedBufferStorage(instanceBuffer_, GLsizeiptr(bytes), instances.data(), 0);
    if (glGetError() == GL_OUT_OF_MEMORY) {
        std::fprintf(stderr, "Out of GPU memory allocating %.0f MB of instances\n",
                     double(bytes) / (1024.0 * 1024.0));
        glDeleteBuffers(1, &instanceBuffer_);
        instanceBuffer_ = 0;
        return false;
    }
    instanceBytes_ = bytes;
    g_residentBytes += instanceBytes_;
    g_counters.bytesUploaded += instanceBytes_;
    count_ = count;
    return true;
}

void GrassField::draw(const SceneParams& scene) const {
    if (count_ == 0) return;
    glUseProgram(program_);
    glProgramUniformMatrix4fv(program_, locViewProj_, 1, GL_FALSE, &scene.viewProj[0][0]);
    glProgramUniform1f(program_, locTime_, scene.time);
    glProgramUniform2fv(program_, locWindDir_, 1, &scene.windDir[0]);
    glProgramUniform1f(program_, locWindStrength_, scene.windStrength);
    glProgramUniform3fv(program_, locSunDir_, 1, &scene.sunDir[0]);
    glProgramUniform3fv(program_, locCameraPos_, 1, &scene.cameraPos[0]);
    glProgramUniform3fv(program_, locFogColor_, 1, &scene.fogColor[0]);
    glProgramUniform1f(program_, locFogDensity_, scene.fogDensity);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, instanceBuffer_);
    glBindVertexArray(vao_);
    glDisable(GL_CULL_FACE); // blades are two-sided
    glDrawElementsInstanced(GL_TRIANGLES, indexCount_, GL_UNSIGNED_SHORT, nullptr, count_);

    g_counters.drawCalls += 1;
    g_counters.instances += uint64_t(count_);
    g_counters.triangles += uint64_t(count_) * uint64_t(trianglesPerBlade());
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
}
