#include "debug_lines.h"

#include "counters.h"
#include "gl_util.h"

namespace {
constexpr int kVertexCount = 24; // 12 edges
}

bool DebugLines::init() {
    program_ = loadProgram({{GL_VERTEX_SHADER, "lines.vert"}, {GL_FRAGMENT_SHADER, "lines.frag"}});
    if (!program_) return false;
    locViewProj_ = glGetUniformLocation(program_, "uViewProj");
    locColor_ = glGetUniformLocation(program_, "uColor");
    glCreateBuffers(1, &vbo_);
    glNamedBufferStorage(vbo_, kVertexCount * sizeof(glm::vec3), nullptr, GL_DYNAMIC_STORAGE_BIT);
    glCreateVertexArrays(1, &vao_);
    glVertexArrayVertexBuffer(vao_, 0, vbo_, 0, sizeof(glm::vec3));
    glEnableVertexArrayAttrib(vao_, 0);
    glVertexArrayAttribFormat(vao_, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(vao_, 0, 0);
    return true;
}

void DebugLines::destroy() {
    glDeleteProgram(program_);
    glDeleteVertexArrays(1, &vao_);
    glDeleteBuffers(1, &vbo_);
}

void DebugLines::drawFrustum(const glm::mat4& frustumViewProj, const glm::mat4& viewProj, const glm::vec3& color) {
    // Corners of the clip-space cube, mapped back to world space.
    const glm::mat4 inverse = glm::inverse(frustumViewProj);
    glm::vec3 corners[8];
    for (int i = 0; i < 8; ++i) {
        const glm::vec4 ndc(i & 1 ? 1.0f : -1.0f, i & 2 ? 1.0f : -1.0f, i & 4 ? 1.0f : -1.0f, 1.0f);
        const glm::vec4 world = inverse * ndc;
        corners[i] = glm::vec3(world) / world.w;
    }
    const int edges[kVertexCount] = {0, 1, 2, 3, 0, 2, 1, 3,  // near rectangle
                                     4, 5, 6, 7, 4, 6, 5, 7,  // far rectangle
                                     0, 4, 1, 5, 2, 6, 3, 7}; // sides
    glm::vec3 vertices[kVertexCount];
    for (int i = 0; i < kVertexCount; ++i) vertices[i] = corners[edges[i]];
    glNamedBufferSubData(vbo_, 0, sizeof(vertices), vertices);

    glUseProgram(program_);
    glProgramUniformMatrix4fv(program_, locViewProj_, 1, GL_FALSE, &viewProj[0][0]);
    glProgramUniform3fv(program_, locColor_, 1, &color[0]);
    glBindVertexArray(vao_);
    glDrawArrays(GL_LINES, 0, kVertexCount);
    g_counters.drawCalls += 1;
}
