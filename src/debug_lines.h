#pragma once
#include <GL/glew.h>
#include <glm/glm.hpp>

// Draws a camera frustum as 12 lines, e.g. the frozen culling camera.
class DebugLines {
public:
    bool init();
    void destroy();
    // Edges of the frustum of `frustumViewProj`, seen through `viewProj`.
    void drawFrustum(const glm::mat4& frustumViewProj, const glm::mat4& viewProj, const glm::vec3& color);

private:
    GLuint program_ = 0, vao_ = 0, vbo_ = 0;
    GLint locViewProj_ = -1, locColor_ = -1;
};
