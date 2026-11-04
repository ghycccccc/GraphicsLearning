#version 460 core

uniform mat4 uViewProj;
uniform float uHalfSize;

out vec3 vWorldPos;

void main() {
    vec2 corner = vec2(gl_VertexID & 1, gl_VertexID >> 1) * 2.0 - 1.0;
    vWorldPos = vec3(corner.x, 0.0, corner.y) * uHalfSize;
    gl_Position = uViewProj * vec4(vWorldPos, 1.0);
}
