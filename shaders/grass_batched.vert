#version 460 core
// Pre-transformed world-space vertices (static or dynamic batch): no per-blade data at all.

layout(location = 0) in vec4 aPosT;     // world position at rest, t along the blade
layout(location = 1) in vec4 aNormal;   // packed 10:10:10:2 snorm
layout(location = 2) in vec3 aExtra;    // color variation, wind phase offset, wind weight

uniform mat4 uViewProj;
uniform float uTime;
uniform vec2 uWindDir;
uniform float uWindStrength;

out vec3 vWorldPos;
out vec3 vNormal;
out float vT;
out float vVariation;
flat out uint vDrawId;

void main() {
    // Static batches keep procedural wind (weight = h * t^2); dynamic batches bake it on the
    // CPU and store weight 0. The normal is not re-tilted by wind here, a deliberate shortcut.
    float phase = uTime * 1.7 + aExtra.y;
    float sway = (0.5 + 0.35 * sin(phase) + 0.15 * sin(phase * 2.3 + 1.0)) * uWindStrength;
    vec3 p = aPosT.xyz;
    p.xz += uWindDir * (sway * aExtra.z);

    vNormal = normalize(aNormal.xyz);
    vWorldPos = p;
    vT = aPosT.w;
    vVariation = aExtra.x;
    vDrawId = 0u;
    gl_Position = uViewProj * vec4(p, 1.0);
}
