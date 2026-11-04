#version 460 core

layout(location = 0) in vec2 aBlade; // x across the blade in [-0.5, 0.5], y = t in [0, 1]

struct Instance {
    vec4 posYaw; // xyz root, w yaw
    vec4 params; // x height, y width, z bend, w color variation
};
layout(std430, binding = 0) readonly buffer Instances { Instance instances[]; };

uniform mat4 uViewProj;
uniform float uTime;
uniform vec2 uWindDir;
uniform float uWindStrength;

out vec3 vWorldPos;
out vec3 vNormal;
out float vT;
out float vVariation;

void main() {
    Instance inst = instances[gl_InstanceID];
    vec3 root = inst.posYaw.xyz;
    float c = cos(inst.posYaw.w), s = sin(inst.posYaw.w);
    vec3 right = vec3(c, 0.0, -s);
    vec3 fwd = vec3(s, 0.0, c);
    float h = inst.params.x, w = inst.params.y, bend = inst.params.z;
    float t = aBlade.y;

    // Procedural wind: a travelling wave plus a gust term, stronger towards the tip.
    float phase = uTime * 1.7 + dot(root.xz, vec2(0.13, 0.09));
    float sway = (0.5 + 0.35 * sin(phase) + 0.15 * sin(phase * 2.3 + 1.0)) * uWindStrength;
    vec3 wind = vec3(uWindDir.x, 0.0, uWindDir.y);

    vec3 p = root
           + right * (aBlade.x * w * (1.0 - t))
           + vec3(0.0, t * h, 0.0)
           + fwd * (bend * h * t * t)
           + wind * (sway * h * t * t);

    vec3 dPdt = vec3(0.0, h, 0.0) + fwd * (2.0 * bend * h * t) + wind * (2.0 * sway * h * t);
    vNormal = normalize(cross(right, dPdt));
    vWorldPos = p;
    vT = t;
    vVariation = inst.params.w;
    gl_Position = uViewProj * vec4(p, 1.0);
}
