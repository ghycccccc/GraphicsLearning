#version 460 core
// Variants (one must be defined by the host):
//   INSTANCE_FROM_SSBO        instanced draw, per-blade data indexed by gl_InstanceID
//   INSTANCE_FROM_ATTRIBUTES  instanced draw, per-blade data as divisor-1 vertex attributes
//   INSTANCE_FROM_UNIFORMS    one draw per blade, data set as uniforms before each draw

layout(location = 0) in vec2 aBlade; // x across the blade in [-0.5, 0.5], y = t in [0, 1]

// Per-blade data (32 bytes):
//   a = (root.x, root.z, yaw, color variation)
//   b = (height, width, bend.x, bend.z)  bend = world-space tip offset as a fraction of height
#if defined(INSTANCE_FROM_SSBO)
struct Instance { vec4 a; vec4 b; };
layout(std430, binding = 0) readonly buffer Instances { Instance instances[]; };
#elif defined(INSTANCE_FROM_ATTRIBUTES)
layout(location = 1) in vec4 aInstanceA;
layout(location = 2) in vec4 aInstanceB;
#else
uniform vec4 uInstanceA;
uniform vec4 uInstanceB;
#endif

uniform mat4 uViewProj;
uniform float uTime;
uniform vec2 uWindDir;
uniform float uWindStrength;
uniform bool uProceduralWind; // false when the CPU already folded wind into bend

out vec3 vWorldPos;
out vec3 vNormal;
out float vT;
out float vVariation;

// Must match windSway() in grass.cpp.
float windSway(vec2 rootXZ) {
    float phase = uTime * 1.7 + dot(rootXZ, vec2(0.13, 0.09));
    return (0.5 + 0.35 * sin(phase) + 0.15 * sin(phase * 2.3 + 1.0)) * uWindStrength;
}

void main() {
#if defined(INSTANCE_FROM_SSBO)
    vec4 ia = instances[gl_InstanceID].a;
    vec4 ib = instances[gl_InstanceID].b;
#elif defined(INSTANCE_FROM_ATTRIBUTES)
    vec4 ia = aInstanceA;
    vec4 ib = aInstanceB;
#else
    vec4 ia = uInstanceA;
    vec4 ib = uInstanceB;
#endif
    vec3 root = vec3(ia.x, 0.0, ia.y);
    vec3 right = vec3(cos(ia.z), 0.0, -sin(ia.z));
    float h = ib.x, w = ib.y;
    vec2 bend = ib.zw;
    if (uProceduralWind) bend += uWindDir * windSway(root.xz);
    vec3 bend3 = vec3(bend.x, 0.0, bend.y);
    float t = aBlade.y;

    vec3 p = root + right * (aBlade.x * w * (1.0 - t)) + vec3(0.0, t * h, 0.0) + bend3 * (h * t * t);
    vec3 dPdt = vec3(0.0, h, 0.0) + bend3 * (2.0 * h * t);

    vNormal = normalize(cross(right, dPdt));
    vWorldPos = p;
    vT = t;
    vVariation = ia.w;
    gl_Position = uViewProj * vec4(p, 1.0);
}
