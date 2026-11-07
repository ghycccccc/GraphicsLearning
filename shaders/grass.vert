#version 460 core
// Variants (exactly one source must be defined by the host):
//   INSTANCE_FROM_UNIFORMS    one draw per blade, CPU record set as uniforms before each draw
//   INSTANCE_FROM_ATTRIBUTES  instanced draw, divisor-1 vertex attributes (the VAO formats
//                             decode either InstanceLayout, so no PACKED_LAYOUT variant)
//   INSTANCE_FROM_SSBO        instanced draw, blade data fetched by gl_InstanceID
// Modifiers:
//   PACKED_LAYOUT  SSBO data is the 16 B packed encoding (see InstanceLayout in grass.h)
//   DYNAMIC_BEND   current bend comes from a per-frame buffer (CPU upload or compute sim)
//                  instead of the static rest bend
//   VISIBLE_LIST   (SSBO only) instances are entries of a culled id list, see below

layout(location = 0) in vec2 aBlade; // x across the blade in [-0.5, 0.5], y = t in [0, 1]

// ===== MUST READ (M4 #7): how a culled draw finds its blades ==============================
// Without culling, instance i of the draw is blade i. With culling, a list of visible blade
// ids sits in between: instance i of draw d is blade visibleIds[baseInstance(d) + i].
//   CPU per-blade culling: one draw, list written by the CPU, baseInstance 0.
//   CPU chunk culling:     one indirect command per visible chunk; baseInstance = where the
//                          chunk's range starts in the chunk-sorted id list.
//   GPU culling:           one indirect command per LOD; baseInstance = start of that LOD's
//                          region, instanceCount written by the compute shader.
// gl_InstanceID does NOT include baseInstance, hence the explicit gl_BaseInstance (GL 4.6).
// Everything per blade (static data, bends) is then indexed by the blade id, so culling works
// with every animation mode. Divisor attributes would apply baseInstance automatically but
// cannot follow an id list, which is why culling is SSBO-only.
#ifdef VISIBLE_LIST
layout(std430, binding = 2) readonly buffer VisibleIds { uint visibleIds[]; };
#  define BLADE_ID visibleIds[gl_BaseInstance + gl_InstanceID]
#else
#  define BLADE_ID uint(gl_InstanceID)
#endif
// ==========================================================================================

#if defined(INSTANCE_FROM_UNIFORMS)
uniform vec4 uInstanceA; // root.x, root.z, yaw (radians), color variation
uniform vec4 uInstanceB; // height, width, bend.x, bend.z
#elif defined(INSTANCE_FROM_ATTRIBUTES)
layout(location = 1) in vec2 aRoot;
layout(location = 2) in float aYaw01;
layout(location = 3) in float aVariation;
layout(location = 4) in float aRestBend01;
layout(location = 5) in vec2 aSize;
#  ifdef DYNAMIC_BEND
layout(location = 6) in vec2 aBend;
#  endif
#else
#  ifdef PACKED_LAYOUT
layout(std430, binding = 0) readonly buffer Statics { uvec4 statics[]; };
layout(std430, binding = 1) readonly buffer Bends { uint bends[]; };
#  else
struct StaticBlade { vec4 a; vec4 b; };
layout(std430, binding = 0) readonly buffer Statics { StaticBlade statics[]; };
layout(std430, binding = 1) readonly buffer Bends { vec2 bends[]; };
#  endif
#endif

uniform mat4 uViewProj;
uniform float uTime;
uniform vec2 uWindDir;
uniform float uWindStrength;
uniform bool uProceduralWind; // false when the bend already includes wind

out vec3 vWorldPos;
out vec3 vNormal;
out float vT;
out float vVariation;
flat out uint vDrawId; // which command of a multi-draw (LOD or chunk), for the debug view

struct Blade {
    vec2 root;
    float yaw;
    float variation;
    float height;
    float width;
    vec2 bend; // world-space tip offset as a fraction of height
};

// Must match windSway() in grass.cpp and grass_sim.comp.
float windSway(vec2 rootXZ) {
    float phase = uTime * 1.7 + dot(rootXZ, vec2(0.13, 0.09));
    return (0.5 + 0.35 * sin(phase) + 0.15 * sin(phase * 2.3 + 1.0)) * uWindStrength;
}

Blade fetchBlade() {
    Blade b;
#if defined(INSTANCE_FROM_UNIFORMS)
    b.root = uInstanceA.xy;
    b.yaw = uInstanceA.z;
    b.variation = uInstanceA.w;
    b.height = uInstanceB.x;
    b.width = uInstanceB.y;
    b.bend = uInstanceB.zw;
#else
    uint id = BLADE_ID;
    float yaw01, restBend01;
    vec2 size;
#  if defined(INSTANCE_FROM_ATTRIBUTES)
    b.root = aRoot;
    yaw01 = aYaw01;
    b.variation = aVariation;
    restBend01 = aRestBend01;
    size = aSize;
#  elif defined(PACKED_LAYOUT)
    uvec4 s = statics[id];
    b.root = uintBitsToFloat(s.xy);
    yaw01 = float(s.z & 0xFFFFu) / 65535.0;
    b.variation = float((s.z >> 16) & 0xFFu) / 255.0;
    restBend01 = float(s.z >> 24) / 255.0;
    size = unpackHalf2x16(s.w);
#  else
    StaticBlade s = statics[id];
    b.root = s.a.xy;
    yaw01 = s.a.z;
    b.variation = s.a.w;
    size = s.b.xy;
    restBend01 = s.b.z;
#  endif
    b.yaw = yaw01 * 6.2831853;
    b.height = size.x;
    b.width = size.y;
#  if !defined(DYNAMIC_BEND)
    b.bend = vec2(sin(b.yaw), cos(b.yaw)) * (0.5 * restBend01);
#  elif defined(INSTANCE_FROM_ATTRIBUTES)
    b.bend = aBend;
#  elif defined(PACKED_LAYOUT)
    b.bend = unpackHalf2x16(bends[id]);
#  else
    b.bend = bends[id];
#  endif
#endif
    return b;
}

void main() {
    Blade blade = fetchBlade();
    vec3 root = vec3(blade.root.x, 0.0, blade.root.y);
    vec3 right = vec3(cos(blade.yaw), 0.0, -sin(blade.yaw));
    float h = blade.height, w = blade.width;
    vec2 bend = blade.bend;
    if (uProceduralWind) bend += uWindDir * windSway(blade.root);
    vec3 bend3 = vec3(bend.x, 0.0, bend.y);
    float t = aBlade.y;

    vec3 p = root + right * (aBlade.x * w * (1.0 - t)) + vec3(0.0, t * h, 0.0) + bend3 * (h * t * t);
    vec3 dPdt = vec3(0.0, h, 0.0) + bend3 * (2.0 * h * t);

    vNormal = normalize(cross(right, dPdt));
    vWorldPos = p;
    vT = t;
    vVariation = blade.variation;
    vDrawId = uint(gl_DrawID);
    gl_Position = uViewProj * vec4(p, 1.0);
}
