#version 460 core

in vec3 vWorldPos;
in vec3 vNormal;
in float vT;
in float vVariation;
flat in uint vDrawId;

uniform vec3 uSunDir;
uniform vec3 uCameraPos;
uniform vec3 uFogColor;
uniform float uFogDensity;
uniform bool uColorByDraw; // debug: tint by gl_DrawID (LOD with GPU culling, chunk with CPU chunks)

out vec4 oColor;

void main() {
    vec3 n = normalize(vNormal);
    if (!gl_FrontFacing) n = -n;

    vec3 base = mix(vec3(0.04, 0.14, 0.02), vec3(0.33, 0.52, 0.10), vT);
    base = mix(base, base * vec3(1.3, 1.1, 0.55), vVariation * 0.6);

    vec3 toCamera = uCameraPos - vWorldPos;
    float dist = length(toCamera);
    float diffuse = max(dot(n, uSunDir), 0.0);
    float ao = mix(0.35, 1.0, vT);
    float translucency = pow(max(dot(-toCamera / dist, uSunDir), 0.0), 4.0) * vT * 0.4;
    vec3 color = base * ((0.35 + 0.75 * diffuse) * ao + translucency);
    if (uColorByDraw) {
        // Golden-ratio hue steps: neighbouring draw ids get clearly different colors.
        vec3 tint = 0.5 + 0.5 * cos(6.2831853 * (fract(float(vDrawId) * 0.618034) + vec3(0.0, 0.33, 0.67)));
        color = tint * (0.3 + 0.7 * vT);
    }

    float fog = (1.0 - exp(-dist * uFogDensity)) * (uColorByDraw ? 0.25 : 1.0); // keep far LODs readable
    oColor = vec4(mix(color, uFogColor, fog), 1.0);
}
