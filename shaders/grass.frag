#version 460 core

in vec3 vWorldPos;
in vec3 vNormal;
in float vT;
in float vVariation;

uniform vec3 uSunDir;
uniform vec3 uCameraPos;
uniform vec3 uFogColor;
uniform float uFogDensity;

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

    float fog = 1.0 - exp(-dist * uFogDensity);
    oColor = vec4(mix(color, uFogColor, fog), 1.0);
}
