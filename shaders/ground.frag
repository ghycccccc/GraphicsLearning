#version 460 core

in vec3 vWorldPos;

uniform vec3 uSunDir;
uniform vec3 uCameraPos;
uniform vec3 uFogColor;
uniform float uFogDensity;

out vec4 oColor;

void main() {
    vec3 base = vec3(0.09, 0.11, 0.04);
    vec3 color = base * (0.4 + 0.6 * max(uSunDir.y, 0.0));
    float fog = 1.0 - exp(-length(uCameraPos - vWorldPos) * uFogDensity);
    oColor = vec4(mix(color, uFogColor, fog), 1.0);
}
