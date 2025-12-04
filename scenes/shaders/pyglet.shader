
in vec3 aPos;
in vec3 aNormal;
in vec2 aUV;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform vec3 uLightDir;

varying vec3 vNormalWS;
varying vec3 vLightDirWS;
varying vec2 vUV;

out vec4 FragColor;

void vertex() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vNormalWS = mat3(uModel) * aNormal;
    vLightDirWS = normalize(uLightDir);
    vUV = aUV;
    gl_Position = uProj * uView * worldPos;
}

void fragment() {
    vec3 N = normalize(vNormalWS);
    float NdotL = max(dot(N, vLightDirWS), 0.0);
    vec3 base = vec3(0.82, 0.71, 0.55); // BEIGE-ish
    vec3 color = base * (0.2 + 0.8 * NdotL);
    FragColor = vec4(color, 1.0);
}
