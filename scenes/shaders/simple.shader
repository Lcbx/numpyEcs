
in vec3 aPos;
in vec3 aNormal;
in vec2 aUV;

// from instance buffer
// ik, this is messy. but no other way to pass a mat4 with glsl
in vec4 uModel_0;
in vec4 uModel_1;
in vec4 uModel_2;
in vec4 uModel_3;
in vec4 uTint;

uniform mat4 uView;
uniform mat4 uProj;
uniform vec3 uLightDir;

varying vec3 vNormalWS;
varying vec2 vUV;
flat varying vec4 tint;


void vertex() {
    mat4 uModel = mat4(uModel_0, uModel_1, uModel_2, uModel_3);
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vNormalWS = mat3(uModel) * aNormal;
    vUV = aUV;
    gl_Position = uProj * uView * worldPos;
    tint = uTint;
}

out vec4 FragColor;

void fragment() {
    vec3 N = normalize(vNormalWS);
    float NdotL = max(dot(N, uLightDir.xyz), 0.0);
    vec3 color = tint.rgb * (0.2 + 0.8 * NdotL);
    FragColor = vec4(color, tint.a);
}
