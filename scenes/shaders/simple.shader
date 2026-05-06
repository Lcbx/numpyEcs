
in vec3 aPos;
in vec3 aNormal;
in vec2 aUV;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform vec4 uLightDir; // TODO: properly pack vec3
uniform vec4 uTint;

varying vec3 vNormalWS;
varying vec2 vUV;


void vertex() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vNormalWS = mat3(uModel) * aNormal;
    vUV = aUV;
    gl_Position = uProj * uView * worldPos;
}

out vec4 FragColor;

void fragment() {
    vec3 N = normalize(vNormalWS);
    float NdotL = max(dot(N, uLightDir.xyz), 0.0);
    //vec3 color = vec3(0.82, 0.71, 0.55) * (0.2 + 0.8 * NdotL);
    vec3 color = uTint.rgb * (0.2 + 0.8 * NdotL);
    //FragColor = vec4(color, uTint.a);
    FragColor = vec4(color, 1.0);
}
