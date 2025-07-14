

in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

uniform mat4 matModel;
uniform mat4 mvp;


varying vec2 fragNormal;

void vertex(){
    fragNormal = normalize(mat3(matModel) * vertexNormal).xy;
    vec4 vertex = vec4(vertexPosition, 1.0);
    gl_Position = mvp*vertex;
}

//layout(location = 1) out vec4 outAO;
out vec4 finalColor;
out vec4 outAO;
void fragment() {
    finalColor = vec4(1);
    outAO = vec4(fragNormal, gl_FragCoord.z, 1);
}
