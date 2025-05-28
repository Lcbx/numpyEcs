#version 330

in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

uniform mat4 matModel;
uniform mat4 mvp;

uniform vec3 lightDir;

out vec4 fragColor;

void main(){
    fragColor = vertexColor;
    vec4 vertex = vec4(vertexPosition, 1);
    vec3 fragNormal = normalize(mat3(matModel) * vertexNormal);
    vertex.xyz -= fragNormal * 0.1;
    //vertex.xyz -= lightDir * 0.1;
    vertex = mvp*vertex;
    gl_Position = vertex;
}