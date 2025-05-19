#version 330

in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

uniform mat4 matModel;
uniform mat4 mvp;

uniform mat4 lightVP;

out vec2 fragTexCoord;
out vec4 fragColor;
out vec4 fragShadowClipSpace;

void main(){
    fragTexCoord = vertexTexCoord;
    fragColor = vertexColor;
	
	vec4 vertex = vec4(vertexPosition, 1.0);
    gl_Position = mvp*vertex;
	
    fragShadowClipSpace = lightVP*matModel*vertex;
}