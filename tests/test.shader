
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

uniform mat4 matModel;
uniform mat4 mvp;

#if FEATURES.BIAS

const float bias = {{ PARAMS.bias |default(0.0001) }};

#include "test_include.shader"

#endif

varying vec4 fragColor;

void vertex(){
	fragColor = vertexColor;
	vec4 vertex = vec4(vertexPosition, 1.0);
	vertex = mvp*vertex;
#if FEATURES.BIAS
	vertex.z *= plus_one(bias);
	//vertex.z += bias;
#endif
	gl_Position = vertex;
}


out vec4 finalColor;
void fragment() {
	finalColor = fragColor;
}