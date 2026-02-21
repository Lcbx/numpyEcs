
in vec3 vertexPosition;
in vec3 vertexNormal;
in vec2 vertexTexCoord;

uniform mat4 matModel;
uniform mat4 mvp;

// uniform array parsing test
uniform vec3[5] test;
uniform vec3[5][ 3] test2;

#if FEATURES.BIAS

const float bias = {{ PARAMS.bias |default(0.0001) }};

// contains plus_one definition
#include "test_include.shader"

#endif

varying vec4 fragColor;

void vertex(){
	fragColor = vec4(0.5, 0.1, 1.0, 1.0);
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