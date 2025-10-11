

in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

uniform mat4 matModel;
uniform mat4 mvp;

//const float bias = 0.0001;

varying vec4 fragColor;

void vertex(){
	fragColor = vertexColor;
	vec4 vertex = vec4(vertexPosition, 1.0);
	vertex = mvp*vertex;
	//vertex.z *= (1.0 + bias);
	//vertex.z += bias;
	gl_Position = vertex;
}


out vec4 finalColor;
void fragment() {
	finalColor = fragColor;
}

