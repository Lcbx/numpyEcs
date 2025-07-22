

in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

uniform mat4 matModel;
uniform mat4 mvp;

//const float bias = 0.00001;

varying vec4 fragColor;

void vertex(){
	fragColor = vertexColor;
	vec4 vertex = vec4(vertexPosition, 1.0);
	//vertex.z *= (1.0 + bias);
	gl_Position = mvp*vertex;
}


out vec4 finalColor;
void fragment() {
	finalColor = fragColor;
}
