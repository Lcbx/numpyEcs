
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

uniform mat4 mvp;

void vertex(){
	vec4 vertex = vec4(vertexPosition, 1.0);
	vertex = mvp*vertex;
	gl_Position = vertex;
}

void fragment() {
	// pass
}
