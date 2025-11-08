
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

//uniform mat4 matModel;
uniform mat4 mvp;

//varying vec3 fragNormal;

void vertex(){
	//fragNormal = normalize(mat3(matModel) * vertexNormal).xyz;
	vec4 vertex = vec4(vertexPosition, 1.0);
	gl_Position = mvp*vertex;
}

out vec4 outAO;
void fragment() {
	// deducing the normals from the depth values works good enough
	// so no need for this :
	//outAO = vec4(fragNormal * 0.5 + 0.5, 1);
}
