

in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

uniform mat4 matModel;
uniform mat4 mvp;

uniform vec3 bias; // - dir_to_camera * bias_value

varying vec2 fragNormal;

void vertex(){
    vec4 vertex = vec4(vertexPosition, 1.0);
    fragNormal = normalize(mat3(matModel) * vertexNormal).xy;
    vertex.xyz += bias;
    gl_Position = mvp*vertex;
}


out vec4 finalColor;
void fragment() {
    finalColor = vec4(fragNormal, gl_FragCoord.z, 1);
}
