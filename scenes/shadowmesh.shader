

in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

uniform mat4 matModel;
uniform mat4 mvp;

uniform vec3 bias; // lightDir * bias_value

varying vec4 fragColor;

void vertex(){
    fragColor = vertexColor;
    vec4 vertex = vec4(vertexPosition, 1.0);
    //vec3 fragNormal = normalize(mat3(matModel) * vertexNormal);
    //vertex.xyz -= fragNormal * bias;
    vertex.xyz += bias;
    gl_Position = mvp*vertex;
}


out vec4 finalColor;
void fragment() {
    //float d = gl_FragCoord.z;
    //const float layer_size = 10.0;
    //float meshId = roundEven( d * layer_size) / layer_size;
    //finalColor = vec4(meshId, fragColor.gba);
    finalColor = fragColor;
}
