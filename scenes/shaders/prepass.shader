
in vec3 vPos;
in vec3 vNormal;
in vec2 vUV;

// from instance buffer
in vec3 iPosition;
in vec4 iRotation; // quaternion
in vec3 iScale;    // last float16 is unused
in uint iTint;

uniform mat4 uView;
uniform mat4 uProj;
uniform vec3 uLightDir;

vec3 quat_rotate(vec4 q, vec3 v) {
    vec3 t = cross(q.xyz, v) * 2.0;
    return v + q.w * t + cross(q.xyz, t);
}


void vertex(){
    vec3 worldPos = iPosition + quat_rotate(iRotation, vPos * iScale);
    gl_Position = uProj * uView * vec4(worldPos, 1.0);
}

void fragment() {
	// pass
}
