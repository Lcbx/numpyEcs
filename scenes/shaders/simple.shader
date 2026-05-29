
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

varying vec3 vNormalWS;
varying vec2 vUV;
flat varying uint tint;

vec3 quat_rotate(vec4 q, vec3 v) {
    vec3 t = cross(q.xyz, v) * 2.0;
    return v + q.w * t + cross(q.xyz, t);
}


void vertex() {
    vec3 worldPos = iPosition + quat_rotate(iRotation, vPos * iScale);
    vec3 normal = vNormal;
    if(iScale!=vec3(1.0,1.0,10)) normal /= iScale;
    vec3 world_normal = quat_rotate(iRotation, normal);

    gl_Position = uProj * uView * vec4(worldPos, 1.0);
    vNormalWS = world_normal;
    vUV = vUV;
    tint = iTint;
}

// TODO: move color utilities into a shader_include file

float srgb_to_linear_channel(float c) {
    if( c <= 0.04045 ){
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

vec4 unpack_rgba8_srgb(uint c) {
    vec4 rgba = vec4(
        float( c         & 255u),
        float((c >> 8u)  & 255u),
        float((c >> 16u) & 255u),
        float((c >> 24u) & 255u)
    ) / 255.0;

    return vec4(
        srgb_to_linear_channel(rgba.r),
        srgb_to_linear_channel(rgba.g),
        srgb_to_linear_channel(rgba.b),
        rgba.a
    );
}


out vec4 FragColor;

void fragment() {
    vec3 N = normalize(vNormalWS);
    float NdotL = max(dot(N, uLightDir.xyz), 0.0);
    vec4 color = unpack_rgba8_srgb(tint);
    FragColor = vec4(color.rgb  * (0.2 + 0.8 * NdotL), color.a);
}
