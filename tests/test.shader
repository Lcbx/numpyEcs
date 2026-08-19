in vertexPosition: vec3f;
in vertexNormal: vec3f;
in vertexTexCoord: vec2f;

uniform matModel: mat4x4f;
uniform mvp: mat4x4f;


#if FEATURES.BIAS

const bias: f32 = {{ PARAMS.bias | default(0.0001) }};

#include "test_include.shader"

#endif

@position varying position: vec4f;
varying fragColor: vec4f;

out color: vec4f;


@vertex
fn vertex() {
	out.fragColor = vec4f(0.5, 0.1, 1.0, 1.0);
	var position = mvp * vec4f(in.vertexPosition, 1.0);

#if FEATURES.BIAS
	position.z *= plus_one(bias);
#endif

	out.position = position;
}

@fragment
fn fragment() {
	out.color = in.fragColor;
}