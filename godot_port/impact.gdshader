shader_type spatial;
render_mode depth_prepass_alpha,cull_disabled;

#define toon

uniform vec3 color : source_color;
uniform float roughness : hint_range(0,1);
uniform float specular : hint_range(0,1);

uniform float displacement_str;
uniform float displacement_speed;
uniform float displacement_dimming;

uniform float erosion_scale : hint_range(.1,10);
uniform sampler2D erosion_curve : repeat_disable;
uniform float mist_transparency;

uniform float shadow_coverage;
uniform float shadow_max;

uniform sampler2D noise : filter_linear_mipmap;

instance uniform float instance_rand;
varying flat float instance_id;
varying flat float lifetime_percent;
varying float dimming;

void vertex() {
	// from ParticleProcessMaterial
	lifetime_percent = INSTANCE_CUSTOM.y;
	instance_id = float(INSTANCE_ID) * .1;
	
	float displacement = texture(noise, UV + vec2(0.,instance_id + instance_rand + TIME * displacement_speed)).r;
	VERTEX += (displacement * displacement_str - displacement_str) * NORMAL;
	dimming = displacement;
}

void fragment() {
	ALBEDO = color * mix(1., dimming, displacement_dimming);
	//ALBEDO = color;
	//ALBEDO = vec3(instance_rand);
	float erosion = texture(noise, UV * erosion_scale + vec2(instance_rand + instance_id)).r;
	float erosion_strength = texture(erosion_curve, vec2(lifetime_percent)).r;
	float mixed = smoothstep(0., erosion * 2., erosion_strength);
	float applied_erosion = max(round(mixed), mixed * mist_transparency);
	ALPHA *= applied_erosion;
	ROUGHNESS = roughness;
	SPECULAR = specular;
}

#ifdef toon
void light() {
	// Diffuse lighting.
	float NdotL = (dot(NORMAL, LIGHT) + shadow_coverage ) * ATTENUATION;
	NdotL = clamp(NdotL, 0., 1.);
	vec3 diffuse = LIGHT_COLOR /PI;
	float t = NdotL;
	t = max(t, shadow_max);
	//t -= mod(NdotL, .6) * .2;
	diffuse *= t;
	DIFFUSE_LIGHT += diffuse;
}
#endif