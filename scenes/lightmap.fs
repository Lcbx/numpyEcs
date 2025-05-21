#version 330

in vec2 fragTexCoord;
in vec3 fragNormal;
in vec4 fragColor;

in vec4 fragShadowClipSpace;

uniform sampler2D texture0; // diffuse texture keyword

uniform vec3 lightDir;
uniform mat4 lightVP;
uniform sampler2D texture_shadowmap;

out vec4 finalColor;


bool between(vec2 v, vec2 bottomLeft, vec2 topRight) {
    vec2 s = step(bottomLeft, v) - step(topRight, v);
    return bool(s.x * s.y);   
}

float random(vec2 co) {
    return fract(dot(co, vec2(3,8)) * dot(co.yx, vec2(7,5)) * 0.03);
}

void main()
{
    finalColor = fragColor * texture( texture0, fragTexCoord );
	
	vec3 projCoords = fragShadowClipSpace.xyz / fragShadowClipSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    vec2 shadowTexCoords = projCoords.xy;
	
	if( !between(shadowTexCoords, vec2(0), vec2(1) ) ) return;
	float fragmentDepth = projCoords.z;
	float shadowDepth = texture(texture_shadowmap, shadowTexCoords).r;
	
	// Calculate simple bias based on angle between normal and light
	float NDotL = dot(fragNormal, lightDir);
	float bias = 0.001 * (1.5 - NDotL);
	
	float shadow = fragmentDepth > shadowDepth + bias ? 0.5 : 1.0;
	
	finalColor = vec4( vec3(shadow) * finalColor.rgb, 1);
}