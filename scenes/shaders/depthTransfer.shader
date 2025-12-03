
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

varying vec2 fragTexCoord;

uniform sampler2D texture0; // depth

//out vec4 finalColor;

void vertex(){
	fragTexCoord = vertexTexCoord;
}

void fragment() {
	vec2 toUv = 1.0 / textureSize(texture0, 0);
	vec2 uv = fragTexCoord;

	float depth =      texture(texture0, uv + vec2(-1,-1) * toUv).x;
	depth = max(depth, texture(texture0, uv + vec2( 0,-1) * toUv).x );
	depth = max(depth, texture(texture0, uv + vec2( 1,-1) * toUv).x );
	depth = max(depth, texture(texture0, uv + vec2(-1, 1) * toUv).x );
	depth = max(depth, texture(texture0, uv                     ).x );
	depth = max(depth, texture(texture0, uv + vec2( 1, 0) * toUv).x );
	depth = max(depth, texture(texture0, uv + vec2( 0, 1) * toUv).x );
	depth = max(depth, texture(texture0, uv + vec2( 1, 1) * toUv).x );

	//ivec2 uv = ivec2(gl_FragCoord.xy);
	//float depth =
	//depth =            texelFetch(texture0, uv + ivec2(-1,-1), 0).x;
	//depth = max(depth, texelFetch(texture0, uv + ivec2( 0,-1), 0).x );
	//depth = max(depth, texelFetch(texture0, uv + ivec2( 1,-1), 0).x );
	//depth = max(depth, texelFetch(texture0, uv + ivec2(-1, 1), 0).x );
	//depth = max(depth, texelFetch(texture0, uv               , 0).x );
	//depth = max(depth, texelFetch(texture0, uv + ivec2( 1, 0), 0).x );
	//depth = max(depth, texelFetch(texture0, uv + ivec2( 0, 1), 0).x );
	//depth = max(depth, texelFetch(texture0, uv + ivec2( 1, 1), 0).x );;

	//finalColor = vec4(vec3(depth), 1);
	gl_FragDepth = depth;
}
