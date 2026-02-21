import pytest

from shader_util import *

RenderContext.setup(highpower=False)

# will throw if invalid
shader1, pipeline = build_shader_program('tests/test.shader')
shader2, _ = build_shader_program('tests/test.shader',features={"BIAS"})
shader3, _ = build_shader_program('tests/test.shader',features={"BIAS"}, params={"bias":0.01})

pipeline.generate_uniform_buffer()

bias_decl = "const float bias = "
func_decl = "float plus_one(float bias)"
vertex_bias = "vertex.z *= plus_one(bias);"

def test_shader_variants():
	assert bias_decl not in shader1.vertex_glsl
	assert "vertex.z *=" not in shader1.vertex_glsl

	assert bias_decl + "0.0001" in shader2.vertex_glsl
	assert vertex_bias in shader2.vertex_glsl

	assert bias_decl + "0.01" in shader3.vertex_glsl
	assert vertex_bias in shader3.vertex_glsl

	for shader in (shader1, shader2, shader3):
		assert func_decl not in shader.fragment_glsl
		assert vertex_bias not in shader.fragment_glsl