import pytest

from shader_util import *
from pyrr import Matrix44

RenderContext.setup(highpower=False)

# will throw if invalid
source1, shader1 = build_shader_program('tests/test.shader')
source2, shader2 = build_shader_program('tests/test.shader',features={"BIAS"})
source3, shader3 = build_shader_program('tests/test.shader',features={"BIAS"}, params={"bias":0.01})

bias_decl = "const float bias = "
func_decl = "float plus_one(float bias)"
vertex_bias = "vertex.z *= plus_one(bias);"

def test_shader_variants():
	assert bias_decl not in source1.vertex_glsl
	assert "vertex.z *=" not in source1.vertex_glsl

	assert bias_decl + "0.0001" in source2.vertex_glsl
	assert vertex_bias in source2.vertex_glsl

	assert bias_decl + "0.01" in source3.vertex_glsl
	assert vertex_bias in source3.vertex_glsl

	for shader in (source1, source2, source3):
		assert func_decl not in shader.fragment_glsl
		assert vertex_bias not in shader.fragment_glsl

def test_uniform_buffer():
	u = shader1.UniformBuffer()
	u.uniforms['mvp'] = Matrix44.from_scale([1.0, 1.0, 1.0], dtype=np.float32)
	u.write_uniforms()