import pytest

from shader_util import BetterShader
from raylib import InitWindow

# no way to init raylib without this afaik
InitWindow(1, 1, b"test")

shader1 = BetterShader('tests/test.shader')
shader2 = BetterShader('tests/test.shader',features={"BIAS"})
shader3 = BetterShader('tests/test.shader',features={"BIAS"}, params={"bias":0.01})

def test_valid():
	assert shader1.valid()
	assert shader2.valid()
	assert shader3.valid()

bias_decl = "const float bias = "
func_decl = "float plus_one(float bias)"
vertex_bias = "vertex.z *= plus_one(bias);"

def test_shader_variants():
	assert bias_decl not in shader1.vertex_glsl
	assert "vertex.z *=" not in shader1.vertex_glsl

	print(shader2.vertex_glsl)
	assert bias_decl + "0.0001" in shader2.vertex_glsl
	assert vertex_bias in shader2.vertex_glsl

	assert bias_decl + "0.01" in shader3.vertex_glsl
	assert vertex_bias in shader3.vertex_glsl

	for shader in (shader1, shader2, shader3):
		assert func_decl not in shader.fragment_glsl
		assert vertex_bias not in shader.fragment_glsl
