import raylib as rl
from pyray import Shader, Vector2, Vector3, Vector4, \
	Material, Texture, RenderTexture, \
	ffi, rl_load_texture

import re
from typing import Any, Sequence

""" # in rlgl.h

void SetShaderValue(Shader shader, int locIndex, const void *value, int uniformType);			   // Set shader uniform value
void SetShaderValueV(Shader shader, int locIndex, const void *value, int uniformType, int count);   // Set shader uniform value vector

# uniform types :
	case RL_SHADER_UNIFORM_FLOAT: glUniform1fv(locIndex, count, (float *)value); break;
	case RL_SHADER_UNIFORM_VEC2: glUniform2fv(locIndex, count, (float *)value); break;
	case RL_SHADER_UNIFORM_VEC3: glUniform3fv(locIndex, count, (float *)value); break;
	case RL_SHADER_UNIFORM_VEC4: glUniform4fv(locIndex, count, (float *)value); break;
	case RL_SHADER_UNIFORM_INT:  glUniform1iv(locIndex, count, (int *)value); break;
	case RL_SHADER_UNIFORM_IVEC2: glUniform2iv(locIndex, count, (int *)value); break;
	case RL_SHADER_UNIFORM_IVEC3: glUniform3iv(locIndex, count, (int *)value); break;
	case RL_SHADER_UNIFORM_IVEC4: glUniform4iv(locIndex, count, (int *)value); break;
#if !defined(GRAPHICS_API_OPENGL_ES2)
	case RL_SHADER_UNIFORM_UINT: glUniform1uiv(locIndex, count, (unsigned int *)value); break;
	case RL_SHADER_UNIFORM_UIVEC2: glUniform2uiv(locIndex, count, (unsigned int *)value); break;
	case RL_SHADER_UNIFORM_UIVEC3: glUniform3uiv(locIndex, count, (unsigned int *)value); break;
	case RL_SHADER_UNIFORM_UIVEC4: glUniform4uiv(locIndex, count, (unsigned int *)value); break;
#endif
	case RL_SHADER_UNIFORM_SAMPLER2D: glUniform1iv(locIndex, count, (int *)value); break;

"""

# helper to detect cdata struct type
def _struct_info(v):
	ty = ffi.typeof(v)
	if ty.kind == 'struct':
		name = ty.cname  # ex: 'struct Vector2'
		return name.replace('struct ', '')
	return None

shader_enum = {
	'int':rl.RL_SHADER_UNIFORM_INT,
	'float':rl.RL_SHADER_UNIFORM_FLOAT,
	'Vector2':rl.RL_SHADER_UNIFORM_VEC2,
	'Vector3':rl.RL_SHADER_UNIFORM_VEC3,
	'Vector4':rl.RL_SHADER_UNIFORM_VEC4,
	'Color':rl.RL_SHADER_UNIFORM_VEC4,
}

def SetShaderValue(shader : Shader, loc: int, value: Any) -> None:
	tested = value
	size = 1

	if isinstance(value, Sequence):
		if not value: raise ValueError("Cannot upload an empty sequence as a uniform")
		tested = value[0]
		size = len(value)

	str_type = (
		 'int'   if isinstance(tested, int)
	else 'float' if isinstance(tested, float)
	else _struct_info(tested) )

	if str_type == 'Matrix':
		rl.SetShaderValueMatrix(shader,loc,value)
		return
	elif str_type == 'Texture':
		rl.SetShaderValueTexture(shader,loc,value)
		return

	uni_type = shader_enum[str_type]

	if size == 1:
		c_arr = ffi.new(f'{str_type}*', value)
		rl.SetShaderValue(shader, loc, c_arr, uni_type)
	else:
		c_arr = ffi.new(f'{str_type}[{size}]', value)
		rl.SetShaderValueV(shader, loc, c_arr, uni_type, size)


""" # in rlgl.h
# for color :
PIXELFORMAT_UNCOMPRESSED_GRAYSCALE = 1,	 // 8 bit per pixel (no alpha)
PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA,		// 8*2 bpp (2 channels)
PIXELFORMAT_UNCOMPRESSED_R5G6B5,			// 16 bpp
PIXELFORMAT_UNCOMPRESSED_R8G8B8,			// 24 bpp
PIXELFORMAT_UNCOMPRESSED_R5G5B5A1,		  // 16 bpp (1 bit alpha)
PIXELFORMAT_UNCOMPRESSED_R4G4B4A4,		  // 16 bpp (4 bit alpha)
PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,		  // 32 bpp
PIXELFORMAT_UNCOMPRESSED_R32,			   // 32 bpp (1 channel - float)
PIXELFORMAT_UNCOMPRESSED_R32G32B32,		 // 32*3 bpp (3 channels - float)
PIXELFORMAT_UNCOMPRESSED_R32G32B32A32,	  // 32*4 bpp (4 channels - float)
PIXELFORMAT_UNCOMPRESSED_R16,			   // 16 bpp (1 channel - half float)
PIXELFORMAT_UNCOMPRESSED_R16G16B16,		 // 16*3 bpp (3 channels - half float)
PIXELFORMAT_UNCOMPRESSED_R16G16B16A16,	  // 16*4 bpp (4 channels - half float)
PIXELFORMAT_COMPRESSED_DXT1_RGB,			// 4 bpp (no alpha)
PIXELFORMAT_COMPRESSED_DXT1_RGBA,		   // 4 bpp (1 bit alpha)
PIXELFORMAT_COMPRESSED_DXT3_RGBA,		   // 8 bpp
PIXELFORMAT_COMPRESSED_DXT5_RGBA,		   // 8 bpp
PIXELFORMAT_COMPRESSED_ETC1_RGB,			// 4 bpp
PIXELFORMAT_COMPRESSED_ETC2_RGB,			// 4 bpp
PIXELFORMAT_COMPRESSED_ETC2_EAC_RGBA,	   // 8 bpp
PIXELFORMAT_COMPRESSED_PVRT_RGB,			// 4 bpp
PIXELFORMAT_COMPRESSED_PVRT_RGBA,		   // 4 bpp
PIXELFORMAT_COMPRESSED_ASTC_4x4_RGBA,	   // 8 bpp
PIXELFORMAT_COMPRESSED_ASTC_8x8_RGBA		// 2 bpp

# for depth raylib lets opengl choose (not easy to set)
"""
def create_render_buffer(width : int, height:int,
	colorFormat:int=rl.PIXELFORMAT_UNCOMPRESSED_R8G8B8,
	depth_map:bool=False
	) -> RenderTexture :
	# has a color buffer by default
	#target = rl.LoadRenderTexture(width, height)

	target = RenderTexture()
	target.id = rl.rlLoadFramebuffer()
	
	if target.id > 0:
		rl.rlEnableFramebuffer(target.id)

		target.texture.width = width
		target.texture.height = height
		
		if colorFormat:
			target.texture.id = rl_load_texture(None, width, height, colorFormat, 1)
			target.texture.format = colorFormat
			target.texture.mipmaps = 1
			rl.rlFramebufferAttach(target.id, target.texture.id, rl.RL_ATTACHMENT_COLOR_CHANNEL0, rl.RL_ATTACHMENT_TEXTURE2D, 0)

		if depth_map:
			target.depth.id = rl.rlLoadTextureDepth(width, height, False)
			target.depth.width = width
			target.depth.height = height
			target.depth.format = 19
			target.depth.mipmaps = 1
			rl.rlFramebufferAttach(target.id, target.depth.id, rl.RL_ATTACHMENT_DEPTH, rl.RL_ATTACHMENT_TEXTURE2D, 0)
		
		rl.rlDisableFramebuffer()

	return target

def SetMaterialTexture(mat : Material, loc : int, tex : Texture):
	mat_ptr = ffi.new(f'Material*', mat)
	rl.SetMaterialTexture(mat_ptr, loc, tex)

def LoadModelAnimations(path : str):
	anims_cnt = ffi.new(f'int*', 0)
	anims = rl.LoadModelAnimations(path, anims_cnt)
	return [anims[i] for i in range(anims_cnt[0])]



class BetterShader:
	"""
	Parses a shader definition file containing two functions: vertex() and fragment().
	Extracts uniforms, varying, in, and out variables, then generates GLSL code for both stages.
	"""
	# Regex to capture qualifier, type, and name from declarations
	_decl_pattern = re.compile(r"\n(uniform|in|out|varying|const)\s+(\S+)\s+([^;]+).*?;")
	# Regex to extract function bodies
	_func_pattern = re.compile(
		r'\n'                            # start at a newline
		r'[\w\*\s&<>]+?\s+'              # return type (e.g. void, bool, vec4, const mat4&)
		r'([A-Za-z_]\w*)'                # function name
		r'\s*\(([^)]*)\)\s*'             # argument list
		r'(?:\{\n|\n\{\n)'               # opening brace on same or next line
		r'([\s\S]*?)'                    # function body (non-greedy)
		r'\n\}'                          # closing brace at column 0
	)

	_opengl_version = '#version 420'
	_vertex_start = 'void vertex()'
	_fragment_start = 'void fragment()'
	_main_start = 'void main()'

	def __init__(self, filepath):
		self.filepath = filepath
		self.uniforms = []		# list of uniform names
		self.uniform_locs = {}	# dict of uniform locs
		self.varyings = []		# list of varying names
		self.ins = []			# list of in-variable names
		self.outs = []			# list of out-variable names
		self.consts = []		# list of constants
		self.functions = []	   	# list of function definitions
		self.vertex_glsl = ''	# Generated GLSL code for vertex shader
		self.fragment_glsl = ''	# Generated GLSL code for fragment shader
		

		self._parse_file(filepath)
		self._generate_glsl()

		self.shader = rl.LoadShaderFromMemory(
			self.vertex_glsl.encode('utf-8'),
			self.fragment_glsl.encode('utf-8')
		)
		
		for type, name in self.uniforms:
			self.uniform_locs[name] = rl.GetShaderLocation(self.shader, name.encode('utf-8'))

	def valid(self) -> bool:
		return self.shader.id > 0

	def __enter__(self) -> None:
		rl.BeginShaderMode(self.shader)

	def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
		rl.EndShaderMode();


	def __setattr__(self, name: str, value: Any):
		if hasattr(self, 'uniform_locs') and name in self.uniform_locs:
			SetShaderValue(self.shader, self.uniform_locs[name], value)
		else:
			object.__setattr__(self, name, value)

	def _parse_file(self, filepath):
		with open(filepath, 'r', encoding="utf8") as f:
			text = f.read()

		decls = self._decl_pattern.findall(text)
		for qual, typ, name in decls:
			if qual == 'uniform':
				self.uniforms.append((typ, name))
			elif qual == 'varying':
				self.varyings.append((typ, name))
			elif qual == 'in':
				self.ins.append((typ, name))
			elif qual == 'out':
				self.outs.append((typ, name))
			elif qual == 'const':
				self.consts.append((typ, name))

		self.functions = [ m.group(0).strip() for m in self._func_pattern.finditer(text) ]

		for i, f in enumerate(self.functions):
			if f.startswith(self._vertex_start): self._vertex_body = i
			elif f.startswith(self._fragment_start): self._fragment_body = i


	def _generate_glsl(self):
		# Vertex Shader
		if hasattr(self, '_vertex_body'):
			v_lines = [
			self._opengl_version, '',
			*map(lambda kv: f'in {kv[0]} {kv[1]};', self.ins), '',
			*map(lambda kv: f'uniform {kv[0]} {kv[1]};', self.uniforms), '',
			*map(lambda kv: f'out {kv[0]} {kv[1]};', self.varyings), '',
			*map(lambda kv: f'const {kv[0]} {kv[1]};', self.consts), '',
			*self.functions[:self._vertex_body], '',
			self.functions[self._vertex_body].replace(self._vertex_start, self._main_start)
			]
			self.vertex_glsl = '\n'.join(v_lines)
		
		# will use the default vertex shader
		else:
			self.vertex_glsl = ''
			self._vertex_body = -1

		functions_after_vertex_but_not_fragment = [
		  self.functions[i] for i in range(self._vertex_body+1,len(self.functions))
		  if i != self._fragment_body
		]

		# Fragment Shader
		f_lines = [
		self._opengl_version, '',
		*map(lambda kv: f'in {kv[0]} {kv[1]};', self.varyings), '',
		*map(lambda kv: f'uniform {kv[0]} {kv[1]};', self.uniforms), '',
		*map(lambda kv: f'out {kv[0]} {kv[1]};', self.outs), '',
		*map(lambda kv: f'const {kv[0]} {kv[1]};', self.consts), '',
		*functions_after_vertex_but_not_fragment, '',
		self.functions[self._fragment_body].replace(self._fragment_start, self._main_start)
		]
		self.fragment_glsl = '\n'.join(f_lines)