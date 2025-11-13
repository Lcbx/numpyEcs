import raylib as rl

from pyray import ( Vector2, Vector3, Vector4,
	Shader, Material, Texture, RenderTexture, Camera3D,
	ffi, rl_load_texture
)

from OpenGL.GL import (
	glBindFramebuffer, glBlitFramebuffer,
	glClear, glEnable, glDisable, glPolygonOffset,
	glBindTexture, glGenerateMipmap, glDepthMask,
	GL_READ_FRAMEBUFFER, GL_DRAW_FRAMEBUFFER,
	GL_DEPTH_BUFFER_BIT, GL_COLOR_BUFFER_BIT, GL_NEAREST,
	GL_POLYGON_OFFSET_FILL, GL_MULTISAMPLE, GL_TEXTURE_2D,
	GL_DEPTH_TEST, GL_TRUE, GL_FALSE
)

import os
import re
from typing import Any, Sequence, Dict, Optional
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from pyray import Vector2, Vector3, Color, Camera3D, RenderTexture


def InitWindow(w:float, h:float, title:str):
	# NOTE: using MSAA with prepass generate z-artifacts
	#rl.SetConfigFlags(rl.FLAG_MSAA_4X_HINT) #|rl.FLAG_WINDOW_RESIZABLE)
	#rl.SetConfigFlags(rl.FLAG_WINDOW_TOPMOST | rl.FLAG_WINDOW_UNDECORATED)
	rl.SetTraceLogLevel(rl.LOG_WARNING)
	rl.InitWindow(w, h, title.encode())
	#rl.SetTargetFPS(60)

	display = rl.GetCurrentMonitor()
	monitor_w = rl.GetMonitorWidth(display)
	monitor_h = rl.GetMonitorHeight(display)
	rl.SetWindowPosition(monitor_w - w, 0)
	#rl.SetWindowPosition(0, 0)
	#rl.SetWindowSize(w, h)
	#w, WINDOW_h = monitor_w, monitor_h
	##print(w, h)
	#rl.MaximizeWindow()
	rl.SetWindowState(rl.FLAG_WINDOW_UNDECORATED)
	#rl.ToggleFullscreen()
	#rl.SetExitKey(0)




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
	#print('FRAMEBUFFER ID : ', target.id)
	
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

def GenTextureMipmaps(texture : Texture):
	glBindTexture(GL_TEXTURE_2D, texture.id)
	glGenerateMipmap(GL_TEXTURE_2D)
	glBindTexture(GL_TEXTURE_2D, 0)

# NOTE: putting a smaller texture into a bigger one is not allowed
def TransferDepth(from_fbo:int, f_w:int, f_h:int, to_fbo:int, t_w:int, t_h:int):
	glBindFramebuffer(GL_READ_FRAMEBUFFER, from_fbo)
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, to_fbo)
	glBlitFramebuffer(
		0, 0, f_w, f_h,
		0, 0, t_w, t_h,
		#GL_COLOR_BUFFER_BIT,
		GL_DEPTH_BUFFER_BIT,
		GL_NEAREST
	)

def ClearColorBuffer():
	glClear(GL_COLOR_BUFFER_BIT)

def SetPolygonOffset(value:float):
	glEnable(GL_POLYGON_OFFSET_FILL)
	glPolygonOffset(value, 0.0)
def DisablePolygonOffset():
	glDisable(GL_POLYGON_OFFSET_FILL)

def EnableMultisampling():
	glEnable(GL_MULTISAMPLE)
def DisableMultisampling():
	glEnable(GL_MULTISAMPLE)

def DrawTexture(tex:Texture, width:float, height:float):
	# screen-wide rectangle, y-flipped due to default OpenGL coordinates
	rl.DrawTextureRec(tex, (0, 0, width, -height), (0, 0), rl.WHITE)

def ClearBuffers():
	rl.ClearBackground(rl.WHITE)


# TODO : toggle automatically based on whether it is a postprocess or 3d pass 
def DisableDepth():
	glDepthMask(GL_FALSE)
	glDisable(GL_DEPTH_TEST)
	#pass
def EnableDepth():
	glDepthMask(GL_TRUE)
	glEnable(GL_DEPTH_TEST)
	#pass


class DefaultFalseDict(Dict):
	"""Dict that returns False for any missing key; used in feature resolution"""
	def __missing__(self, key):
		return False

class BetterShader:
	"""
	Parses a shader definition file containing two functions: vertex() and fragment().
	Extracts uniforms, varying, in, and out variables, then generates GLSL code for both stages.
	"""

	# Regex to capture qualifier, type, and name from declarations
	_decl_pattern = re.compile(r"\n(?>layout\(location = (\d+)\) )?(uniform|in|out|varying|const)\s+(\S+)\s+([^;]+).*?;", re.MULTILINE)

	# Regex to extract function definitions with bodies
	_func_pattern = re.compile(
		r'\n'                            # start at a newline
		r'[\w\*\s&<>]+?\s+'              # return type (e.g. void, bool, vec4, const mat4&)
		r'([A-Za-z_]\w*)'                # function name
		r'\s*\(([^)]*)\)\s*'             # argument list
		r'(?:\{\n|\n\{\n)'               # opening brace on same or next line
		r'([\s\S]*?)'                    # function body (non-greedy)
		r'\n\}'                          # closing brace at column 0
	)

	_vertex_start = 'void vertex()'
	_fragment_start = 'void fragment()'
	_main_start = 'void main()'

	def __init__(
		self,
		filepath: str,
		*,
		features: Optional[Sequence[str]] = None,
		params: Optional[Dict[str, Any]] = None,
		glsl_version: str = '#version 430'
	):
		"""
		:param filepath: Path to the master shader definition file (template).
		:param features: Dict of feature flags for conditionals, e.g. {'FEATURE_FOG': True}.
		:param params: Any extra variables you want available in templates.
		:param glsl_version: Override #version (defaults to #version 430).
		"""
		self.filepath = filepath
		self._basedir = os.path.dirname(os.path.abspath(filepath))
		self._opengl_version = glsl_version
		self.uniforms = []	   # list[(type, name)]
		self.uniform_locs = {}   # name -> location id
		self.varyings = []	   # list[(type, name)]
		self.ins = []			# list[(loc, type, name)]
		self.outs = []		   # list[(loc, type, name)]
		self.consts = []		 # list[(type, name)]
		self.functions = []	  # list[str]
		self.vertex_glsl = ''	# rendered vertex GLSL
		self.fragment_glsl = ''  # rendered fragment GLSL


		#print(f'compiling {filepath}')

		# Render the shader source through Jinja2 (handles #if / #include)
		text = self._render_template(features, params)

		# Parse the rendered text for declarations & functions
		self._parse_rendered_text(text)

		# Generate final vertex/fragment GLSL
		self._generate_glsl()

		rl.TraceLog(rl.LOG_INFO, f'compiling {filepath}'.encode())

		# Compile via raylib
		self.shaderStruct = rl.LoadShaderFromMemory(
			self.vertex_glsl.encode(),
			self.fragment_glsl.encode()
		)

		for typ, name in self.uniforms:
			self.uniform_locs[name] = rl.GetShaderLocation(self.shaderStruct, name.encode('utf-8'))

	def valid(self) -> bool:
		return self.shaderStruct.id > 0

	def __enter__(self) -> None:
		rl.BeginShaderMode(self.shaderStruct)

	def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
		rl.EndShaderMode()

	def __setattr__(self, name: str, value: Any) -> None:
		try:
			SetShaderValue(self.shaderStruct, self.uniform_locs[name], value)
		except: pass
		object.__setattr__(self, name, value)

	def _render_template(self, features: Sequence[str], params: Dict[str, Any]) -> str:
		"""
		evaluate the preprocessor sections using jinja2
		"""
		env = Environment(
			loader=FileSystemLoader(self._basedir),
			undefined=StrictUndefined,		# fail fast for missing vars
			autoescape=False,				 # GLSL is not HTML
			keep_trailing_newline=True,
			trim_blocks=True,
			lstrip_blocks=True,
			line_statement_prefix='#'		 # << key: enable #if/#endif/#include
		)
		template_name = os.path.basename(self.filepath)
		template = env.get_template(template_name)

		ctx = {}

		featuresDict = DefaultFalseDict()
		if features is not None: featuresDict.update({k: True for k in features})

		return template.render(FEATURES=featuresDict, PARAMS=params or {})

	
	def _parse_rendered_text(self, text: str):
		# Declarations
		for loc, qual, typ, name in self._decl_pattern.findall(text):
			if qual == 'uniform':
				self.uniforms.append((typ, name))
			elif qual == 'varying':
				self.varyings.append((typ, name))
			elif qual == 'in':
				self.ins.append((loc, typ, name))
			elif qual == 'out':
				self.outs.append((loc, typ, name))
			elif qual == 'const':
				self.consts.append((typ, name))

		# Functions (full match = whole function; we keep their text)
		self.functions = [m.group(0).strip() for m in self._func_pattern.finditer(text)]

		#print(*[ f[0:20] for f in self.functions], sep='\n')

		# Find vertex()/fragment() bodies
		for i, f in enumerate(self.functions):
			if f.startswith(self._vertex_start):
				self._vertex_body = i
				#print('-> got vert', hasattr(self, '_vertex_body'))
			elif f.startswith(self._fragment_start):
				self._fragment_body = i
				#print('-> got frag', hasattr(self, '_fragment_start'))


	def _generate_glsl(self):
		def inoutFmt(loc, typ, name, inoutStr):
			return (f'layout(location = {loc}) ' if loc else '') + f'{inoutStr} {typ} {name};'
		def inStr(entry):
			return inoutFmt(*entry, 'in')
		def outStr(entry):
			return inoutFmt(*entry, 'out')

		v_lines = (
			self._opengl_version, '',
			*map(inStr, self.ins), '',
			*map(lambda kv: f'uniform {kv[0]} {kv[1]};', self.uniforms), '',
			*map(lambda kv: f'out {kv[0]} {kv[1]};', self.varyings), '',
			*map(lambda kv: f'const {kv[0]} {kv[1]};', self.consts), '',
			*self.functions[:self._vertex_body], '',
			self.functions[self._vertex_body].replace(self._vertex_start, self._main_start)
		)
		self.vertex_glsl = '\n'.join(v_lines)

		# Helper: functions after vertex() but not fragment()
		start_idx = (self._vertex_body + 1) if self._vertex_body is not None else 0
		functions_after_vertex_but_not_fragment = [
			self.functions[i] for i in range(start_idx, len(self.functions))
			if i != self._fragment_body
		]

		# Fragment shader (required)
		f_lines = [
			self._opengl_version, '',
			*map(lambda kv: f'in {kv[0]} {kv[1]};', self.varyings), '',
			*map(lambda kv: f'uniform {kv[0]} {kv[1]};', self.uniforms), '',
			*map(outStr, self.outs), '',
			*map(lambda kv: f'const {kv[0]} {kv[1]};', self.consts), '',
			*functions_after_vertex_but_not_fragment, '',
			self.functions[self._fragment_body].replace(self._fragment_start, self._main_start)
		]
		self.fragment_glsl = '\n'.join(f_lines)


class RenderContext:

	def __init__(self,
		shader:BetterShader = None,
		texture:RenderTexture = None,
		camera:Camera3D = None,
		clipPlanes:(float, float) = None):
		self.shader = shader
		self.texture = texture
		self.camera = camera
		self.clipPlanes = clipPlanes

	def __enter__(self):
		if self.shader: rl.BeginShaderMode(self.shader.shaderStruct)
		if self.texture: rl.BeginTextureMode(self.texture)
		if self.clipPlanes: rl.rlSetClipPlanes(self.clipPlanes[0], self.clipPlanes[1])
		if self.camera: rl.BeginMode3D(self.camera)
		return self

	def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
		if self.camera: rl.EndMode3D()
		if self.texture: rl.EndTextureMode()
		rl.EndShaderMode()

class WatchTimer:
	nesting : int = 0
	timers: list = []
	report = ''
	
	def __init__(self, region:str):
		self.region = region
	
	def __enter__(self) -> None:
		self.start_time = rl.GetTime()
		self.nesting = WatchTimer.nesting
		WatchTimer.nesting += 1
		WatchTimer.timers.append(self)

	def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
		WatchTimer.nesting -= 1
		self.message = self.get_message()
		rl.TraceLog(rl.LOG_DEBUG, self.message.encode())
	
	def get_message(self):
		#return ('  ' * self.nesting + f'{self.region} : { self.elapsed_ms() :.1f}ms')	
		return ('  ' * self.nesting + f'{self.region} : { self.elapsed_percent() :.0f}%')	
	
	def elapsed_ms(self):
		return (rl.GetTime() - self.start_time) * 1000.0

	def elapsed_percent(self):
		ft = rl.GetFrameTime() + 0.0001
		return (rl.GetTime() - self.start_time) / ft * 100.0
	
	def capture():
		WatchTimer.report = '\n'.join( list(map(
			lambda t: t.message if hasattr(t, 'message') else t.get_message(),
			WatchTimer.timers))
		)
		WatchTimer.timers.clear()

	def display(x, y, size, color):
		rl.DrawText(WatchTimer.report.encode(), x, y, size, color)