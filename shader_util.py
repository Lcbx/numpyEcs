

import numpy as np
from pyrr import Matrix44 as Mat4, Vector3 as Vec3, Vector4 as Vec4
import glfw, ctypes, atexit, time

import wgpu
from wgpu.utils.glfw_present_info import get_glfw_present_info

import os
import re
from glob import glob
from typing import Any, Sequence, Iterable, List, Dict, Tuple, NamedTuple, Callable
from jinja2 import Environment, FileSystemLoader, StrictUndefined



class Camera:
	def __init__(self, position: tuple, target: tuple, up: tuple, fovy_deg:float, near:float=0.1, far:float=1000.0, perspective:bool=True):
		self.position = Vec3(position)
		self.target = Vec3(target)
		self.up = Vec3(up)
		self.fovy_deg = fovy_deg
		self.near = near
		self.far = far
		self.perspective = perspective

	def view(self) -> Mat4:
		return Mat4.look_at(self.position, self.target, self.up)

	def projection(self, aspect) -> Mat4:
		if self.perspective:
			return Mat4.perspective_projection( self.fovy_deg, aspect, self.near, self.far )

		top = self.fovy_deg * 0.5;
		right = top*aspect;
		return Mat4.orthogonal_projection(-right, right, top, -top, self.near, self.far)


type GLFWwindow_ptr = ctypes._Pointer[glfw._GLFWwindow]
class RenderContext:
	# simple window loop on top of glfw
	# inspired by https://github.com/pygfx/rendercanvas/blob/main/rendercanvas/glfw.py

	canvas : wgpu.GPUCanvasContext|None = None
	adapter : wgpu.GPUAdapter
	device : wgpu.GPUDevice
	presentation_format : str

	window : GLFWwindow_ptr
	windowDimensions : tuple[float,float] = (0.0,0.0)
	aspect : float = 1.0

	depth_format : str = wgpu.TextureFormat.depth24plus
	depth :  wgpu.GPUTextureView

	# event_name:[handlers]
	# handlers return True if no need to propagate further
	# NOTE: we should in reverse order if more recent handler have priority
	event_handlers : Dict[str, List[Callable]] = {}

	def __init__(self):
		raise Exception('this class is not meant to be instanciated')

	@classmethod
	def InitWindow(cls, w:float, h:float, title:str, vsync:bool=True, highpower:bool=True) -> None:

		print(f'{glfw.__version__=}')
		glfw.init()
		atexit.register(glfw.terminate)
		
		# in case monitor setup becomes relevant
		monitors = glfw.get_monitors()
		for monitor in monitors:
			mode = glfw.get_video_mode(monitor)
			print(f'{mode=}')

		# NOTE: we can set to fullscreen with glfw.set_window_monitor(monitor)

		glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
		#glfw.window_hint(glfw.RESIZABLE, True)
		#glfw.window_hint(glfw.DECORATED, False)

		monitor = glfw.get_primary_monitor()
		monitor_x, monitor_y, monitor_w, monitor_h = glfw.get_monitor_workarea(monitor)
		glfw.window_hint(glfw.POSITION_X, monitor_w-w)
		glfw.window_hint(glfw.POSITION_Y, 30) 

		# width, height, title, monitor (for full screen), window (for opengl context sharing)
		cls.window = glfw.create_window(w, h, title, None, None)
		cls.setup_callbacks()
		#glfw.set_window_pos(cls.window, monitor_w-w, 30)

		# wgpu context
		present_info = get_glfw_present_info(cls.window, vsync=vsync)
		cls.canvas = wgpu.gpu.get_canvas_context(present_info)

		cls.setup(highpower=highpower)
		cls.presentation_format = cls.canvas.get_preferred_format(cls.adapter)
		cls.canvas.configure(device=cls.device, format=cls.presentation_format, usage=wgpu.TextureUsage.RENDER_ATTACHMENT)

		#print(cls.canvas._get_capabilities_screen(cls.adapter))
		# ['Fifo', 'FifoRelaxed', 'Mailbox', 'Immediate']
		#cls.canvas.set_present_mode('FifoRelaxed') #edited the wgpu library to add this method
		#cls.canvas._configure_screen_real()

	@classmethod
	def setup_callbacks(cls) -> None:

		# key is key down, key up
		# char is resulting utf8 character
		# ex: pressing shift + 0 up and down generates ')' char
		glfw.set_key_callback(cls.window, cls.setup_event('key'))
		glfw.set_char_callback(cls.window, cls.setup_event('char'))

		# mouse stuff
		glfw.set_mouse_button_callback(cls.window, cls.setup_event('mouse_click'))
		glfw.set_cursor_pos_callback(cls.window, cls.setup_event('mouse_move'))
		glfw.set_cursor_enter_callback(cls.window, cls.setup_event('mouse_enter'))
		glfw.set_scroll_callback(cls.window, cls.setup_event('mouse_scroll'))#, info_log=True))


	@classmethod
	def setup(cls, *, highpower:bool) -> None:
		""" setup gpu compute, not necessarily with canvas output. used notably for tests """
		request_params = {'power_preference': 'high-performance' if highpower else 'low-power' }
		if cls.canvas: request_params['canvas'] = cls.canvas
		cls.adapter = wgpu.gpu.request_adapter_sync(**request_params)
		cls.device = cls.adapter.request_device_sync()


	@classmethod
	def updateWindowSize(cls, wh : Tuple[int, int]) -> None:
		# NOTE: some versions of glfw send resize events
		w, h = wh
		cls.windowDimensions = wh
		cls.canvas.set_physical_size(w, h)
		cls.aspect = w/h
		cls.depth = cls.device.create_texture(
			size=(w, h, 1),
			format= cls.depth_format,
			usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
		).create_view()


	@classmethod
	def capture_mouse(cls, capture:bool=True) -> None:
		""" hide and keep the mouse inside the window
		NOTE: there's also glfw.CURSOR_HIDDEN which lets the cursor get out of the window
		you can create cursors ex: cursor = glfw.create_cursor(image, 0, 0) # generating image is some work though
		use standard cursors ex: cursor = glfw.create_standard_cursor(cursor_flag)
		then glfw.set_cursor(self._window, cursor)
		"""
		glfw.set_input_mode(cls.window, glfw.CURSOR, glfw.CURSOR_DISABLED if capture else glfw.CURSOR_NORMAL)


	@classmethod
	def subscribe_event(cls, channel_name:str, handler:Callable) -> None:
		cls.event_handlers[channel_name].append(handler)

	@classmethod
	def setup_event(cls, channel_name:str, mapper:Callable|None=None, info_log:bool=False) -> Callable:
		
		def print_handler(*args):
			print(channel_name, *args)
		
		handlers = []
		if info_log:
			handlers.append(print_handler)

		cls.event_handlers[channel_name] = handlers

		# usually first arg is window and we don't need that
		def deflt_mapper(*args)->None:
			return args[1:]

		mapper = mapper if mapper else deflt_mapper

		def handlers_call(*args)->None:
			if mapper: args = mapper(*args)
			for handler in cls.event_handlers[channel_name]:
				if handler(*args): break

		return handlers_call


	@classmethod
	def cleanup(cls):

		#cls.device._poll(True) # wait for last frame

		cls.canvas.unconfigure()

		glfw.destroy_window(cls.window)

		# work around https://github.com/glfw/glfw/issues/1766
		end_time = time.perf_counter() + 0.1
		while time.perf_counter() < end_time:
			glfw.wait_events_timeout(end_time - time.perf_counter())


	@classmethod
	def WindowLoop(cls) -> bool:

		wh = glfw.get_framebuffer_size(cls.window)
		if wh != cls.windowDimensions and wh[0]>0 and wh[1]>0:
			cls.updateWindowSize(wh)

		cls.canvas.present()
		# TODO: apply frame pacing here

		glfw.poll_events()

		if glfw.window_should_close(cls.window):
			cls.cleanup()
			return False

		return True

	@classmethod
	def RenderPass(cls, *args, **kwargs) -> '_RenderPass':
		return _RenderPass(*args, **kwargs)

	@classmethod
	def Shader(cls, *args, **kwargs) -> '_Shader':
		return _Shader(*args, **kwargs)


class _RenderPass:

	def __init__(self,
			#shader:BetterShader = None,
			camera:Camera|None = None,
			#texture:img.Framebuffer = None
			clear_color:tuple=(0.0, 0.0, 0.0, 1.0)
		):
		#self.shader = shader
		self.camera:Camera|None= camera
		self.clear_color : Tuple[float,float,float,float] = clear_color
		#self.texture = texture
		self.render_pass:wgpu.GPURenderCommandsMixin = None
		self.command_encoder:wgpu.CommandEncoder = None

	def __enter__(self):
		# TODO: support drawing into a custom framebuffer (self.texture)
		#if self.shader: self.shader.bind()
		#if self.texture: self.texture.bind()
		if self.camera:
			self.view = self.camera.view()
			# TODO: determine aspect from texture dimensions
			aspect = RenderContext.aspect
			self.projection = self.camera.projection(aspect)
		#	if 'uView' in self.shader._uniforms: self.shader['uView'] = self.view
		#	if 'uProj' in self.shader._uniforms: self.shader['uProj'] = self.projection
		#	if 'uViewProj' in self.shader._uniforms: self.shader['uViewProj'] = self.view @ self.projection
		self.command_encoder = RenderContext.device.create_command_encoder()
		self.render_pass = self.command_encoder.begin_render_pass(
			color_attachments=[
				wgpu.RenderPassColorAttachment(
					view=RenderContext.canvas.get_current_texture().create_view(),
					clear_value=self.clear_color,
					load_op="clear",
					store_op="store",
				)
			],
			depth_stencil_attachment=
				wgpu.RenderPassDepthStencilAttachment(
			        view = RenderContext.depth,
			        depth_clear_value = 1.0,
			        depth_load_op= "clear",
			        depth_store_op= "store",
				)
		)
		return self

	def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
		self.render_pass.end()
		RenderContext.device.queue.submit([
			self.command_encoder.finish()
		])

	def draw(self, mesh:'Mesh', shader:'_Shader', uniforms:'_UniformBuffer') -> None:
		render_pass = self.render_pass
		render_pass.set_pipeline(mesh.create_draw_call(shader))
		render_pass.set_bind_group(0, uniforms.bind_group)
		render_pass.set_vertex_buffer(0, mesh.vertex_buffer.handle)
		if mesh.instanced:
			render_pass.set_vertex_buffer(1, mesh.instance_buffer.handle)
		render_pass.set_index_buffer(mesh.index_buffer.handle, wgpu.IndexFormat.uint32)

		#index_count (int) – The number of indices to draw.
		#instance_count (int) – The number of instances to draw. Default 1.
		#first_index (int) – The index offset. Default 0.
		#base_vertex (int) – A number added to each index in the index buffer. Default 0.
		#first_instance (int) – The instance offset. Default 0.
		render_pass.draw_indexed(mesh.index_count, mesh.instance_count, 0, 0, 0)


_GL_to_dtype: Dict[str, np.dtype] = {
	"float":	 np.dtype((np.float32, ())),
	"vec2": 	 np.dtype((np.float32, (2,))),
	"vec3": 	 np.dtype((np.float32, (3,))),
	"vec4": 	 np.dtype((np.float32, (4,))),
	"mat4": 	 np.dtype((np.float32, (4,4))),
	"uint": 	 np.dtype((np.uint32, ())),
	"uvec2":	 np.dtype((np.int32, (2,))),
	"uvec3":	 np.dtype((np.int32, (3,))),
	"uvec4":	 np.dtype((np.int32, (4,))),
	"int":  	 np.dtype((np.int32, ())),
	"ivec2":	 np.dtype((np.int32, (2,))),
	"ivec3":	 np.dtype((np.int32, (3,))),
	"ivec4":	 np.dtype((np.int32, (4,))),
	"sampler2D": np.dtype((np.uint32, ())),
}

def glsl_to_dtype(spec:str)-> np.dtype:

	# array case
	if '[' in spec:
		definition = spec.replace(']', '').split('[')
		spec = definition[0]
		counts = tuple(map(int,definition[1:]))
	else: counts = None

	info = _GL_to_dtype.get(spec)
	if info is None:
		raise ValueError(f"Unsupported type {spec!r}")

	if counts:
		# rebuild item dimensions
		info = np.dtype( (info.base, info.shape + counts) )
	
	return info


def make_std430_dtype( original: List[Tuple[str, np.dtype]] ) -> np.dtype:

	dfields: List[Tuple[str, np.dtype]] = []
	offset = 0
	pad_idx = 0

	for name, dtype in original:
		print(name, dtype)
		dtype = np.dtype(dtype)

		dfields.append((name, dtype))
		offset += dtype.itemsize

		#print(dfields)
		missing_offset = (16 - offset) % 16
		if missing_offset != 0:
			#print(f'adding new pad {missing_offset}')
			dfields.append((f"_pad{pad_idx}", np.dtype(("u1", missing_offset))))
			offset += missing_offset
			pad_idx += 1

	#print(dfields)

	#return np.dtype(dfields, align=False)
	return np.dtype(dfields, align=True)



_VertexFormat : Dict[Any, Tuple[str, int]] = {
	np.int8:	("sint8",   1),
	np.uint8:	("uint8",   1),
	np.int16:	("sint16",  2),
	np.uint16:	("uint16",  2),
	np.int32:	("sint32",  4),
	np.uint32:	("uint32",  4),
	np.uint32:	("uint32",  4),
	np.float16:	("float16", 2),
	np.float32:	("float32", 4),
	np.float64:	("float64", 8),
}
# NOTE: wgpu.vertexFormat enum are strings in wgpu-py 
def dtype_to_vertex_format(dtype: np.dtype) -> List | str:

	dtype = np.dtype(dtype)

	#print(dtype)

	if dtype.fields is not None:
		fields_data = dtype.fields.values()
		#print(fields_data)
		res = []
		for field_dtype, offset in fields_data:
			vFormat = dtype_to_vertex_format(field_dtype)
			#print('vFormat', vFormat, type(vFormat))
			if isinstance(vFormat,str):
				res.append( (vFormat, offset) )
			elif isinstance(vFormat,List):
				for vf, local_offset in vFormat:
					res.append( (vf, offset+local_offset) )
		return res

	# Scalar case
	if dtype.subdtype is None:
		try:
			(prefix, scalar_size) = _VertexFormat[dtype]
		except KeyError:
			raise TypeError(f"Unsupported vertex scalar dtype: {dtype}")

		return getattr(wgpu.VertexFormat, prefix)

	# Array case
	base_dtype, shape = dtype.subdtype
	base_type = base_dtype.type

	count = shape[0]

	if count < 1 or count > 4:
		raise ValueError(f"VertexFormat arrays must have 1–4 components (found {count})")
	try:
		(prefix, scalar_size) = _VertexFormat[base_type]
	except KeyError:
		raise TypeError(f"Unsupported vertex base dtype: {base_type}")

	shape_len = len(shape)
	vFormat = (
			 getattr(wgpu.VertexFormat, prefix) if count == 1
		else getattr(wgpu.VertexFormat, f"{prefix}x{count}")
	)

	if shape_len == 1:
		return vFormat
	# matrix 
	if shape_len == 2:
		row_count = shape[1]
		return [ (vFormat, scalar_size * count * i) for i in range(row_count) ]
	
	else:
		raise TypeError(f"Only 1D or 2D array dtypes are valid vertex attributes (found {shape})")


# TODO: don't compile shaders we already compiled before
class _Shader:


	def __init__(self, *args, **kwargs):
		device = RenderContext.device
		source = self.ShaderSource(*args, **kwargs)
		self.source = source

		#print(source.vertex_glsl)
		#print(source.fragment_glsl)

		try:
			self.vert_module = device.create_shader_module(label="shader.vert",code=source.vertex_glsl)
		except Exception as e:
			print(source.vertex_glsl)
			raise e
		try:
			self.frag_module = device.create_shader_module(label="shader.frag",code=source.fragment_glsl)
		except Exception as e:
			print(source.fragment_glsl)
			raise e

		self.uniforms_dtype = make_std430_dtype(
			[(name, glsl_to_dtype(spec)) for spec, name in self.source.uniforms]
		)

		self.bind_group_layout = device.create_bind_group_layout(
			entries=[
				wgpu.BindGroupLayoutEntry(
					binding=0,
					visibility=wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
					buffer=wgpu.BufferBindingLayout(
						type=wgpu.BufferBindingType.uniform,
						has_dynamic_offset=False,
						#min_binding_size=self.uniforms_dtype.itemsize,
					),
				)
			]
		)		

		self.pipeline_layout = device.create_pipeline_layout(
			bind_group_layouts=[self.bind_group_layout]
		)

	# NOTE: we pass the shader since the mesh needs it for setup
	def UniformBuffer(self, *args, **kwargs) -> '_UniformBuffer':
		return _UniformBuffer(self, *args, **kwargs)

	def ShaderSource(self, *args, **kwargs) -> 'ShaderSource':
		return ShaderSource(*args, **kwargs)

class GpuBuffer:

	def __init__(self, content : np.ndarray, usage:int | wgpu.BufferUsage):#, max_count:int=1):
		device = RenderContext.device
		self.content = content
		self.handle = device.create_buffer_with_data(data=content, usage=usage)
		#buffer_size = max(max_count, content.size) * content.itemsize
		#print(buffer_size)
		#self.handle = device.create_buffer(size=content.itemsize * buffer_item_max_count, usage=usage)

	def upload(self) -> None:
		RenderContext.device.queue.write_buffer(self.handle, 0, self.content)


class _UniformBuffer(GpuBuffer):

	def __init__(self, shader : _Shader):
		device = RenderContext.device
		self.shader = shader

		content = np.empty((1),dtype=shader.uniforms_dtype)

		super().__init__(content, wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

		self.bind_group = device.create_bind_group(
			layout=shader.bind_group_layout,
			entries=[
				wgpu.BindGroupEntry(
					binding=0,
					resource=wgpu.BufferBinding(
						buffer=self.handle,
						offset=0,
						size=content.nbytes,
					),
				)
			],
		)



# TODO: pool meshes with same vertex attributes
class Mesh:

	# NOTE: is there a point to supporting meshes with no indices these days
	def __init__(self, vertex_data:np.ndarray, indices:np.ndarray, instance_data:np.ndarray|None = None):
		device = RenderContext.device

		self.index_count = indices.size
		self.instanced = isinstance(instance_data, np.ndarray)
		vertex_dtype = vertex_data.dtype

		vformat = dtype_to_vertex_format(vertex_dtype)
		vertex_loc_count = len(vformat) 
		print("instance", vformat, vertex_dtype.itemsize)
		vertexAttributes = [ wgpu.VertexAttribute(
				shader_location = i,
				offset=offset,
				format=vertexFormat
			) for i, (vertexFormat, offset)
			in enumerate(vformat)
		]
		

		# TODO: test if view(np.uint8) truly tobytes() but wihout a copy ?
		self.vertex_buffer = GpuBuffer(vertex_data, wgpu.BufferUsage.VERTEX)
		self.index_buffer = GpuBuffer(indices, wgpu.BufferUsage.INDEX)

		self.buffers_spec = [
			wgpu.VertexBufferLayout(
				array_stride=vertex_dtype.itemsize,
				step_mode=wgpu.VertexStepMode.vertex,
				attributes=vertexAttributes
			)
		]
		self.instance_count = instance_data.size if self.instanced else 1

		if self.instanced:
			instance_dtype = instance_data.dtype
			vformat = dtype_to_vertex_format(instance_dtype)
			print("instance", vformat, instance_dtype.itemsize)
			instanceAttributes = [ wgpu.VertexAttribute(
					shader_location = i + vertex_loc_count,
					offset=offset,
					format=vertexFormat
				) for i, (vertexFormat, offset)
				in enumerate(vformat)
			]

			self.instance_buffer = GpuBuffer(instance_data, wgpu.BufferUsage.VERTEX|wgpu.BufferUsage.COPY_DST)

			self.buffers_spec.append(
				wgpu.VertexBufferLayout(
					array_stride=instance_dtype.itemsize,
					step_mode=wgpu.VertexStepMode.instance,
					attributes=instanceAttributes
				))

	def create_draw_call(self, shader:_Shader) -> wgpu.GPURenderPipeline:
		return RenderContext.device.create_render_pipeline(
			layout=shader.pipeline_layout,
			vertex=wgpu.VertexState(
				module=shader.vert_module,
				entry_point="main",
				buffers=self.buffers_spec,
			),
			primitive=wgpu.PrimitiveState(
				topology=wgpu.PrimitiveTopology.triangle_list,
				cull_mode=wgpu.CullMode.back,
				front_face=wgpu.FrontFace.ccw,
			),
			depth_stencil=wgpu.DepthStencilState(
				format=RenderContext.depth_format,
				depth_write_enabled=True,
				depth_compare=wgpu.CompareFunction.less,
			),
			fragment=wgpu.FragmentState(
				module=shader.frag_module,
				entry_point="main",
				targets=[
					wgpu.ColorTargetState(format=RenderContext.presentation_format)
				],
			),
		)


def build_shader_program(shaderPath : str, **kwargs) -> Tuple['ShaderSource', _Shader]:
	sh = RenderContext.Shader(filepath=shaderPath, **kwargs)
	return sh.source, sh


class DefaultFalseDict(Dict):
	# Dict that returns False for any missing key; used in feature resolution
	def __missing__(self, key):
		return False

# Parses a shader definition file containing two functions: vertex() and fragment().
# Extracts uniforms, varying, in, and out variables, then generates GLSL code for both stages.
class ShaderSource:

	# Regex to capture qualifier, type, and name from declarations
	_decl_pattern = re.compile(
		r"^(?:layout\(location\s*=\s*(\d+)\)\s+)?"
		r"(uniform|in|out|(?:flat\s+)?varying|const)\s+"
		r"(\w+(?:\s*\[\s*\d*\s*\])*)\s+"
		r"([^;]+).*?;",
		re.MULTILINE
	)

	# Regex to extract function definitions with bodies
	_func_pattern = re.compile(
		r'\n'						# start at a newline
		r'[\w\*\s&<>]+?\s+'			# return type (e.g. void, bool, vec4, const mat4&)
		r'([A-Za-z_]\w*)'			# function name
		r'\s*\(([^)]*)\)\s*'		# argument list
		r'(?:\{\n|\n\{\n)'			# opening brace on same or next line
		r'([\s\S]*?)'				# function body (non-greedy)
		r'\n\}'						# closing brace at column 0
	)

	def create_variable_regex(self, name:str) -> re.Pattern:
		escaped = re.escape(name)
		_GLSL_IDENT_CHAR = "[A-Za-z0-9_]"
		pattern = rf"(?<!{_GLSL_IDENT_CHAR}){escaped}(?!{_GLSL_IDENT_CHAR})"
		return re.compile(pattern)

	_vertex_start = 'void vertex()'
	_fragment_start = 'void fragment()'
	_main_start = 'void main()'

	def __init__(
		self,
		*,
		source: str|None = None,
		filepath: str|None = None,
		basedir: str|None = None,
		features: Sequence[str]|None = None,
		params: Dict[str, Any]|None = None,
		glsl_version: str = '#version 450 core'
	):
		# :param source: master shader definition code (template).
		# :param filepath: Path to the master shader definition file (template).
		# :param features: Dict of feature flags for conditionals, e.g. {'FEATURE_FOG': True}.
		# :param params: Any extra variables you want available in templates.
		# :param glsl_version: Override #version.

		assert( source or filepath )

		self.source = source if source else open(filepath, 'r').read()
		self._basedir = basedir if basedir else os.path.dirname(os.path.abspath(filepath)) if filepath else glob('**/shaders/', recursive=True)
		self._glsl_version = glsl_version
		
		self.uniforms = []	  # list[(type, name)]
		self.varyings = []	  # list[(type, name)]
		self.ins = []		   # list[(loc, type, name)]
		self.outs = []		  # list[(loc, type, name)]
		self.consts = []		# list[(type, name)]
		self.functions = []	 # list[str]
		self.vertex_glsl = ''   # rendered vertex GLSL
		self.fragment_glsl = '' # rendered fragment GLSL

		#print(f'{self._basedir=}')
		#print(f'compiling {filepath}')

		# Render the shader source through Jinja2 (handles #if / #include)
		text = self._render_template(features, params)

		# Parse the rendered text for declarations & functions
		self._parse_rendered_text(text)

		# Generate final vertex/fragment GLSL
		self._generate_glsl()

	def _render_template(self, features: Sequence[str]|None, params: Dict[str, Any]|None) -> str:
		# evaluate the preprocessor sections using jinja2

		env = Environment(
			loader=FileSystemLoader(self._basedir),
			undefined=StrictUndefined,  # fail fast for missing vars
			autoescape=False,			# GLSL is not HTML
			keep_trailing_newline=True,
			trim_blocks=True,
			lstrip_blocks=True,
			line_statement_prefix='#' # << key: enable #if/#endif/#include
		)
		template = env.from_string(self.source)

		featuresDict = DefaultFalseDict()
		if features is not None: featuresDict.update({k: True for k in features})

		return template.render(FEATURES=featuresDict, PARAMS=params or {})

	
	def _parse_rendered_text(self, text: str) -> None:
		# Declarations
		for loc, qual, typ, name in self._decl_pattern.findall(text):
			if qual == 'uniform':
				self.uniforms.append((typ, name))
			elif qual == 'varying':
				self.varyings.append((loc, typ, name, False))
			elif qual == 'flat varying':
				self.varyings.append((loc, typ, name, True))
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


	def _generate_glsl(self) -> None:
		def inoutFmt(loc, tup, *, inoutStr):
			if len(tup) == 4:
				loc_, typ, name, flat = tup
			else:
				loc_, typ, name = tup
				flat = False
			loc = loc_ if loc_ else loc
			return f'layout(location = {loc}){(' flat' if flat else '')} {inoutStr} {typ} {name};'
		def inStr(entry):
			return inoutFmt(*entry, inoutStr='in')
		def outStr(entry):
			return inoutFmt(*entry, inoutStr='out')

		# workaround for simple uniforms : make one big uniform struct and use aliases
		uniforms_definition = f'layout(set=0, binding=0, std430) uniform Uniforms {{\n {(
				"".join(f"\t{typ} {name};\n" for typ, name in self.uniforms)
			)} }} _U;'

		# replace uniform references by alias
		for i, f in enumerate(self.functions):
			for _, name  in self.uniforms:
				rx = self.create_variable_regex(name)
				self.functions[i] = rx.sub(f'_U.{name}', f)

		v_lines = (
			self._glsl_version, '',
			*map(inStr, enumerate(self.ins)), '',
			uniforms_definition, '',
			*map(outStr, enumerate(self.varyings)), '',
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
			self._glsl_version, '',
			*map(inStr, enumerate(self.varyings)), '',
			uniforms_definition, '',
			*map(outStr, enumerate(self.outs)), '',
			*map(lambda kv: f'const {kv[0]} {kv[1]};', self.consts), '',
			*functions_after_vertex_but_not_fragment, '',
			self.functions[self._fragment_body].replace(self._fragment_start, self._main_start)
		]
		self.fragment_glsl = '\n'.join(f_lines)



from pygltflib import GLTF2, BufferView, Accessor

def _get_data_from_accessor(gltf: GLTF2, accessor_index: int) -> np.ndarray:
	acc: Accessor = gltf.accessors[accessor_index]
	bv: BufferView = gltf.bufferViews[acc.bufferView]
	buf = gltf.buffers[bv.buffer]

	if buf.uri is None:  # GLB
		bin_chunk = gltf.binary_blob()
	else:
		raise NotImplementedError("External buffers not handled")

	b = bv.byteOffset or 0
	e = b + (bv.byteLength or 0)
	view_bytes = memoryview(bin_chunk)[b:e]

	type_num_comps = {
		"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16
	}[acc.type]

	np_dtype = {
		5120: np.int8,
		5121: np.uint8,
		5122: np.int16,
		5123: np.uint16,
		5125: np.uint32,
		5126: np.float32
	}[acc.componentType]

	stride = bv.byteStride or (np.dtype(np_dtype).itemsize * type_num_comps)
	count = acc.count
	offset = acc.byteOffset or 0
	tight = np.dtype(np_dtype).itemsize * type_num_comps

	arr = np.frombuffer(view_bytes, dtype=np_dtype, count=count * type_num_comps, offset=offset)
	if stride != tight:
		rec = np.empty((count, type_num_comps), dtype=np_dtype)
		base = offset
		for i in range(count):
			start = base + i * stride
			rec[i] = np.frombuffer(view_bytes, dtype=np_dtype, count=type_num_comps, offset=start)
		return rec
	else:
		return arr.reshape(count, type_num_comps)



def load_gltf_first_mesh(glb_path: str) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
	gltf = GLTF2().load(glb_path)
	mesh = gltf.meshes[0]
	prim = mesh.primitives[0]
	
	#print(f'{mesh=}')

	pos = _get_data_from_accessor(gltf, prim.attributes.POSITION).astype(np.float32)
	nor = (
		_get_data_from_accessor(gltf, prim.attributes.NORMAL).astype(np.float16)
		if prim.attributes.NORMAL is not None
		else np.zeros_like(pos, dtype=np.float16)
	)
	uv = (
		_get_data_from_accessor(gltf, prim.attributes.TEXCOORD_0).astype(np.float16)
		if prim.attributes.TEXCOORD_0 is not None
		else np.zeros((pos.shape[0], 2), dtype=np.float16)
	)
	idx = _get_data_from_accessor(gltf, prim.indices)

	return pos, nor, uv, idx


def load_gltf_first_mesh_interleaved(glb_path: str) -> Tuple[np.ndarray,np.ndarray]:
	( pos, nor, uv, idx ) = load_gltf_first_mesh(glb_path)

	N = pos.shape[0]
	vertex_dtype = np.dtype(
	[
		("position", np.float32, (3,)),
		("normal",   np.float32, (3,)), # float16x3 does not exist in wgpu
		("uv",	   	 np.float16, (2,)),
	], align=False,   # important: keep it tightly packed (no padding surprises)
	)

	#print("stride:", vertex_dtype.itemsize)
	#print("offsets:", {n: vertex_dtype.fields[n][1] for n in vertex_dtype.names})

	verts = np.empty(N, dtype=vertex_dtype)
	verts["position"] = pos
	verts["normal"]   = nor
	verts["uv"]		  = uv

	# interleave
	return verts, idx.astype(np.uint32).reshape(-1)


"""

def GenTextureMipmaps(texture : img.Texture):
	#gl.glBindTexture(gl.GL_TEXTURE_2D, texture.id)
	#gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
	#gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
	raise Exception("not implemented")

# NOTE: putting a smaller texture into a bigger one is not allowed
def TransferDepth(from_fbo:int, f_w:int, f_h:int, to_fbo:int, t_w:int, t_h:int):
	#gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, from_fbo)
	#gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, to_fbo)
	#gl.glBlitFramebuffer(
	#	0, 0, f_w, f_h,
	#	0, 0, t_w, t_h,
	#	#gl.GL_COLOR_BUFFER_BIT,
	#	gl.GL_DEPTH_BUFFER_BIT,
	#	gl.GL_NEAREST
	#)
	raise Exception("not implemented")

def ClearBuffers():
	#gl.glClear(gl.GL_COLOR_BUFFER_BIT|gl.GL_DEPTH_BUFFER_BIT)
	raise Exception("not implemented")
def ClearColorBuffer():
	#gl.glClear(gl.GL_COLOR_BUFFER_BIT)
	raise Exception("not implemented")

def setClearColor(*color):
	#gl.glClearColor(*color)
	raise Exception("not implemented")


def SetPolygonOffset(value:float, flat:float=0.0):
	#gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
	#gl.glPolygonOffset(value, flat)
	raise Exception("not implemented")
def DisablePolygonOffset():
	#gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
	raise Exception("not implemented")

def DrawTexture(tex:img.Texture, width:float, height:float):
	# screen-wide rectangle, y-flipped due to default OpenGL coordinates
	#rl.DrawTextureRec(tex, (0, 0, width, -height), (0, 0), rl.WHITE)
	raise Exception("not implemented")


# TODO : toggle automatically based on whether it is a postprocess or 3d pass 
def DisableDepth():
	#gl.glDepthMask(gl.GL_FALSE)
	#gl.glDisable(gl.GL_DEPTH_TEST)
	raise Exception("not implemented")
def EnableDepth():
	#gl.glDepthMask(gl.GL_TRUE)
	#gl.glEnable(gl.GL_DEPTH_TEST)
	raise Exception("not implemented")

def EnableCullFace():
	#gl.glEnable(gl.GL_CULL_FACE)
	raise Exception("not implemented")
def DisableCullFace():
	#gl.glDisable(gl.GL_CULL_FACE)
	raise Exception("not implemented")

class BetterShader(Program):
	__slots__ = 'vertex_glsl', 'fragment_glsl'

	def __init__(self, *shaders: Shader) -> None:
		Program.__init__(self, *shaders)
		for s in shaders:
			source = Shader._get_shader_source(s._id)
			if s.type == 'vertex':
				self.vertex_glsl = source
			elif s.type == 'fragment':
				self.fragment_glsl = source
			else: raise Exception('unsupported shader type')

def create_frame_buffer(width:int, height:int,
						colorFormat = gl.GL_RGBA8,
						depth_map:bool = False,
						samples:int = 4):

	#framebuffer = img.Framebuffer()

	#create_tex = create_simple_texture if samples==1 else TextureMsaa.create

	#if colorFormat:
	#	color_tex = create_tex(width=width, height=height, internalformat=colorFormat, samples=samples)
	#	framebuffer.attach_texture(color_tex, attachment=gl.GL_COLOR_ATTACHMENT0)

	#if depth_map:
	#	depth_tex = create_tex(width=width, height=height, internalformat=gl.GL_DEPTH_COMPONENT24, fmt=gl.GL_DEPTH_COMPONENT, samples=samples)
	#	framebuffer.attach_texture(depth_tex, attachment=gl.GL_DEPTH_ATTACHMENT)
	#	depth = depth_tex
	#else:
	#	depth_rb = img.buffer.Renderbuffer.create(width, height, gl.GL_DEPTH_COMPONENT24, samples=samples)
	#	framebuffer.attach_renderbuffer(depth_rb, attachment=gl.GL_DEPTH_ATTACHMENT)
	#	depth = depth_rb
	#return framebuffer
	raise Exception("not implemented")



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


class FastMaterial(Group):
	
	def __init__(self, program:Program, order:int=0, parent:Group|None=None) -> None:
		super().__init__(order, parent)
		self.program = program
		self.uniforms = {}

	def set_state(self) -> None:
		self.program.use()
		for name, value in self.uniforms.items():
			self.program[name] = value

	def __hash__(self) -> int:
		hashable = str(sorted(self.uniforms.items()))
		#print(hashable)
		return hash( (self.program, self.order, self.parent, hashable) )

	def __eq__(self, other) -> bool:
		return (self.__class__ is other.__class__ and
				self.program == other.program and
				self.order == other.order and
				self.parent == other.parent and
				self.uniforms == other.uniforms)


class Mesh:
	def __init__(self, program:Program, positions, normals, UVs, indices, name= '') -> None:
		self.program = program
		self.positions = np.array(positions).reshape(-1)
		self.normals = np.array(normals).reshape(-1)
		self.UVs = np.array(UVs).reshape(-1)
		self.indices = np.array(indices).reshape(-1)
		self.group = FastMaterial(program)
		self.vertex_count = self.positions.size // 3
		self.name = name
		self.instances = [] 

	def __setitem__(self, name :str, value)->None:
		self.group.uniforms[name] = value

	def draw(self, batch:Batch=None, transform:Mat4=None):
		#if batch is None: batch = get_default_batch()
		if 'uModel' in self.program._uniforms:
			if transform is None: transform = Mat4.identity()
			self.program['uModel'] = transform
		vlist = self.program.vertex_list_indexed(
			self.vertex_count,
			gl.GL_TRIANGLES,
			self.indices,
			batch=batch,
			group=self.group,
			aPos=('f', self.positions),
			aNormal=('f', self.normals),
			aUV=('f', self.UVs),
		)
		self.instances.append(vlist)
		return vlist


def _get_data_from_accessor(gltf: GLTF2, accessor_index: int) -> np.ndarray:
	acc: Accessor = gltf.accessors[accessor_index]
	bv: BufferView = gltf.bufferViews[acc.bufferView]
	buf = gltf.buffers[bv.buffer]

	# load buffer data
	if buf.uri is None:  # GLB
		bin_chunk = gltf.binary_blob()  # bytes
	else:
		# external .bin (not the case for .glb)
		raise NotImplementedError("External buffers not handled")

	# slice underlying bytes for this view
	b = bv.byteOffset or 0
	e = b + (bv.byteLength or 0)
	view_bytes = memoryview(bin_chunk)[b:e]

	# stride/offset into accessor
	comp_type = acc.componentType
	type_num_comps = {
		"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16
	}[acc.type]

	np_dtype = {
		5120: np.int8,
		5121: np.uint8,
		5122: np.int16,
		5123: np.uint16,
		5125: np.uint32,
		5126: np.float32
	}[comp_type]

	stride = bv.byteStride or (np.dtype(np_dtype).itemsize * type_num_comps)
	count = acc.count
	offset = acc.byteOffset or 0

	# read tightly into numpy (use frombuffer then stride)
	arr = np.frombuffer(view_bytes, dtype=np_dtype, count=count*type_num_comps, offset=offset)
	if stride != np.dtype(np_dtype).itemsize * type_num_comps:
		raise NotImplementedError("Interleaved views not handled")
	#	# handle interleaved views (rare in simple exports)
	#	# fall back to manual gathering
	#	rec = np.empty((count, type_num_comps), dtype=np_dtype)
	#	base = offset
	#	for i in range(count):
	#		start = base + i*stride
	#		rec[i] = np.frombuffer(view_bytes, dtype=np_dtype, count=type_num_comps, offset=start)
	#	return rec
	#else:
	return arr.reshape(count, type_num_comps)


def load_gltf_meshes(program : Program, glb_path: str)-> IndexedVertexList:
	gltf = GLTF2().load(glb_path)

	meshes = []
	for mesh in gltf.meshes:
		for prim in mesh.primitives:
			pos = _get_data_from_accessor(gltf, prim.attributes.POSITION).astype(np.float32)
			nor = _get_data_from_accessor(gltf, prim.attributes.NORMAL).astype(np.float32) if prim.attributes.NORMAL is not None else np.zeros_like(pos)
			uv  = _get_data_from_accessor(gltf, prim.attributes.TEXCOORD_0).astype(np.float32) if prim.attributes.TEXCOORD_0 is not None else np.zeros((pos.shape[0],2), dtype=np.float32)
			idx = _get_data_from_accessor(gltf, prim.indices)
			if idx.dtype != np.uint32:
				idx = idx.astype(np.uint32)

		model = Mesh(program, pos, nor, uv, idx)
		meshes.append(model)

	return meshes



# NOTES:
# * default shader takes tint as a uniform so this can't set multiple colors
# * would perform better with instanciation (but this allows keeping the same shader)
# * we should also just update the vertex list each frame when moving them instead of recreating a mesh
def Cubes(program, positions, sizes, color
	) -> Mesh:

	CUBE_POSITIONS_24 = np.array((
		# +Z (front)
		(-0.5,-0.5,+0.5), (+0.5,-0.5,+0.5), (+0.5,+0.5,+0.5), (-0.5,+0.5,+0.5),
		# -Z (back)
		(+0.5,-0.5,-0.5), (-0.5,-0.5,-0.5), (-0.5,+0.5,-0.5), (+0.5,+0.5,-0.5),
		# +X (right)
		(+0.5,-0.5,+0.5), (+0.5,-0.5,-0.5), (+0.5,+0.5,-0.5), (+0.5,+0.5,+0.5),
		# -X (left)
		(-0.5,-0.5,-0.5), (-0.5,-0.5,+0.5), (-0.5,+0.5,+0.5), (-0.5,+0.5,-0.5),
		# +Y (top)
		(-0.5,+0.5,+0.5), (+0.5,+0.5,+0.5), (+0.5,+0.5,-0.5), (-0.5,+0.5,-0.5),
		# -Y (bottom)
		(-0.5,-0.5,-0.5), (+0.5,-0.5,-0.5), (+0.5,-0.5,+0.5), (-0.5,-0.5,+0.5),
	))

	CUBE_NORMALS_24 = np.array(
		([ 0.0, 0.0, 1.0] * 4) +   # +Z
		([ 0.0, 0.0,-1.0] * 4) +   # -Z
		([ 1.0, 0.0, 0.0] * 4) +   # +X
		([-1.0, 0.0, 0.0] * 4) +   # -X
		([ 0.0, 1.0, 0.0] * 4) +   # +Y
		([ 0.0,-1.0, 0.0] * 4),	# -Y
		dtype=np.float32
	)

	CUBE_UVS_24 = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0] * 6, dtype=np.float32)

	CUBE_INDICES_36 = np.array([
		0, 1, 2,  2, 3, 0,		# +Z
		4, 5, 6,  6, 7, 4,		# -Z
		8, 9, 10,  10, 11, 8,	 # +X
		12, 13, 14,  14, 15, 12,  # -X
		16, 17, 18,  18, 19, 16,  # +Y
		20, 21, 22,  22, 23, 20,  # -Y
	], dtype=np.uint32)

	positions = np.asarray(positions, dtype=np.float32)
	sizes	 = np.asarray(sizes, dtype=np.float32)

	n = positions.shape[0]

	# repeat normals/uvs per cube
	normals = np.tile(CUBE_NORMALS_24, n)   # (N*24*3,)
	uvs	 = np.tile(CUBE_UVS_24, n)	   # (N*24*2,)

	# (N,3): positions & sizes
	# (N,24,3): base scaled per-cube + per-cube position
	pos = (positions[:, None, :] + CUBE_POSITIONS_24[None, :, :] * sizes[:, None, :]).reshape(-1).astype(np.float32)

	# (N,36): base indices + 24*i
	indices = (CUBE_INDICES_36[None, :] + (np.arange(n, dtype=np.uint32) * 24)[:, None]).reshape(-1).astype(np.uint32)

	mesh = Mesh(program, pos, normals, uvs, indices)
	mesh['uTint'] = color
	return mesh

"""