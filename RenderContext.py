from __future__ import annotations

from typing import Any, Type, Sequence, Iterator, Iterable, List, Dict, Tuple, Callable
from time import perf_counter as get_time, sleep
import time, os, re
from threading import Lock
from dataclasses import dataclass

import wgpu
from wgpu.utils.glfw_present_info import get_glfw_present_info
from wgpu import BufferUsage

import numpy as np

from glob import glob
from jinja2 import Environment, FileSystemLoader, StrictUndefined

Color = tuple[float, float, float, float]


#useful for buffer resizing
# 0->1, 1->2, 2->4, 3->4, 4->8, 5->8
def higher_pow2(n: int|np.uint32|np.uint64) -> int:
	return 1 << int(n).bit_length()

def is_pow2(n:int|np.uint32|np.uint64) -> int:
	return  n & (n-1) == 0


class _RenderContext:
	# simple window loop on top of glfw
	# inspired by https://github.com/pygfx/rendercanvas/blob/main/rendercanvas/glfw.py
	def __init__(self):
		self.canvas: wgpu.GPUCanvasContext | None = None
		self.adapter: wgpu.GPUAdapter
		self.device: wgpu.GPUDevice
		self.presentation_format: str

		self.window: glfw._GLFWwindow
		self.windowDimensions: tuple[float, float] = (0.0, 0.0)
		self.aspect: float = 1.0

		self.target_frame_time: float | None = None
		self.frame_start: float = 0.0
		self.frame_time: float = 0.0

		self.depth_format: str = wgpu.TextureFormat.depth24plus
		self.depth: wgpu.GPUTextureView
		self._screen_view: wgpu.GPUTextureView | None = None

		# event_name:[handlers]
		# handlers return True if no need to propagate further
		self.event_handlers: Dict[str, List[Callable]] = {}

		# before startup -> resource_name:func
		# after startup  -> resource_name:resource
		# used to setup utils if imported
		self.resources = {}

	def InitWindow(
		cls,
		w: float,
		h: float,
		title: str,
		highpower_gpu: bool = True,
		target_fps: int = 0,
	) -> None:
		"""Init a window and wgpu rendering context.

		:param highpower_gpu: use the high performance gpu if there are multiple
		:param target_fps: -1 is limitless, 0 is vsync, other values is fps limit
		"""
		global glfw
		import glfw, atexit

		glfw.init()
		atexit.register(cls.cleanup)

		glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)

		monitor = glfw.get_primary_monitor()
		monitor_x, monitor_y, monitor_w, monitor_h = glfw.get_monitor_workarea(monitor)
		glfw.window_hint(glfw.POSITION_X, monitor_w - w)
		glfw.window_hint(glfw.POSITION_Y, 30)

		# had weird case where vsync was not blocking
		if target_fps == 0:
			video_mode = glfw.get_video_mode(monitor)
			target_fps = video_mode.refresh_rate

		if target_fps != -1:
			cls.target_frame_time = 1.0 / float(target_fps)

		cls.window = glfw.create_window(w, h, title, None, None)
		cls.setup_callbacks()
		cls.setup_graphics(vsync=target_fps == 0, highpower_gpu=highpower_gpu)

		for name, init in cls.resources.items():
			cls.resources[name] = init()

		cls.frame_start = get_time()

	def setup_graphics(cls, *, vsync: bool, highpower_gpu: bool) -> None:
		"""Setup gpu compute & render surface."""
		present_info = get_glfw_present_info(cls.window, vsync=vsync)
		cls.canvas = wgpu.gpu.get_canvas_context(present_info)
		cls.setup_graphics_backend(highpower_gpu=highpower_gpu)
		cls.presentation_format = cls.canvas.get_preferred_format(cls.adapter)
		cls.canvas.configure(device=cls.device, format=cls.presentation_format)

	def setup_graphics_backend(cls, *, highpower_gpu: bool) -> None:
		"""Setup gpu compute, not necessarily with canvas output. Used notably for tests."""
		request_params = {
			"power_preference": "high-performance" if highpower_gpu else "low-power"
		}
		if cls.canvas:
			request_params["canvas"] = cls.canvas
		cls.adapter = wgpu.gpu.request_adapter_sync(**request_params)
		cls.device = cls.adapter.request_device_sync()

	def setup_callbacks(cls) -> None:
		glfw.set_key_callback(cls.window, cls.setup_event("key"))
		glfw.set_char_callback(cls.window, cls.setup_event("char"))
		glfw.set_mouse_button_callback(cls.window, cls.setup_event("mouse_click"))
		glfw.set_cursor_pos_callback(cls.window, cls.setup_event("mouse_move"))
		glfw.set_cursor_enter_callback(cls.window, cls.setup_event("mouse_enter"))
		glfw.set_scroll_callback(cls.window, cls.setup_event("mouse_scroll"))

	def updateWindowSize(cls, wh: Tuple[int, int]) -> None:
		w, h = wh
		cls.windowDimensions = wh
		cls.canvas.set_physical_size(w, h)
		cls.aspect = w / h
		cls.depth = cls.device.create_texture(
			size=(w, h, 1),
			format=cls.depth_format,
			usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
		).create_view()

	def capture_mouse(cls, capture: bool = True) -> None:
		"""Hide and keep the mouse inside the window."""
		glfw.set_input_mode(
			cls.window,
			glfw.CURSOR,
			glfw.CURSOR_DISABLED if capture else glfw.CURSOR_NORMAL,
		)

	def subscribe_event(cls, channel_name: str, handler: Callable) -> None:
		cls.event_handlers[channel_name].append(handler)

	def setup_event(
		cls,
		channel_name: str,
		mapper: Callable | None = None,
		info_log: bool = False,
	) -> Callable:
		def print_handler(*args):
			print(channel_name, *args)

		handlers = []
		if info_log:
			handlers.append(print_handler)
		cls.event_handlers[channel_name] = handlers

		def deflt_mapper(*args) -> tuple:
			return args[1:]

		mapper = mapper if mapper else deflt_mapper

		def handlers_call(*args) -> None:
			if mapper:
				args = mapper(*args)
			for handler in cls.event_handlers[channel_name]:
				if handler(*args):
					break

		return handlers_call

	def cleanup(cls) -> None:
		if cls.canvas:
			cls.canvas.unconfigure()

		if hasattr(cls, "window"):
			glfw.destroy_window(cls.window)

		# work around https://github.com/glfw/glfw/issues/1766
		end_time = get_time() + 0.1
		while get_time() < end_time:
			glfw.wait_events_timeout(end_time - get_time())
		glfw.terminate()

	def window_loop(cls) -> bool:
		""" Present the previous image, update window/input state, and continue the loop. """
		cls.present()

		wh = glfw.get_framebuffer_size(cls.window)
		if wh != cls.windowDimensions and wh[0] > 0 and wh[1] > 0:
			cls.updateWindowSize(wh)

		cls._frame_pacing()
		glfw.poll_events()
		return not glfw.window_should_close(cls.window)

	def present(cls) -> None:
		cls.canvas.present()
		cls._screen_view = None

	_FRAME_WAIT_MARGIN = 0.001 # 1ms

	def _frame_pacing(cls) -> None:
		now = get_time()
		cls.frame_time = now - cls.frame_start

		if cls.target_frame_time and cls.frame_time < cls.target_frame_time:
			sleep_time = cls.target_frame_time - cls.frame_time - cls._FRAME_WAIT_MARGIN
			if sleep_time > 0.0:
				sleep(sleep_time)
			wait_end = cls.frame_start + cls.target_frame_time
			while now < wait_end:
				now = get_time()

		cls.frame_start = now

	# commands -----------------------------------------------------------------

	def commands(cls, label: str | None = None) -> CommandContext:
		return CommandContext(label=label)

	def Command(cls, label: str | None = None) -> wgpu.GPUCommandEncoder:
		"""Raw command encoder escape hatch."""
		return cls.device.create_command_encoder(label=label or "")

	def submit(cls, *command_buffers: wgpu.GPUCommandBuffer) -> None:
		cls.device.queue.submit(list(command_buffers))

	# attachments ---------------------------------------------------------------

	def screen(
		cls,
		*,
		clear: Color | None = None,
		store: bool = True,
	) -> ColorAttachment:
		if cls._screen_view is None:
			cls._screen_view = cls.canvas.get_current_texture().create_view()
		return ColorAttachment(cls._screen_view, cls.presentation_format, clear, store)

	def depth_attachment(
		cls,
		*,
		clear: float | None = None,
		store: bool = True,
		read_only: bool = False,
	) -> DepthAttachment:
		return DepthAttachment(cls.depth, cls.depth_format, clear, store, read_only)

	# resource factories --------------------------------------------------------

	def Buffer(cls, *args, **kwargs) -> GpuBuffer:
		return GpuBuffer(*args, **kwargs)

	def Texture(cls, *args, **kwargs) -> Texture:
		return Texture(*args, **kwargs)

	def Sampler(cls, *args, **kwargs) -> Sampler:
		return Sampler(*args, **kwargs)

	def Mesh(cls, *args, **kwargs) -> Mesh:
		return Mesh(*args, **kwargs)

	def Shader(cls, *args, **kwargs) -> Shader:
		return Shader(*args, **kwargs)

	def RenderPipeline(cls, *args, **kwargs) -> RenderPipeline:
		return RenderPipeline(*args, **kwargs)

	def ComputePipeline(cls, *args, **kwargs) -> ComputePipeline:
		return ComputePipeline(*args, **kwargs)


RenderContext = _RenderContext()


class CommandContext:
	"""One independent GPU command encoder.

	Multiple CommandContexts may be encoded independently and later submitted in the
	order in which their resulting command buffers should execute.
	"""

	def __init__(self, label: str | None = None):
		self.handle = RenderContext.device.create_command_encoder(label=label or "")
		self.finished = False

	def render_pass(
		self,
		*,
		color: ColorAttachment | Sequence[ColorAttachment] = (),
		depth: DepthAttachment | None = None,
		label: str | None = None,
	) -> RenderPass:
		return RenderPass(self, color=color, depth=depth, label=label)

	def compute_pass(self, *, label: str | None = None) -> ComputePass:
		return ComputePass(self, label=label)

	def clear_buffer(
		self,
		buffer: GpuBuffer,
		*,
		offset: int = 0,
		size: int | None = None,
	) -> None:
		self.handle.clear_buffer(buffer.handle, offset, size)

	def copy_buffer(
		self,
		source: GpuBuffer,
		destination: GpuBuffer,
		*,
		source_offset: int = 0,
		destination_offset: int = 0,
		size: int | None = None,
	) -> None:
		if size is None:
			size = min(source.handle.size - source_offset, destination.handle.size - destination_offset)
		self.handle.copy_buffer_to_buffer(
			source.handle,
			source_offset,
			destination.handle,
			destination_offset,
			size,
		)

	def finish(self, label: str | None = None) -> wgpu.GPUCommandBuffer:
		if self.finished:
			raise RuntimeError("CommandContext.finish() called more than once")
		self.finished = True
		return self.handle.finish(label=label or "")


class RenderPass:
	def __init__(
		self,
		commands: CommandContext,
		*,
		color: ColorAttachment | Sequence[ColorAttachment] = (),
		depth: DepthAttachment | None = None,
		label: str | None = None,
	):
		self.commands = commands
		self.color = [color] if isinstance(color, ColorAttachment) else list(color)
		self.depth = depth
		self.label = label

		self.handle: wgpu.GPURenderPassEncoder | None = None
		self.pipeline: RenderPipeline | None = None
		self.bind_groups: Dict[int, BindGroup | wgpu.GPUBindGroup] = {}
		self._gpu_pipeline: wgpu.GPURenderPipeline | None = None

	@property
	def color_formats(self) -> tuple[str, ...]:
		return tuple(attachment.format for attachment in self.color)

	@property
	def depth_format(self) -> str | None:
		return self.depth.format if self.depth else None

	def __enter__(self) -> RenderPass:
		self.handle = self.commands.handle.begin_render_pass(
			label=self.label or "",
			color_attachments=[attachment.descriptor() for attachment in self.color],
			depth_stencil_attachment=self.depth.descriptor() if self.depth else None,
		)
		return self

	def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
		self.end()

	def end(self) -> None:
		if self.handle is not None:
			self.handle.end()
			self.handle = None

	def set_pipeline(self, pipeline: RenderPipeline) -> None:
		self.pipeline = pipeline
		self._variant = None

	def set_bind_group(
		self,
		index: int,
		bindings: BindGroup | wgpu.GPUBindGroup,
	) -> None:
		if isinstance(bindings, BindGroup) and bindings.group != index:
			raise ValueError( f"BindGroup belongs to group {bindings.group}, not group {index}" )
		self.bind_groups[index] = bindings

	def set_vertex_buffer(
		self,
		slot: int,
		buffer: GpuBuffer,
		*,
		offset: int = 0,
		size: int | None = None,
	) -> None:
		self._require_handle().set_vertex_buffer(slot, buffer.handle, offset, size)

	def set_index_buffer(
		self,
		buffer: GpuBuffer,
		*,
		format: str,
		offset: int = 0,
		size: int | None = None,
	) -> None:
		self._require_handle().set_index_buffer(buffer.handle, format, offset, size)

	def draw(
		self,
		vertex_count: int,
		*,
		instance_count: int = 1,
		first_vertex: int = 0,
		first_instance: int = 0,
	) -> None:
		self._prepare_pipeline(())
		self._require_handle().draw(
			vertex_count,
			instance_count,
			first_vertex,
			first_instance,
		)

	def draw_indexed(
		self,
		index_count: int,
		*,
		instance_count: int = 1,
		first_index: int = 0,
		base_vertex: int = 0,
		first_instance: int = 0,
	) -> None:
		self._prepare_pipeline(())
		self._require_handle().draw_indexed(
			index_count,
			instance_count,
			first_index,
			base_vertex,
			first_instance,
		)

	def draw_mesh(
		self,
		mesh: Mesh,
		*,
		instances: GpuBuffer | None = None,
		instance_count: int | None = None,
	) -> None:
		buffers = [(mesh.vertex_dtype, wgpu.VertexStepMode.vertex)]
		if instances is not None:
			buffers.append((instances.content.dtype, wgpu.VertexStepMode.instance))

		self._prepare_pipeline(buffers)
		handle = self._require_handle()

		vertex_start, vertex_count = mesh.vertex_range
		handle.set_vertex_buffer(
			0,
			mesh.vertex_buffer.handle,
			vertex_start * mesh.vertex_dtype.itemsize,
			vertex_count * mesh.vertex_dtype.itemsize,
		)

		if instances is not None:
			handle.set_vertex_buffer(1, instances.handle)

		index_start, index_count = mesh.index_range
		handle.set_index_buffer(
			mesh.index_buffer.handle,
			mesh.index_format,
			index_start * mesh.index_dtype.itemsize,
			index_count * mesh.index_dtype.itemsize,
		)

		if instance_count is None:
			instance_count = instances.content.size if instances is not None else 1
		handle.draw_indexed(mesh.index_count, instance_count, 0, 0, 0)

	
	def _prepare_pipeline(
		self,
		buffers: Sequence[tuple[np.dtype, str]],
	) -> None:
		if self.pipeline is None:
			raise RuntimeError("No RenderPipeline has been set on this pass")

		variant = self.pipeline.get_variant(self, buffers)
		handle = self._require_handle()

		if variant is not self._variant:
			handle.set_pipeline(variant.handle)
			self._variant = variant

		for index, bindings in self.bind_groups.items():
			if isinstance(bindings, BindGroup):
				bindings = variant.bind_group(bindings)

			handle.set_bind_group(index, bindings)

	def _require_handle(self) -> wgpu.GPURenderPassEncoder:
		if self.handle is None:
			raise RuntimeError("RenderPass is not active")
		return self.handle


class ComputePass:
	def __init__(self, commands: CommandContext, *, label: str | None = None):
		self.commands = commands
		self.label = label
		self.handle: wgpu.GPUComputePassEncoder | None = None
		self.pipeline: ComputePipeline | None = None

	def __enter__(self) -> ComputePass:
		self.handle = self.commands.handle.begin_compute_pass(label=self.label or "")
		return self

	def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
		self.end()

	def end(self) -> None:
		if self.handle is not None:
			self.handle.end()
			self.handle = None

	def set_pipeline(self, pipeline: ComputePipeline) -> None:
		self.pipeline = pipeline
		self._require_handle().set_pipeline(pipeline.handle)

	def set_bind_group(self, index: int, bindings: BindGroup | wgpu.GPUBindGroup ) -> None:
		if self.pipeline is None:
			raise RuntimeError("Set the ComputePipeline before its bind groups")
		if isinstance(bindings, BindGroup) and bindings.group != index:
			raise ValueError( f"BindGroup belongs to group {bindings.group}, not {index}" )
		bindings = self.pipeline._variant.bind_group(bindings)
		self._require_handle().set_bind_group(index, bindings)

	def dispatch(self, x: int, y: int = 1, z: int = 1) -> None:
		self._require_handle().dispatch_workgroups(x, y, z)

	def dispatch_indirect(self, buffer: GpuBuffer, offset: int = 0) -> None:
		self._require_handle().dispatch_workgroups_indirect(buffer.handle, offset)

	def _require_handle(self) -> wgpu.GPUComputePassEncoder:
		if self.handle is None:
			raise RuntimeError("ComputePass is not active")
		return self.handle


@dataclass(frozen=True)
class ColorAttachment:
	view: wgpu.GPUTextureView
	format: str
	clear: Color | None = None
	store: bool = True

	def descriptor(self) -> wgpu.RenderPassColorAttachment:
		return wgpu.RenderPassColorAttachment(
			view=self.view,
			clear_value=self.clear or (0.0, 0.0, 0.0, 0.0),
			load_op="clear" if self.clear is not None else "load",
			store_op="store" if self.store else "discard",
		)


@dataclass(frozen=True)
class DepthAttachment:
	view: wgpu.GPUTextureView
	format: str
	clear: float | None = None
	store: bool = True
	read_only: bool = False

	def descriptor(self) -> wgpu.RenderPassDepthStencilAttachment:
		return wgpu.RenderPassDepthStencilAttachment(
			view=self.view,
			depth_clear_value=self.clear or 0.0,
			depth_load_op="clear" if self.clear is not None else "load",
			depth_store_op="store" if self.store else "discard",
			depth_read_only=self.read_only,
		)

# numpy buffers ---------------------------------------------------------------

_VertexFormat: Dict[Any, Tuple[str, int]] = {
	np.int8:	("sint8",   1),
	np.uint8:	("uint8",   1),
	np.int16:	("sint16",  2),
	np.uint16:	("uint16",  2),
	np.int32:	("sint32",  4),
	np.uint32:	("uint32",  4),
	np.int64:	("sint64",  8),
	np.uint64:	("uint64",  8),
	np.uint64:	("uint32",  4),
	np.float16:	("float16", 2),
	np.float32:	("float32", 4),
	np.float64:	("float64", 8),
}


def dtype_to_vertex_format(dtype: np.dtype) -> List | str:
	dtype = np.dtype(dtype)
	if dtype.fields is not None:
		res = []
		for name, (field_dtype, offset) in dtype.fields.items():
			vformat = dtype_to_vertex_format(field_dtype)
			if isinstance(vformat, str):
				res.append((vformat, offset))
			else:
				for vf, local_offset in vformat:
					res.append((vf, offset + local_offset))
		return res

	if dtype.subdtype is None:
		try:
			prefix, scalar_size = _VertexFormat[dtype.type]
		except KeyError:
			raise TypeError(f"Unsupported vertex scalar dtype: {dtype}") from None
		return getattr(wgpu.VertexFormat, prefix)

	base_dtype, shape = dtype.subdtype
	base_type = base_dtype.type
	count = shape[0]
	if count < 1 or count > 4:
		raise ValueError(
			f"VertexFormat arrays must have 1-4 components (found {count} for {dtype.subdtype})"
		)

	try:
		prefix, scalar_size = _VertexFormat[base_type]
	except KeyError:
		raise TypeError(f"Unsupported vertex base dtype: {base_type}") from None

	vformat = getattr(wgpu.VertexFormat, prefix if count == 1 else f"{prefix}x{count}")
	if len(shape) == 1:
		return vformat
	if len(shape) == 2:
		row_count = shape[1]
		return [(vformat, scalar_size * count * i) for i in range(row_count)]

	raise TypeError(
		f"Only 1D or 2D array dtypes are valid vertex attributes (found {shape})"
	)


class GpuBuffer:
	def __init__(
		self,
		content: np.ndarray,
		usage: wgpu.BufferUsage,
		upload: bool = True,
		label: str | None = None,
	):
		self.content = np.asarray(content)
		self.usage = usage
		self.label = label
		self.generation = 0

		if upload:
			self.handle = RenderContext.device.create_buffer_with_data(
				label=label or "",
				data=self.content,
				usage=self.usage,
			)
		else:
			self.handle = RenderContext.device.create_buffer(
				label=label or "",
				size=max(4, self.content.nbytes),
				usage=self.usage,
			)

	def resize(self, count: int) -> bool:
		size = higher_pow2(max(4, count * self.content.itemsize))
		if size <= self.handle.size:
			return False

		self.handle = RenderContext.device.create_buffer(
			label=self.label or "",
			size=size,
			usage=self.usage,
		)
		self.generation += 1
		return True

	def upload(self) -> None:
		RenderContext.device.queue.write_buffer(self.handle, 0, self.content)

	def upload_range(self, start: int, count: int) -> None:
		if count <= 0:
			return
		uploaded = self.content[start:start + count]
		RenderContext.device.queue.write_buffer(
			buffer=self.handle,
			buffer_offset=start * self.content.itemsize,
			data=uploaded,
		)

	def write(self, data: np.ndarray, *, offset: int = 0) -> None:
		self.content = np.asarray(data)
		RenderContext.device.queue.write_buffer(self.handle, offset, self.content)


class GpuBufferPool:
	def __init__(
		self,
		dtype: np.dtype,
		usage: wgpu.BufferUsage,
		initial_count: int = 0,
	):
		self.dtype = np.dtype(dtype)
		self.used = 0
		self.buffer = GpuBuffer(
			np.empty(initial_count, dtype=self.dtype),
			usage | wgpu.BufferUsage.COPY_DST,
			upload=False,
		)

	def alloc(self, data: np.ndarray) -> Tuple[int, int]:
		start = self.used
		count = data.size
		end = start + count

		self._ensure_capacity(end)
		self.buffer.content[start:end] = data
		self.buffer.upload_range(start, count)
		self.used = end
		return start, count

	def _ensure_capacity(self, count: int) -> None:
		old_content = self.buffer.content
		if count <= old_content.size:
			return

		new_count = higher_pow2(max(1, count))
		new_content = np.empty(new_count, dtype=self.dtype)
		new_content[:self.used] = old_content[:self.used]
		self.buffer.content = new_content

		if self.buffer.resize(new_count):
			self.buffer.upload_range(0, self.used)


class Mesh:
	# Cache geometry with the same dtype in shared pools, as in the original implementation.
	vertex_buffers: Dict[np.dtype, GpuBufferPool] = {}
	index_buffers: Dict[np.dtype, GpuBufferPool] = {}

	index_formats = {
		np.dtype(np.uint16): wgpu.IndexFormat.uint16,
		np.dtype(np.uint32): wgpu.IndexFormat.uint32,
	}

	def __init__(self, vertex_data: np.ndarray, indices: np.ndarray):
		self.index_count = indices.size
		self.vertex_dtype = np.dtype(vertex_data.dtype)
		self.index_dtype = np.dtype(indices.dtype)

		if self.index_dtype not in self.index_formats:
			raise TypeError(f"Unsupported index dtype: {self.index_dtype}")

		vertex_pool = self.vertex_buffers.get(self.vertex_dtype)
		if vertex_pool is None:
			vertex_pool = GpuBufferPool(self.vertex_dtype, wgpu.BufferUsage.VERTEX)
			self.vertex_buffers[self.vertex_dtype] = vertex_pool

		index_pool = self.index_buffers.get(self.index_dtype)
		if index_pool is None:
			index_pool = GpuBufferPool(self.index_dtype, wgpu.BufferUsage.INDEX)
			self.index_buffers[self.index_dtype] = index_pool

		self.vertex_buffer = vertex_pool.buffer
		self.vertex_range = vertex_pool.alloc(vertex_data)

		self.index_buffer = index_pool.buffer
		self.index_range = index_pool.alloc(indices)
		self.index_format = self.index_formats[self.index_dtype]


# textures --------------------------------------------------------------------

class Texture:
	def __init__(
		self,
		size: tuple[int, int] | tuple[int, int, int],
		*,
		format: str,
		usage: wgpu.TextureUsage,
		mip_level_count: int = 1,
		label: str | None = None,
	):
		if len(size) == 2:
			size = (*size, 1)
		self.size = size
		self.format = format
		self.usage = usage
		self.handle = RenderContext.device.create_texture(
			label=label or "",
			size=size,
			format=format,
			usage=usage,
			mip_level_count=mip_level_count,
		)

	def view(self, **kwargs) -> TextureView:
		return TextureView(self, self.handle.create_view(**kwargs))

	def color_attachment(
		self,
		*,
		clear: Color | None = None,
		store: bool = True,
	) -> ColorAttachment:
		return ColorAttachment(self.handle.create_view(), self.format, clear, store)

	def depth_attachment(
		self,
		*,
		clear: float | None = None,
		store: bool = True,
		read_only: bool = False,
	) -> DepthAttachment:
		return DepthAttachment(self.handle.create_view(), self.format, clear, store, read_only)


@dataclass(frozen=True)
class TextureView:
	texture: Texture
	handle: wgpu.GPUTextureView

	@property
	def format(self) -> str:
		return self.texture.format


class Sampler:
	def __init__(self, **descriptor: Any):
		self.handle = RenderContext.device.create_sampler(**descriptor)


# WGSL shader metadata --------------------------------------------------------


_SCALAR_TYPES = {
	"f32": (np.dtype(np.float32), 4, 4),
	"i32": (np.dtype(np.int32), 4, 4),
	"u32": (np.dtype(np.uint32), 4, 4),
}
_SCALAR_ALIAS = {"f": "f32", "i": "i32", "u": "u32"}


def _round_up(alignment: int, value: int) -> int:
	return (value + alignment - 1) // alignment * alignment


def _type_info(typ, *, uniform=False):
	typ = re.sub(r"\s+", "", typ)
	if typ in _SCALAR_TYPES:
		return _SCALAR_TYPES[typ]

	m = re.fullmatch(r"vec([234])(?:<([fiu]32)>|([fiu]))", typ)
	if m:
		n = int(m.group(1))
		dtype, _, size = _SCALAR_TYPES[m.group(2) or _SCALAR_ALIAS[m.group(3)]]
		return np.dtype((dtype, (n,))), 8 if n == 2 else 16, n * size

	m = re.fullmatch(r"mat([234])x([234])(?:<f32>|f)", typ)
	if m:
		cols, rows = map(int, m.groups())
		column, align, size = _type_info(f"vec{rows}f")
		stride = _round_up(align, size)
		if column.itemsize != stride:
			raise ValueError(f"{typ} requires padded matrix columns")
		return np.dtype((np.float32, (cols, rows))), align, cols * stride

	m = re.fullmatch(r"array<(.+),(\d+)>", typ)
	if m:
		dtype, align, size = _type_info(m.group(1), uniform=uniform)
		count = int(m.group(2))
		if uniform:
			align = max(16, align)
		stride = _round_up(align, size)
		if dtype.itemsize != stride:
			raise ValueError(f"{typ} requires padded array elements")
		return np.dtype((dtype, (count,))), align, count * stride

	raise ValueError(f"unsupported WGSL dtype {typ!r}")


def _struct_dtype(fields, *, uniform=False):
	fields = list(fields)
	if not uniform:
		return np.dtype([(name, _type_info(typ)[0]) for typ, name in fields])

	names, formats, offsets = [], [], []
	offset, struct_align = 0, 1
	for typ, name in fields:
		dtype, align, size = _type_info(typ, uniform=True)
		offset = _round_up(align, offset)
		names.append(name); formats.append(dtype); offsets.append(offset)
		offset += size
		struct_align = max(struct_align, align)
	return np.dtype({
		"names": names, "formats": formats, "offsets": offsets,
		"itemsize": _round_up(struct_align, offset),
	})

class _ShaderFileLoader(FileSystemLoader):
	def get_source(self, environment, template):
		source, filename, uptodate = super().get_source(environment, template)
		return source + "\n", filename, uptodate

class DefaultFalseDict(Dict):
	def __missing__(self, key):
		return False

class ShaderSource:
	_decl_pattern = re.compile(
		r"^(?P<attrs>(?:@(?:location\(\d+\)|position)[ \t\n]*)*)"
		r"(?:(?P<flat>flat)[ \t]+)?"
		r"(?P<qual>uniform|in|out|varying|texture|sampler)[ \t]+"
		r"(?P<name>[A-Za-z_]\w*)"
		r"(?:[ \t]*:[ \t]*(?P<type>[^;\n]+?))?"
		r"[ \t]*;[ \t]*(?:\n|$)",
		re.MULTILINE,
	)
	_func_pattern = re.compile(
		r"^(?P<attrs>(?:(?:@[A-Za-z_]\w*(?:\([^\n)]*\))?)[ \t\r\n]*)*)"
		r"fn[ \t]+(?P<name>[A-Za-z_]\w*)"
		r"[ \t]*\((?P<args>[^)]*)\)"
		r"(?:[ \t]*->[ \t]*(?P<return>[^\n{]+?))?"
		r"[ \t\r\n]*\{[ \t]*\n"
		r"(?P<body>[\s\S]*?)"
		r"^\}",
		re.MULTILINE,
	)
	_location_pattern = re.compile(r"@location\((\d+)\)")
	_long_comment_pattern = re.compile(r"/\*[\s\S]*?\*/")
	_comment_pattern = re.compile(r"//.*")

	def _strip_wgsl_comments(self, source: str) -> str:
		source = self._long_comment_pattern.sub("", source)
		return self._comment_pattern.sub("", source)

	def __init__(
		self,
		*,
		source: str | None = None,
		filepath: str | None = None,
		wgsl: str | None = None,
		basedir: str | None = None,
		features: Sequence[str] | None = None,
		params: Dict[str, Any] | None = None,
	):
		if filepath:
			basedir = basedir or os.path.dirname(os.path.abspath(filepath))
			with open(filepath, "r", encoding="utf8") as f:
				source = f.read()

		if source is None:
			raise ValueError("source or filepath is required")

		self.source = source
		self._basedir = basedir or "."

		self.uniforms = []
		self.ins = []
		self.varyings = []
		self.outs = []
		self._io = {"in": [], "varying": [], "out": []}
		self.textures = []
		self.samplers = []
		self.resources = []
		self.vertex_entry_points = []
		self.fragment_entry_points = []

		self.wgsl = wgsl or self.compile(features, params)


	@property
	def entry_points(self):
		return {"vertex": self.vertex_entry_points, "fragment": self.fragment_entry_points}

	@property
	def vertex_dtype(self):
		return _struct_dtype((typ, name) for _, typ, name in self.ins)

	@property
	def uniform_dtype(self):
		return _struct_dtype(self.uniforms, uniform=True)

	def compile(self,
		features: Sequence[str] | None = None,
		params: Dict[str, Any] | None = None
	) -> str:
		
		env = Environment(
			loader=_ShaderFileLoader(self._basedir),
			undefined=StrictUndefined,
			autoescape=False,
			keep_trailing_newline=True,
			trim_blocks=True,
			lstrip_blocks=True,
			line_statement_prefix="#",
		)

		feature_dict = DefaultFalseDict()
		if features is not None:
			feature_dict.update({ feature: True for feature in features })

		text = env.from_string(
			self.source
		).render(
			FEATURES=feature_dict,
			PARAMS=params or {},
		)
		text = self._strip_wgsl_comments(text)
		decls = list(self._decl_pattern.finditer(text))
		funcs = list(self._func_pattern.finditer(text))
		self._parse_declarations(decls)
		self._validate_functions(funcs)

		replacements = [(m.start(), m.end(), self._generate_prelude() + "\n\n" if i == 0 else "") for i, m in enumerate(decls)]
		replacements += [(m.start(), m.end(), self._generate_function(m)) for m in funcs]
		for start, end, replacement in sorted(replacements, reverse=True):
			text = text[:start] + replacement + text[end:]

		return  text + "\n"

	def _parse_declarations(self, decls):
		resources = 0
		for m in decls:
			attrs, flat, qual, name, typ = (m.group(k) for k in ("attrs", "flat", "qual", "name", "type"))
			
			if qual != "sampler" and typ is None:
				raise SyntaxError(f"{qual} {name!r} requires a type")

			if qual == "uniform":
				self.uniforms.append((typ, name))
				continue

			if qual in {"texture", "sampler"}:

				typ = typ or "sampler"
				if qual == "sampler" and typ not in {"sampler", "sampler_comparison"}:
					raise SyntaxError(f"invalid sampler type {typ!r}")

				(self.textures if qual == "texture" else self.samplers).append((typ, name))
				self.resources.append((1, resources, typ, name))
				resources += 1
				continue

			position = "@position" in attrs
			loc = self._location_pattern.search(attrs)
			loc = int(loc.group(1)) if loc else None
			if position and (qual != "varying" or loc is not None or flat):
				raise SyntaxError("@position must be a non-flat varying without @location")
			self._io[qual].append([loc, typ, name, bool(flat), position])

		for entries in self._io.values():
			used = {loc for loc, _, _, _, position in entries if loc is not None and not position}
			loc = 0
			for entry in entries:
				if entry[4] or entry[0] is not None:
					continue
				while loc in used:
					loc += 1
				entry[0] = loc
				used.add(loc)
				loc += 1

		self.ins = [tuple(e[:3]) for e in self._io["in"]]
		self.varyings = [tuple(e[:4]) for e in self._io["varying"]]
		self.outs = [tuple(e[:3]) for e in self._io["out"]]

	def _validate_functions(self, funcs):
		positions = sum(position for _, _, _, _, position in self._io["varying"])

		for m in funcs:
			if m.group(0).count("{") != m.group(0).count("}"):
				raise SyntaxError(f"unbalanced braces in function {m.group('name')!r}")

			attrs = m.group("attrs")
			vertex, fragment = "@vertex" in attrs, "@fragment" in attrs

			if not (vertex or fragment): continue

			if vertex:
				self.vertex_entry_points.append(m.group("name"))
			else:
				self.fragment_entry_points.append(m.group("name"))

	def _generate_prelude(self):
		sections = []
		
		if self.uniforms:
			fields = "\n".join(f"\t{name}: {typ}," for typ, name in self.uniforms)
			sections.append(f"struct _Uniforms {{\n{fields}\n}}\n\n@group(0) @binding(0)\nvar<uniform> _U: _Uniforms;")
		
		if self.resources:
			sections.append("\n\n".join(
				f"@group({group}) @binding({binding})\nvar {name}: {typ};"
				for group, binding, typ, name in self.resources
			))
		
		for name, entries in (
				("_VertexIn", self._io["in"]), ("_VertexOut", self._io["varying"]),
				("_FragmentIn", self._io["varying"]), ("_FragmentOut", self._io["out"]),
			):
			if entries:
				sections.append(self._generate_io_struct(name, entries))

		return "\n\n".join(sections)

	def _generate_io_struct(self, name, entries):
		fields = []
		
		for loc, typ, field, flat, position in entries:
			attrs = "@builtin(position)" if position else f"@location({loc})" + (" @interpolate(flat)" if flat else "")
			fields.append(f"\t{attrs} {field}: {typ},")

		return f"struct {name} {{\n" + "\n".join(fields) + "\n}"

	def _generate_function(self, m):
		attrs = m.group("attrs").strip()
		vertex, fragment = "@vertex" in attrs, "@fragment" in attrs
		if not (vertex or fragment):
			return self._rewrite_function_uniforms(m)

		body = self._rewrite_uniforms(m.group("body")).rstrip()
		if vertex:
			args, result, outputs = ("in: _VertexIn" if self.ins else ""), " -> _VertexOut", True
		else:
			args = "in: _FragmentIn" if self.varyings else ""
			outputs = bool(self.outs)
			result = " -> _FragmentOut" if outputs else ""

		lines = [attrs, f"fn {m.group('name')}({args}){result} {{"]
		if outputs:
			lines.append("\tvar out: " + ("_VertexOut;" if vertex else "_FragmentOut;"))
		if body:
			lines.append(body)
		if outputs:
			lines.append("\treturn out;")
		lines.append("}")
		return "\n".join(lines)

	def _rewrite_function_uniforms(self, m):
		text, body = m.group(0), m.group("body")
		start = m.start("body") - m.start()
		return text[:start] + self._rewrite_uniforms(body) + text[start + len(body):]

	def _rewrite_uniforms(self, text):
		for _, name in self.uniforms:
			text = re.sub(rf"(?<![A-Za-z0-9_.]){re.escape(name)}(?![A-Za-z0-9_])", f"_U.{name}", text)
		return text


class Shader:

	def __init__(
		self,
		label: str | None = None,
		*args,
		**kwargs
	):
		try:
			self.info = ShaderSource(**kwargs)
			self.module = RenderContext.device.create_shader_module(
				label=label or kwargs.get('filepath') or "shader",
				code=self.info.wgsl,
			)
		except Exception as ex:
			print(self.info.wgsl)
			raise ex from None

	def UniformBuffer(self) -> UniformBuffer:
		return UniformBuffer(self.info.uniform_dtype)

	def bind_group(self, group: int, **resources: Any) -> BindGroup:
		return BindGroup(self, group, resources)

	def entry(self, stage: str, name: str | None = None) -> str | None:
		entries = self.info.entry_points.get(stage, ())

		if name is not None:
			if name not in entries:
				raise ValueError( f"Shader has no {stage} entry point {name!r}" )
			return name

		if not entries:
			return None

		if len(entries) > 1:
			raise ValueError(
				f"Shader has multiple {stage} entry points; specify one explicitly"
			)

		return entries[0]


class UniformBuffer(GpuBuffer):
	def __init__(self, dtype:np.dtype):
		super().__init__(
			np.zeros(1, dtype=dtype),
			wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
		)


# pipelines / bindings --------------------------------------------------------

class BindGroup:
	def __init__(
		self,
		shader: Shader,
		group: int,
		resources: Dict[str, Any],
	):
		self.shader = shader
		self.group = group
		self.resources = dict(resources)
 
		self.bindings = {
			name: (group, binding)
			for group, binding, _, name in shader.info.resources
		}
		if shader.info.uniforms:
			self.bindings["uniforms"] = (0, 0)

		for name in resources:
			binding = self.bindings.get(name)
			if binding is None:
				raise KeyError(f"Shader has no binding named {name!r}")
			if binding[0] != group:
				raise ValueError(
					f"Binding {name!r} belongs to group {binding[0]}, not {group}"
				)
 
	def cache_key(self) -> tuple:
		return (
			self.group,
			tuple(
				(
					self.bindings[name][1],
					self._resource_key(resource),
				)
				for name, resource in self.resources.items()
			),
		)
 
	def entries(self) -> List[wgpu.BindGroupEntry]:
		return [
			wgpu.BindGroupEntry(
				binding=self.bindings[name][1],
				resource=self._resource(resource),
			)
			for name, resource in self.resources.items()
		]

	def _resource_key(self, resource: Any) -> tuple:
		if isinstance(resource, GpuBuffer):
			return ( id(resource), resource.generation, resource.content.nbytes )

		if isinstance(resource, (Texture, TextureView, Sampler)):
			return (id(resource),)

		return (id(resource),)

	def _resource(self, resource: Any) -> Any:
		if isinstance(resource, GpuBuffer):
			return wgpu.BufferBinding(
				buffer=resource.handle,
				offset=0,
				size=resource.content.nbytes,
			)
		if isinstance(resource, Texture):
			return resource.handle.create_view()

		if isinstance(resource, (TextureView, Sampler) ):
			return resource.handle
		
		return resource



class _PipelineVariant:
	def __init__(
		self,
		handle: wgpu.GPURenderPipeline | wgpu.GPUComputePipeline,
	):
		self.handle = handle
		self._bind_groups: Dict[tuple, wgpu.GPUBindGroup] = {}
		self._lock = Lock()

	def bind_group(self, bindings: BindGroup) -> wgpu.GPUBindGroup:
		key = bindings.cache_key()
		cached = self._bind_groups.get(key)
		if cached is not None:
			return cached

		with self._lock:
			cached = self._bind_groups.get(key)
			if cached is None:
				cached = RenderContext.device.create_bind_group(
					layout=self.handle.get_bind_group_layout(bindings.group),
					entries=bindings.entries(),
				)
				self._bind_groups[key] = cached

		return cached


class RenderPipeline:
	"""Render pipeline specification.

	A concrete GPURenderPipeline is cached for each combination of inferred vertex
	layouts and render-target formats. The user never has to specify those layouts.
	"""

	def __init__(
		self,
		shader: Shader,
		*,
		vertex_entry: str | None = None,
		fragment_entry: str | None = None,
		topology: str = wgpu.PrimitiveTopology.triangle_list,
		cull_mode: str = wgpu.CullMode.back,
		front_face: str = wgpu.FrontFace.ccw,
		depth_test: str | None = wgpu.CompareFunction.less,
		depth_write: bool = True,
		depth_bias: int = 0,
		depth_bias_slope_scale: float = 0.0,
		blend: wgpu.BlendState | None = None,
		label: str | None = None,
	):
		self.shader = shader
		self.vertex_entry = shader.entry("vertex", vertex_entry)
		self.fragment_entry = shader.entry("fragment", fragment_entry)
		self.topology = topology
		self.cull_mode = cull_mode
		self.front_face = front_face
		self.depth_test = depth_test
		self.depth_write = depth_write
		self.depth_bias = depth_bias
		self.depth_bias_slope_scale = depth_bias_slope_scale
		self.blend = blend
		self.label = label

		if self.vertex_entry is None:
			raise ValueError("RenderPipeline shader has no @vertex entry point")

		self._cache: Dict[tuple, _PipelineVariant] = {}
		self._cache_lock = Lock()


	def get_variant(
		self,
		render_pass: RenderPass,
		buffers: Sequence[tuple[np.dtype, str]],
	) -> _PipelineVariant:
		key = (
			tuple(
				(repr(np.dtype(dtype).descr), step_mode)
				for dtype, step_mode in buffers
			),
			render_pass.color_formats,
			render_pass.depth_format,
		)

		cached = self._cache.get(key)
		if cached is not None:
			return cached

		with self._cache_lock:
			cached = self._cache.get(key)
			if cached is None:
				cached = _PipelineVariant(
					self._make_pipeline(render_pass, buffers)
				)
				self._cache[key] = cached

		return cached

	def _make_pipeline(
		self,
		render_pass: RenderPass,
		buffers: Sequence[tuple[np.dtype, str]],
	) -> wgpu.GPURenderPipeline:
		provided_vertex_input = set()
		layouts = []

		vertex_locations = {
			name: location for location, _, name in self.shader.info.ins
		}

		for dtype, step_mode in buffers:
			if dtype.fields is None:
				raise TypeError(f"Vertex buffer dtype must be structured, got {dtype}")

			attributes = []

			for name, (field_dtype, offset) in dtype.fields.items():

				if name in provided_vertex_input:
					raise ValueError(
						f"Vertex attribute {name!r} is provided by multiple buffers"
					)

				field_dtype = np.dtype(field_dtype)
				location = vertex_locations.get(name)
				shader_dtype = self.shader.info.vertex_dtype.fields[name][0]

				field_size = int(np.prod(field_dtype.shape)) if field_dtype.shape else 1
				shader_size = int(np.prod(shader_dtype.shape)) if shader_dtype.shape else 1

				if (
					field_size != shader_size
					or field_dtype.base.kind != shader_dtype.base.kind
				):
					raise TypeError(
						f"Vertex attribute {name!r} has dtype {field_dtype}; "
						f"shader expects {shader_dtype}"
					)

				attributes.append(wgpu.VertexAttribute(
					shader_location=location,
					offset=offset,
					format=dtype_to_vertex_format(field_dtype),
				))
				provided_vertex_input.add(name)

			layouts.append(wgpu.VertexBufferLayout(
				array_stride=dtype.itemsize,
				step_mode=step_mode,
				attributes=attributes,
			))

		missing = vertex_locations.keys() - provided_vertex_input
		if missing:
			raise ValueError(
				f"Vertex shader {self.vertex_entry!r} requires attributes "
				f"not provided by the draw buffers: {', '.join(sorted(missing))}"
			)

		fragment = None

		if render_pass.color_formats:
			if self.fragment_entry is None:
				raise ValueError(
					"Rendering to a color target requires an @fragment entry point"
				)

			fragment = wgpu.FragmentState(
				module=self.shader.module,
				entry_point=self.fragment_entry,
				targets=[
					wgpu.ColorTargetState(
						format=format,
						blend=self.blend,
					)
					for format in render_pass.color_formats
				],
			)

		depth_stencil = None

		if self.depth_test is not None:
			if render_pass.depth_format is None:
				raise ValueError(
					"Pipeline has depth testing enabled but pass has no depth target"
				)

			depth_stencil = wgpu.DepthStencilState(
				format=render_pass.depth_format,
				depth_write_enabled=self.depth_write,
				depth_compare=self.depth_test,
				depth_bias=self.depth_bias,
				depth_bias_slope_scale=self.depth_bias_slope_scale,
			)

		return RenderContext.device.create_render_pipeline(
			label=self.label or "",
			layout="auto",
			vertex=wgpu.VertexState(
				module=self.shader.module,
				entry_point=self.vertex_entry,
				buffers=layouts,
			),
			primitive=wgpu.PrimitiveState(
				topology=self.topology,
				cull_mode=self.cull_mode,
				front_face=self.front_face,
			),
			depth_stencil=depth_stencil,
			fragment=fragment,
		)



class ComputePipeline:
	def __init__(
		self,
		shader: Shader,
		*,
		entry: str | None = None,
		label: str | None = None,
	):
		self.shader = shader
		self.entry = shader.entry("compute", entry)

		if self.entry is None:
			raise ValueError("ComputePipeline shader has no @compute entry point")

		handle = RenderContext.device.create_compute_pipeline(
			label=label or "",
			layout="auto",
			compute=wgpu.ProgrammableStage(
				module=shader.module,
				entry_point=self.entry,
			),
		)
		self._variant = _PipelineVariant(handle)

	@property
	def handle(self) -> wgpu.GPUComputePipeline:
		return self._variant.handle



# compatibility-ish helpers ---------------------------------------------------

def create_depth_framebuffer(
	width: int,
	height: int,
	format: str = wgpu.TextureFormat.depth24plus,
) -> Texture:
	return Texture(
		(width, height),
		format=format,
		usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
	)


def create_depth_sampler() -> Sampler:
	return Sampler(
		address_mode_u=wgpu.AddressMode.clamp_to_edge,
		address_mode_v=wgpu.AddressMode.clamp_to_edge,
		address_mode_w=wgpu.AddressMode.clamp_to_edge,
		compare=wgpu.CompareFunction.less_equal,
		min_filter=wgpu.FilterMode.linear,
		mag_filter=wgpu.FilterMode.linear,
	)
