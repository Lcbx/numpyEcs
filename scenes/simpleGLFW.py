from shader_util import *
from scenes.cube_wgpu import *

WINDOW_W, WINDOW_H = 1800, 900
TITLE = "glfw + wgpu"

context, shouldClose = InitWindow(WINDOW_W, WINDOW_H, TITLE)

draw_frame = setup_drawing_sync(context)

#adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
#device = adapter.request_device_sync(
#	label="Cube Example device",
#	required_limits=None,
#)

while shouldClose():
	draw_frame()
	#device._poll(True)