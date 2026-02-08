from shader_util import *
from scenes.cube_wgpu import *

WINDOW_W, WINDOW_H = 1800, 900
TITLE = "glfw + wgpu"

RenderContext.InitWindow(WINDOW_W, WINDOW_H, TITLE)

draw_frame = setup_drawing_sync(RenderContext.canvas)

camera_dist = 30.0
camera = Camera(
    position=(-20.0, 70.0, 25.0),
    target=(0.0, 10.0, 0.0),
    up=(0.0, 1.0, 0.0),
    fovy_deg=60.0,
    near=0.1, far=1000.0
)

# expects layout(location=...) everywhere
# + ShaderSource does not support uniform structs
#RenderContext.ShaderPipeline(ShaderSource('tests/test.shader'))

rpass = RenderContext.RenderPass(camera = camera)

while not RenderContext.windowShouldClose():
	draw_frame()
	with rpass:
		pass