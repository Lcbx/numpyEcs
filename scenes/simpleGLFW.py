from shader_util import *

#from scenes.cube_wgpu import *

WINDOW_W, WINDOW_H = 1800, 900
TITLE = "glfw + wgpu"

RenderContext.InitWindow(WINDOW_W, WINDOW_H, TITLE)

camera_dist = 30.0
camera = Camera(
    position=(-20.0, 70.0, 25.0),
    target=(0.0, 10.0, 0.0),
    up=(0.0, 1.0, 0.0),
    fovy_deg=60.0,
    near=0.1, far=1000.0
)

rpass = RenderContext.RenderPass(camera = camera)

vertices, indices = load_gltf_first_mesh_interleaved('scenes/resources/rooftop_utility_pole.glb')
index_count = int(indices.size)

shader = RenderContext.ShaderPipeline(ShaderSource('scenes/shaders/simple.shader'), vertices, True)

print('init done')

while not RenderContext.windowShouldClose():
	with rpass:
		pass