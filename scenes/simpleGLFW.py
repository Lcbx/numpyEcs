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

renderpass = RenderContext.RenderPass(camera = camera)

vertices, indices = load_gltf_first_mesh_interleaved('scenes/resources/rooftop_utility_pole.glb')
shader = RenderContext.Shader(ShaderSource(filepath='scenes/shaders/simple.shader'))
mesh = shader.Mesh(vertices, indices)
uniformBuffer = shader.UniformBuffer()

print('init done')

while not RenderContext.windowShouldClose():
    # TODO : update uniformBuffer (camera data is in renderpass)
    # uniformBuffer.uniforms['name'] = value
	with renderpass as rp:
        # TODO : a better api would be renderpass.draw(Mesh)
		mesh.draw(rp, uniformBuffer.bind_group)