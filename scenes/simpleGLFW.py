from shader_util import *
from time import time
from math import cos, sin

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

l = np.array([30.0, 30.0, 25.0]) - np.array([0.0, 0.0, -20.0])
l = l / np.linalg.norm(l)
light_dir = l.astype(np.float32)


# seems to use a weird color space
renderpass = RenderContext.RenderPass(camera = camera, clear_color = (0.02, 0.02, 0.03, 1.0))

vertices, indices = load_gltf_first_mesh_interleaved('scenes/resources/rooftop_utility_pole.glb')
shader = RenderContext.Shader(ShaderSource(filepath='scenes/shaders/simple.shader'))
mesh = shader.Mesh(vertices, indices)
uniformBuffer = shader.UniformBuffer()

print('init done')

scale = 10.0
start_t = time()
frame_start = start_t
fps_frames = 0
while not RenderContext.WindowShouldClose():

    # time since start of the simulation 
    now = time()
    elapsed = now - start_t
    
    # simple FPS to console
    fps_frames += 1
    if now - frame_start >= 1.0:
        print(f"fps {fps_frames}")
        frame_start = now
        fps_frames = 0

    # orbit update
    cam_ang = elapsed * 0.5
    camera.position = (
        cos(cam_ang) * camera_dist,
        camera.position[1],
        sin(cam_ang) * camera_dist
    )

    with renderpass as rp:
        uniformBuffer.uniforms['uView'] = renderpass.view
        uniformBuffer.uniforms['uProj'] = renderpass.projection
        uniformBuffer.uniforms['uModel'] = Mat4.from_scale([scale, scale, scale], dtype=np.float32)
        # the padding / std430 packing does not seem to not work as expected
        uniformBuffer.uniforms['uLightDir'] = (*light_dir, 0.0)
        uniformBuffer.uniforms['uTint'] = (0.7, 0.5, 0.3, 1.0)
        uniformBuffer.write_uniforms()
        # TODO : a better api would be renderpass.draw(Mesh)
        mesh.draw(rp, uniformBuffer.bind_group)
