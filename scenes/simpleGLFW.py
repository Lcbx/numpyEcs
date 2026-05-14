from shader_util import *
from math import cos, sin

WINDOW_W, WINDOW_H = 1800, 900
TITLE = "glfw + wgpu"


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


RenderContext.InitWindow(WINDOW_W, WINDOW_H, TITLE)#, vsync=False)
#RenderContext.capture_mouse()
#RenderContext.setup(highpower=False)

# seems to use a weird color space
renderpass = RenderContext.RenderPass(camera = camera, clear_color = (0.02, 0.02, 0.03, 1.0))
shader = RenderContext.Shader(filepath='scenes/shaders/simple.shader')
vertices, indices = load_gltf_first_mesh_interleaved('scenes/resources/rooftop_utility_pole.glb')
scale = 10.0
Vec3_type = np.dtype( (np.float32, (3,)) )
Vec4_type = np.dtype( (np.float32, (4,)) )
Mat4_type = np.dtype( (np.float32, (4,4)) )
instance_data_type = make_std430_dtype([
    ("uModel", Mat4_type), # NOTE: 4x4 float32 mat is a lot of memory if we just needed position, orientation, scale 
    ("uTint",  Vec4_type)
])
instance_data = np.zeros(10, instance_data_type)
scale_mat = Mat4.from_scale([scale, scale, scale])
instance_data[0]["uModel"] = scale_mat
instance_data[0]["uTint"] = (0.7, 0.5, 0.3, 1.0)
instance_data[1]["uModel"] = scale_mat @ Mat4.from_translation([15.0, 0.0, 15.0]) 
instance_data[1]["uTint"] = (0.3, 0.5, 0.7, 1.0)
mesh = Mesh(vertices, indices, instance_data=instance_data)
#mesh = Mesh(vertices, indices)
uniformBuffer = shader.UniformBuffer()


print('init done')

def clamp(val, val_min, val_max):
    return min(max(val, val_min), val_max)

def scroll_callback(xoff, yoff):
    global camera_dist, camera
    camera_dist = clamp(camera_dist - 5.0 * yoff, 5.0, 100.0)
    y_factor = 1.0 if camera.position[1] < 15.0 else 3.0
    camera.position = tuple(
        ( val - y_factor * yoff if i == 1 else val)
        for i, val in enumerate(camera.position)
    )
    return True

RenderContext.event_handlers['mouse_scroll'].append(scroll_callback)


start_t = time.monotonic()
frame_start = start_t
fps_frames = 0
while RenderContext.WindowLoop():

    # time since start of the simulation 
    now = time.monotonic()
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
        uniformBuffer.content['uView'] = renderpass.view
        uniformBuffer.content['uProj'] = renderpass.projection
        uniformBuffer.content['uLightDir'] = light_dir
        #uniformBuffer.content['uTint'] = (0.7, 0.5, 0.3, 1.0)
        #uniformBuffer.content['uModel'] = Mat4.from_scale([scale, scale, scale], dtype=np.float32)
        uniformBuffer.upload()
        rp.draw(mesh, shader, uniformBuffer)
