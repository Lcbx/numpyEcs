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


RenderContext.InitWindow(WINDOW_W, WINDOW_H, TITLE
)
#, target_fps=120)
#, target_fps=-1)
#RenderContext.capture_mouse()

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
    cp = camera.position
    y_factor = 1.0 if cp.y < 15.0 else 3.0
    #print(f'{y_factor=}')
    camera.position = Vec3( (cp.x, cp.y - y_factor * yoff, cp.z) )
    return True

RenderContext.event_handlers['mouse_scroll'].append(scroll_callback)

#import random as rd
#draw_cube( (0.7, 0.5, 0.3, 1.0), (1,1,1), (0.5, 0.5, 0.5, 1.0))

fps_frames = 0
start_t = getTime()
fps_print_timestamp = start_t
while RenderContext.WindowLoop():

    now = RenderContext.frame_start

    # simple FPS to console
    fps_frames += 1
    if now - fps_print_timestamp >= 1.0:
        print(f"fps {fps_frames}")
        fps_frames = 0
        fps_print_timestamp = now
        #instance_data[1]["uModel"] = scale_mat @ Mat4.from_translation([rd.random()*15.0, 0.0, rd.random()*15.0])
        #mesh.instance_buffer.upload()


    # orbit update (based on time wince start)
    elapsed = now - start_t
    cam_ang = elapsed * 0.5
    camera.position = Vec3( (
        cos(cam_ang) * camera_dist,
        camera.position.y,
        sin(cam_ang) * camera_dist
    ) )


    with renderpass as rp:
        uniformBuffer.content['uView'] = renderpass.view
        uniformBuffer.content['uProj'] = renderpass.projection
        uniformBuffer.content['uLightDir'] = light_dir
        #uniformBuffer.content['uTint'] = (0.7, 0.5, 0.3, 1.0)
        #uniformBuffer.content['uModel'] = Mat4.from_scale([scale, scale, scale], dtype=np.float32)
        uniformBuffer.upload()
        rp.draw(mesh, shader, uniformBuffer)

