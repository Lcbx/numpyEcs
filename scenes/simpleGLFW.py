from RenderContext import *
from Utils import *
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
#)
#, target_fps=120)
, target_fps=-1)
#RenderContext.capture_mouse()

#print(RenderContext.resources)

# seems to use a weird color space
renderpass = RenderContext.RenderPass(camera = camera, clear_color = (0.02, 0.02, 0.03, 1.0))
shader = RenderContext.Shader(filepath='scenes/shaders/simple.shader')
vertices, indices = load_gltf_first_mesh_interleaved('scenes/resources/rooftop_utility_pole.glb')
scale = 10.0
instance_data = np.zeros(10, instance_dtype)
instance_data[0]["iPosition"] = Vec3([15.0, 0.0, 15.0])
instance_data[0]["iRotation"] = Quaternion()
instance_data[0]["iScale"] = [scale] * 4 # only 3 of are used
instance_data[0]["iTint"] = pack_rgba8_srgb(Vec4([0.3, 0.5, 0.7, 1.0]))
print(instance_data[0]["iTint"])
mesh = Mesh(vertices, indices)
mesh.set_instances(instance_data)
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


import psutil # NOTE: this is not a built-in lib
process = psutil.Process(os.getpid())

import tracemalloc
tracemalloc.start()

import gc
_gc_start = None

print(f'{gc.get_threshold()=}')
gc.set_threshold(1000, 30, 2) # deflt 2000, 10, 10


def gc_probe(phase, info):
    global _gc_start

    if phase == "start":
        _gc_start = getTime()
    elif phase == "stop":
        dt_ms = (getTime() - _gc_start) * 1000
        if not any(info.values()): return
        print(
            f"GC gen={info['generation']} "
            f"collected={info['collected']} "
            f"uncollectable={info['uncollectable']} "
            f"time={dt_ms:.3f} ms"
        )
gc.callbacks.append(gc_probe)


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

        #print(WatchTimer.capture())

        rss = process.memory_info().rss
        print(f"Process RSS: {rss / 1024 / 1024:.1f} MiB")
        
        #snapshot = tracemalloc.take_snapshot()
        current, peak = tracemalloc.get_traced_memory()
        print(f"traced current: {current / 1024 / 1024:.2f} MiB")
        print(f"traced peak:    {peak / 1024 / 1024:.2f} MiB")

        #print("gc stats")
        #for s in gc.get_stats():
        #    print(s)
        #for i in range(3):
        #    print("generation", i, len(gc.get_objects(i)))

        print("")


    # orbit update (based on time wince start)
    elapsed = now - start_t
    cam_ang = elapsed * 0.5
    camera.position = Vec3( (
        cos(cam_ang) * camera_dist,
        camera.position.y,
        sin(cam_ang) * camera_dist
    ) )


    with WatchTimer("draw"):
        with renderpass as rp:
            with WatchTimer("draw_cube"):
                draw_cube(  (3, 5, -3), (2,2,2), (0.5, 0.5, 0.5, 1.0), Quaternion.from_eulers([0.6,0.0,0.0]) )
                draw_cube( (-3, 5, -3), (1,5,1), (0.5, 0.5, 0.5, 1.0))

            with WatchTimer("update_uniform"):
                uniformBuffer.content['uView'] = renderpass.view
                uniformBuffer.content['uProj'] = renderpass.projection
                uniformBuffer.content['uLightDir'] = light_dir
                #uniformBuffer.content['uTint'] = (0.7, 0.5, 0.3, 1.0)
                #uniformBuffer.content['uModel'] = Mat4.from_scale([scale, scale, scale], dtype=np.float32)
                uniformBuffer.upload()

            with WatchTimer("draw_model"):
                mesh.draw(rp, shader, uniformBuffer)
            with WatchTimer("flush_cube"):
                flush_cubes(rp, shader, uniformBuffer)

