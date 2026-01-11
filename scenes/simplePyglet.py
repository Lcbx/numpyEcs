import math
import time

from shader_util import *


# ----------------------------
# Config
# ----------------------------
WINDOW_W, WINDOW_H = 1800, 900
TITLE = "pyglet + ShaderProgram + glTF"
model_root = 'scenes/resources/'


camera_dist = 30.0
camera = Camera(
    position=(-20.0, 70.0, 25.0),
    target=(0.0, 10.0, 0.0),
    up=(0.0, 1.0, 0.0),
    fovy_deg=60.0,
    near=0.1, far=1000.0
)
orbit = True

start_t = time.time()
frame_start = 0.0
fps_frames = 0

window = RenderContext.InitWindow(WINDOW_W, WINDOW_H, TITLE)

EnableDepth()
EnableCullFace()
setClearColor(0.15, 0.16, 0.19, 1.0)

# Shader program & batch
program = build_shader_program('scenes/shaders/pyglet.shader')

# Light direction (same as original)
l = np.array([30.0, 30.0, 25.0]) - np.array([0.0, 0.0, -20.0])
l = l / np.linalg.norm(l)
light_dir = l.astype(np.float32)


scale = 5.0
mesh = load_gltf_meshes(program, model_root + 'rooftop_utility_pole.glb')[0]
model_mat = Mat4.from_translation( (0, 15, 5) ) * Mat4.from_scale( (scale, scale, scale) )
mesh['uTint'] = (0.2, 0.5, 0.2, 1.0)
mesh['uModel'] = model_mat
vMesh = mesh.draw(transform=model_mat) # use default batch instead

heightmap = load_gltf_meshes(program, model_root + 'heightmap_mesh.glb')
for h in heightmap:
    h['uTint'] = (0.82, 0.71, 0.55, 1.0)
    h.draw()


@window.event
def on_key_press(symbol, modifiers):
    global orbit
    if symbol == key.O:
        orbit = not orbit

@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    global camera_dist, camera
    scrollspeed = 3.0
    yoff = scroll_y
    mw = scrollspeed * yoff
    if mw != 0 and (camera.position.y > scrollspeed + 0.5 or mw > 0.0):
        tar = np.array([camera.target.x, camera.target.y, camera.target.z], dtype=np.float32)
        cam = np.array([camera.position.x, camera.position.y, camera.position.z], dtype=np.float32)
        camera_dist -= mw * 0.3
        cam[1] += (tar[1] - cam[1]) / (abs(cam[1]) + 0.1) * mw
        camera.position = Vector3(cam)

@window.event
def on_draw():
    global frame_start, fps_frames, camera

    now = time.time()
    elapsed = now - start_t

    if orbit:
        cam_ang = elapsed * 0.5
        camera.position = Vector3((
            math.cos(cam_ang) * camera_dist,
            camera.position.y,
            math.sin(cam_ang) * camera_dist
        ))

    with RenderContext(shader=program, camera=camera):
        window.clear()
        program['uLightDir'] = light_dir

        #meshBatch.draw() # already in default batch

        vCubes = Cubes(program,
            ((2,0,0), (-1,0,0)),    # positions
            ((3,1,2), (1,1,1)),     # sizes
            (0.12, 0.31, 0.65, 1.0) # color
        ).draw()
    vCubes.delete() # delete previous cubes or we add more each frame 

    fps_frames += 1
    if now - frame_start >= 1.0:
        print(f"fps {fps_frames}")
        frame_start = time.time()
        fps_frames = 0

# refresh rate, 1/60 = 60hz, 0 -> afap
run(0)
