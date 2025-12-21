import math
import time
import ctypes

from pyglet.app import run
from pyglet.window import  Window, key, mouse
import pyglet.gl as gl
from shader_util import *


# ----------------------------
# Config
# ----------------------------
WINDOW_W, WINDOW_H = 1800, 900
TITLE = "pyglet + ShaderProgram + glTF"
MODEL_PATH = 'scenes/resources/rooftop_utility_pole.glb'


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

window = InitWindow(WINDOW_W, WINDOW_H, TITLE)

# Minimal explicit GL state (everything else via pyglet abstractions)
gl.glEnable(gl.GL_DEPTH_TEST)
gl.glClearColor(0.15, 0.16, 0.19, 1.0)

# Shader program & batch
program = build_shader_program('scenes/shaders/pyglet.shader')
batch = Batch()

# Light direction (same as original)
l = np.array([30.0, 30.0, 25.0]) - np.array([0.0, 0.0, -20.0])
l = l / np.linalg.norm(l)
light_dir = l.astype(np.float32)


scale = 5.0
model_mat = Mat4.from_scale( (scale, scale, scale) )

cube1 = Cube(program, (2,0,0), (3,1,2) )
cube1.draw(batch)

mesh = load_gltf_first_mesh(program, MODEL_PATH)
#mesh['uModel'] = Mat4.from_translation( (0, 15, 5) ) * model_mat
#mesh['uModel'] = model_mat * model_mat
mesh['uTint'] = (0.2, 0.5, 0.2, 1.0)
mesh.draw(batch)

cube2 = Cube(program, (-1,0,0), (1,1,1) )
cube2.draw(batch)

# ---------------- Events ----------------
@window.event
def on_resize(width, height):
    # Keep viewport in sync; this is one of the few explicit GL calls
    gl.glViewport(0, 0, width, height)

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

    window.clear()
    aspect = window.width / window.height
    view = camera.view()
    proj = camera.proj(aspect)

    program.bind()
    program['uModel'] = model_mat
    program['uView']  = view
    program['uProj']  = proj
    program['uLightDir'] = light_dir
    program['uTint'] = (0.82, 0.71, 0.55, 1.0)

    batch.draw()

    fps_frames += 1
    if now - frame_start >= 1.0:
        print(f"fps {fps_frames}")
        frame_start = time.time()
        fps_frames = 0

@window.event
def on_close():
    program.delete()
    return Window.on_close(window)

# refresh rate, 1/60 = 60hz, 0 -> afap
run(0)
