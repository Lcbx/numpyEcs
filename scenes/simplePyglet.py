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


def build_shader_program():
    source = BetterShaderSource('scenes/shaders/pyglet.shader')
    vert = Shader(source.vertex_glsl, 'vertex')
    frag = Shader(source.fragment_glsl, 'fragment')
    return Program(vert, frag)


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
program = build_shader_program()
batch = Batch()

# Model transform
scale = 5.0
model_mat = Mat4.from_scale([scale, scale, scale])

# Light direction (same as original)
l = np.array([30.0, 30.0, 25.0]) - np.array([0.0, 0.0, -20.0])
l = l / np.linalg.norm(l)
light_dir = l.astype(np.float32)

# Create vertex list in a batch (this hides VBO/VAO/EBO stuff)
mesh_vlist = load_gltf_first_mesh(program, batch, MODEL_PATH)
addCube(program, batch, (2,0,0), (3,1,2) )

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

    # per-frame timing
    now = time.time()
    elapsed = now - start_t

    if orbit:
        cam_ang = elapsed * 0.5
        camera.position = Vector3((
            math.cos(cam_ang) * camera_dist,
            camera.position.y,
            math.sin(cam_ang) * camera_dist
        ))

    # Clears color + depth using pyglet helper
    window.clear()

    aspect = window.width / window.height
    view = camera.view()
    proj = camera.proj(aspect)

    program.bind()
    program['uModel'] = model_mat.reshape(-1).astype('f')
    program['uView']  = view.reshape(-1).astype('f')
    program['uProj']  = proj.reshape(-1).astype('f')
    program['uLightDir'] = light_dir
    program['uTint'] = (0.82, 0.71, 0.55, 1.0)

    batch.draw()

    # simple FPS to console
    fps_frames += 1
    if now - frame_start >= 1.0:
        print(f"fps {fps_frames}")
        frame_start = time.time()
        fps_frames = 0

@window.event
def on_close():
    mesh_vlist.delete()
    program.delete()
    return Window.on_close(window)

# refresh rate, 1/60 = 60hz, 0 -> afap
run(0)
