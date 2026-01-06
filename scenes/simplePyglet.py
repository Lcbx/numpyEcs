import math
import time
import ctypes
import numpy as np

import pyglet
from pyglet.window import key, mouse
from pyglet.gl import (
    glEnable, glClearColor, glViewport,
    GL_DEPTH_TEST, GL_TRIANGLES
)
from pyglet.graphics import Batch
from pyglet.graphics.shader import Shader, ShaderProgram

import shader_util as su
from pyrr import Matrix44, Vector3

# ----------------------------
# Config
# ----------------------------
WINDOW_W, WINDOW_H = 1800, 900
TITLE = "pyglet + ShaderProgram + glTF"
MODEL_PATH = 'scenes/resources/rooftop_utility_pole.glb'

# ----------------------------
# GLSL (unchanged)
# ----------------------------
VERT_SRC = r"""
#version 430 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform vec3 uLightDir;

out vec3 vNormalWS;
out vec3 vLightDirWS;
out vec2 vUV;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vNormalWS = mat3(uModel) * aNormal;
    vLightDirWS = normalize(uLightDir);
    vUV = aUV;
    gl_Position = uProj * uView * worldPos;
}
"""

FRAG_SRC = r"""
#version 430 core
in vec3 vNormalWS;
in vec3 vLightDirWS;
in vec2 vUV;

out vec4 FragColor;

void main() {
    vec3 N = normalize(vNormalWS);
    float NdotL = max(dot(N, vLightDirWS), 0.0);
    vec3 base = vec3(0.82, 0.71, 0.55); // BEIGE-ish
    vec3 color = base * (0.2 + 0.8 * NdotL);
    FragColor = vec4(color, 1.0);
}
"""

def build_shader_program():
    vert = Shader(VERT_SRC, 'vertex')
    frag = Shader(FRAG_SRC, 'fragment')
    return ShaderProgram(vert, frag)

def load_gltf_first_mesh(program, batch, glb_path: str):
    ( pos, nor, uv, idx ) = su.load_gltf_first_mesh(glb_path)
    

    pos_flat = pos.reshape(-1).astype('f')
    nor_flat = nor.reshape(-1).astype('f')
    uv_flat  = uv.reshape(-1).astype('f')
    indices = idx.astype(int).reshape(-1)

    vertex_count = pos.shape[0]

    vlist = program.vertex_list_indexed(
        vertex_count,
        GL_TRIANGLES,
        indices,
        batch=batch,
        aPos=('f', pos_flat),
        aNormal=('f', nor_flat),
        aUV=('f', uv_flat),
    )
    return vlist

# ----------------------------
# Camera
# ----------------------------
class Camera:
    def __init__(self, position, target, up, fovy_deg, near=0.1, far=1000.0):
        self.position = Vector3(position)
        self.target = Vector3(target)
        self.up = Vector3(up)
        self.fovy_deg = fovy_deg
        self.near = near
        self.far = far

    def view(self):
        return Matrix44.look_at(self.position, self.target, self.up, dtype=np.float32)

    def proj(self, aspect):
        return Matrix44.perspective_projection(
            self.fovy_deg, aspect, self.near, self.far, dtype=np.float32
        )

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

# ----------------------------
# Application
# ----------------------------
def main():
    global camera_dist, camera, orbit

    #config = pyglet.gl.Config(
    #    double_buffer=True,
    #    major_version=4,
    #    minor_version=3,
    #    samples = 1,
    #)
    window = pyglet.window.Window(
        width=WINDOW_W,
        height=WINDOW_H,
        caption=TITLE,
        #config=config,
        resizable=False,
        vsync=False
    )
    #window.vsync = False

    # Minimal explicit GL state (everything else via pyglet abstractions)
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.15, 0.16, 0.19, 1.0)

    # Shader program & batch
    program = build_shader_program()
    batch = Batch()

    # Model transform
    scale = 5.0
    model_mat = Matrix44.from_scale([scale, scale, scale], dtype=np.float32)

    # Light direction (same as original)
    l = np.array([30.0, 30.0, 25.0]) - np.array([0.0, 0.0, -20.0])
    l = l / np.linalg.norm(l)
    light_dir = l.astype(np.float32)

    # Create vertex list in a batch (this hides VBO/VAO/EBO stuff)
    mesh_vlist = load_gltf_first_mesh(program, batch, MODEL_PATH)

    # ---------------- Events ----------------
    @window.event
    def on_resize(width, height):
        # Keep viewport in sync; this is one of the few explicit GL calls
        glViewport(0, 0, width, height)

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

        # Bind shader, set uniforms the pyglet way, then draw batch
        program.bind()

        # ShaderProgram accepts numpy arrays / sequences for uniforms.
        program['uModel'] = model_mat.reshape(-1).astype('f')
        program['uView']  = view.reshape(-1).astype('f')
        program['uProj']  = proj.reshape(-1).astype('f')
        program['uLightDir'] = light_dir

        batch.draw()
        ShaderProgram.unbind()

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
        return pyglet.window.Window.on_close(window)

    # refresh rate, 1/60 = 60hz, 0 -> afap
    pyglet.app.run(0) 

main()
