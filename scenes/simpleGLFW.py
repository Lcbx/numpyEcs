import sys, ctypes, math, time
import numpy as np
import shader_util as su

import glfw
from OpenGL.GL import *
from pyrr import Matrix44, Vector3, vector, vector3

# ----------------------------
# Config
# ----------------------------
WINDOW_W, WINDOW_H = 1800, 900
TITLE = "OpenGL 4.3 Window"
MODEL_PATH = 'scenes/resources/rooftop_utility_pole.glb'

# ----------------------------
# Minimal shader (keep or replace with your BetterShader)
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
    // model space -> world
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    // normal (ignore non-uniform scale here for brevity)
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

def compile_shader(src, stype):
    sid = glCreateShader(stype)
    glShaderSource(sid, src)
    glCompileShader(sid)
    ok = glGetShaderiv(sid, GL_COMPILE_STATUS)
    if not ok:
        log = glGetShaderInfoLog(sid).decode()
        raise RuntimeError(f"Shader compile error:\n{log}")
    return sid

def link_program(vs, fs):
    pid = glCreateProgram()
    glAttachShader(pid, vs)
    glAttachShader(pid, fs)
    glLinkProgram(pid)
    ok = glGetProgramiv(pid, GL_LINK_STATUS)
    if not ok:
        log = glGetProgramInfoLog(pid).decode()
        raise RuntimeError(f"Program link error:\n{log}")
    glDetachShader(pid, vs)
    glDetachShader(pid, fs)
    glDeleteShader(vs)
    glDeleteShader(fs)
    return pid

# If you want to keep your BetterShader:
# - Make it return a linked GL program id (GLuint) named program
# - Make sure it exposes uniform locations uModel/uView/uProj/uLightDir
# Then replace the two lines below with your loader.

def build_default_program():
    vs = compile_shader(VERT_SRC, GL_VERTEX_SHADER)
    fs = compile_shader(FRAG_SRC, GL_FRAGMENT_SHADER)
    return link_program(vs, fs)

class GLMesh:
    def __init__(self, vao, vbo, nbo, tbo, ebo, index_count):
        self.vao = vao
        self.vbo = vbo
        self.nbo = nbo
        self.tbo = tbo
        self.ebo = ebo
        self.index_count = index_count

    def destroy(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        glDeleteBuffers(1, [self.nbo])
        glDeleteBuffers(1, [self.tbo])
        glDeleteBuffers(1, [self.ebo])

def load_gltf_first_mesh(glb_path: str) -> GLMesh:
    ( pos, nor, uv, idx ) = su.load_gltf_first_mesh(glb_path)

    # Build VAO/VBOs
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, pos.nbytes, pos, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

    nbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, nbo)
    glBufferData(GL_ARRAY_BUFFER, nor.nbytes, nor, GL_STATIC_DRAW)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

    tbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, tbo)
    glBufferData(GL_ARRAY_BUFFER, uv.nbytes, uv, GL_STATIC_DRAW)
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)

    glBindVertexArray(0)
    return GLMesh(vao, vbo, nbo, tbo, ebo, index_count=idx.size)

# ----------------------------
# Camera / input
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
        return Matrix44.perspective_projection(self.fovy_deg, aspect, self.near, self.far, dtype=np.float32)

# Globals mirroring your original
camera_dist = 30.0
camera = Camera(
    position=(-20.0, 70.0, 25.0),
    target=(0.0, 10.0, 0.0),
    up=(0.0, 1.0, 0.0),
    fovy_deg=60.0,
    near=0.1, far=1000.0
)

orbit = True
def key_callback(window, key, scancode, action, mods):
    global orbit
    if action == glfw.PRESS:
        if key == glfw.KEY_O:
            orbit = not orbit

def scroll_callback(window, xoff, yoff):
    global camera_dist, camera
    scrollspeed = 3.0
    mw = scrollspeed * yoff
    if mw != 0 and (camera.position.y > scrollspeed + 0.5 or mw > 0.0):
        tar = np.array([camera.target.x, camera.target.y, camera.target.z], dtype=np.float32)
        cam = np.array([camera.position.x, camera.position.y, camera.position.z], dtype=np.float32)
        camera_dist -= mw * 0.3
        cam[1] += (tar[1] - cam[1]) / (abs(cam[1]) + 0.1) * mw
        camera.position = Vector3(cam)

# ----------------------------
# Main
# ----------------------------
def main():
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    # request 4.3 core
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

    window = glfw.create_window(WINDOW_W, WINDOW_H, TITLE, None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create window")

    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glViewport(0, 0, WINDOW_W, WINDOW_H)

    # GL state
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.15, 0.16, 0.19, 1.0)

    # program
    program = build_default_program()
    glUseProgram(program)

    # uniforms
    loc_uModel = glGetUniformLocation(program, "uModel")
    loc_uView  = glGetUniformLocation(program, "uView")
    loc_uProj  = glGetUniformLocation(program, "uProj")
    loc_uLight = glGetUniformLocation(program, "uLightDir")

    scale = 5.0
    model_mat = Matrix44.from_scale([scale, scale, scale], dtype=np.float32)

    # light dir (normalize((30,30,25) - (0,0,-20)))
    l = np.array([30.0, 30.0, 25.0]) - np.array([0.0, 0.0, -20.0])
    l = l / np.linalg.norm(l)
    light_dir = l.astype(np.float32)

    # load mesh
    gl_mesh = load_gltf_first_mesh(MODEL_PATH)

    start_t = time.time()
    frame_start = 0.0
    fps_frames = 0

    while not glfw.window_should_close(window):

        # inputs/events
        glfw.poll_events()

        # per-frame timing
        now = time.time()
        elapsed = now - start_t

        # orbit update
        global orbit, camera_dist, camera
        if orbit:
            cam_ang = elapsed * 0.5
            camera.position = Vector3((
                math.cos(cam_ang) * camera_dist,
                camera.position.y,
                math.sin(cam_ang) * camera_dist
            ))

        # draw
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        aspect = WINDOW_W / WINDOW_H
        view = camera.view()
        proj = camera.proj(aspect)

        glUseProgram(program)
        glUniformMatrix4fv(loc_uModel, 1, GL_FALSE, model_mat)
        glUniformMatrix4fv(loc_uView,  1, GL_FALSE, view)
        glUniformMatrix4fv(loc_uProj,  1, GL_FALSE, proj)
        glUniform3fv(loc_uLight, 1, light_dir)

        glBindVertexArray(gl_mesh.vao)
        glDrawElements(GL_TRIANGLES, gl_mesh.index_count, GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glBindVertexArray(0)

        glfw.swap_buffers(window)

        # simple FPS to console
        fps_frames += 1
        if now - frame_start >= 1.0:
            print(f"fps {fps_frames}")
            frame_start = time.time()
            fps_frames = 0

    # cleanup
    gl_mesh.destroy()
    glDeleteProgram(program)
    glfw.terminate()

main()
