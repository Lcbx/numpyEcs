import numpy as np
import shader_util as su
from shader_util import Vector2, Vector3, Color, Camera3D, RenderTexture


WINDOW_w, WINDOW_h = 1800, 900

import glfw
from OpenGL.GL import *

if not glfw.init():
    raise RuntimeError("Failed to initialize GLFW")

## Request OpenGL 4.3 Core Profile
#glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
#glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
#glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
#glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

## Create window and context
#window = glfw.create_window(WINDOW_w, WINDOW_h, "OpenGL 4.3 Window", None, None)
#if not window:
#    glfw.terminate()
#    raise RuntimeError("Failed to create window")

#glfw.make_context_current(window)


su.InitWindow(WINDOW_w, WINDOW_h, "Hello")
#su.rl.CloseWindow()


model_root = b'scenes/resources/'
model = su.rl.LoadModel(model_root + b'heightmap_mesh.glb')
#mesh = model.meshes[0]
#print('vertexCount', mesh.vertexCount)
#print('triangleCount', mesh.triangleCount)

#print('vertices', mesh.vertices)
#print('texcoords', mesh.texcoords)
#print('texcoords2', mesh.texcoords2)
#print('normals', mesh.normals)
#print('tangents', mesh.tangents)
#print('colors', mesh.colors)
#print('indices', mesh.indices)

#print('animVertices', mesh.animVertices)
#print('animNormals', mesh.animNormals)
#print('boneIds', mesh.boneIds)
#print('boneWeights', mesh.boneWeights)
#print('boneMatrices', mesh.boneMatrices)
#print('boneCount', mesh.boneCount)

#print('vaoId', mesh.vaoId)
#print('vboId', mesh.vboId)

camera_dist = 30
camera_nearFar = (0.1, 1000.0)
camera = Camera3D(
	Vector3(-20, 70, 25),
	Vector3(0,10,0),
	Vector3(0,1,0),
	60.0,
	su.rl.CAMERA_PERSPECTIVE
)


lightDir = su.rl.Vector3Normalize(su.rl.Vector3Subtract(Vector3(30, 30, 25), Vector3(0,0,-20)))

def load_shaders():
	global sceneShader

	# Bettersahder is a utility class built on top of raylib shader compilation
	newShader = su.BetterShader('scenes/minimalShader.shader')
	if newShader.valid(): sceneShader = newShader
	else: raise Exception('minimalShader.shader')


sceneShader : su.BetterShader = None
load_shaders()

def run():
	while not su.rl.WindowShouldClose():

		frameTime = su.rl.GetFrameTime()
		
		with su.WatchTimer('update'):
			inputs()
			update(frameTime)
		
		with su.WatchTimer('total draw'):
			

			su.rl.BeginDrawing()
			
			with su.RenderContext(shader=sceneShader, clipPlanes=camera_nearFar, camera=camera) as render:
			
				su.ClearBuffers()
				
				sceneShader.lightDir = lightDir
				draw_scene(render)
		
			su.rl.DrawText(f"fps {su.rl.GetFPS()}".encode(), 10, 10, 20, su.rl.LIGHTGRAY)
			su.WatchTimer.display(10, 40, 20, su.rl.LIGHTGRAY)

			su.rl.EndDrawing()
			su.WatchTimer.capture()


orbit = True
def inputs():
	global camera
	global camera_dist
	global orbit

	if su.rl.IsKeyPressed(su.rl.KEY_R): load_shaders()
	if su.rl.IsKeyPressed(su.rl.KEY_O): orbit = not orbit

	if su.rl.IsKeyPressed(su.rl.KEY_P):
		light_camera.projection = (su.rl.CAMERA_ORTHOGRAPHIC
			if light_camera.projection == su.rl.CAMERA_PERSPECTIVE
			else su.rl.CAMERA_PERSPECTIVE
		)

	scrollspeed = 3.0
	mw = scrollspeed * su.rl.GetMouseWheelMove()
	if mw != 0 and ( camera.position.y > scrollspeed + 0.5 or mw > 0.0):
		cam_pos = np.array([camera.position.x, camera.position.y, camera.position.z])
		tar = np.array([camera.target.x, camera.target.y, camera.target.z])
		camera_dist -= mw * 0.3;
		cam_pos[1] += (tar[1] - cam_pos[1]) / np.linalg.norm(cam_pos[1] + 0.1) * mw
		camera.position = Vector3(cam_pos[0], cam_pos[1], cam_pos[2])


def update(frameTime):
	global camera
	
	time = su.rl.GetTime()
	
	if orbit:
		cam_ang = time * 0.5
		camera.position = Vector3(
			np.cos(cam_ang) * camera_dist,
			camera.position.y,
			np.sin(cam_ang) * camera_dist)


#ver = glGetString(GL_VERSION).decode()
#renderer = glGetString(GL_RENDERER).decode()
#print("GL_VERSION:", ver, "| GL_RENDERER:", renderer)

## Tessellation requires >= 4.0
#maj = int(ver.split('.')[0])
#if maj < 4:
#    raise RuntimeError("Tessellation requires OpenGL 4.0+. Current: " + ver)

#max_patch = glGetInteger(GL_MAX_PATCH_VERTICES)
#print("GL_MAX_PATCH_VERTICES:", max_patch)

#TESSELLATION_CONTROL_SHADER = """
##version 430
#layout(vertices = 3) out;

#void main() {
#    // pass through vertex positions
#    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
#    // set tessellation levels
#    gl_TessLevelInner[0] = 4.0;
#    gl_TessLevelOuter[0] = 4.0;
#    gl_TessLevelOuter[1] = 4.0;
#    gl_TessLevelOuter[2] = 4.0;
#}
#"""

#TESSELLATION_EVALUATION_SHADER = """
##version 430
#layout(triangles, equal_spacing, ccw) in;

#uniform sampler2D heightmap;
#out vec2 texCoord;

#void main() {
#    // bilinear interpolation of patch coords
#    vec2 uv = (gl_TessCoord.x * gl_in[0].gl_Position.xy +
#               gl_TessCoord.y * gl_in[1].gl_Position.xy +
#               gl_TessCoord.z * gl_in[2].gl_Position.xy);// +
#               //gl_TessCoord.w * gl_in[3].gl_Position.xy);

#    texCoord = uv * 0.5 + 0.5;

#    // sample height from texture
#    float h = texture(heightmap, texCoord).r;

#    // place patch according to height
#    gl_Position = vec4(uv.x, h, uv.y, 1.0);
#}
#"""


#program = glCreateProgram()
#for src, stype in [
#    (sceneShader.vertex_glsl, GL_VERTEX_SHADER),
#    (TESSELLATION_CONTROL_SHADER, GL_TESS_CONTROL_SHADER),
#    (TESSELLATION_EVALUATION_SHADER, GL_TESS_EVALUATION_SHADER),
#    (sceneShader.fragment_glsl, GL_FRAGMENT_SHADER)
#]:
#	shaderId = glCreateShader(stype)
#	glShaderSource(shaderId, src)
#	glCompileShader(shaderId)
#	if not glGetShaderiv(shaderId, GL_COMPILE_STATUS):
#		raise RuntimeError(glGetShaderInfoLog(shaderId).decode())
#	glAttachShader(program, shaderId)
#sceneShader.shaderStruct.id = program

#glLinkProgram(program)
#if not glGetProgramiv(program, GL_LINK_STATUS):
#    raise RuntimeError(glGetProgramInfoLog(program).decode())

def draw_scene(render:su.RenderContext, randomize_color=False):

	for i in range(model.materialCount):
		model.materials[i].shader = render.shader.shaderStruct

	scale = 0.01
	su.rl.DrawModelEx(model, Vector3(0,0,0), Vector3(1,0,0), 0.0, Vector3(scale,scale,scale), su.rl.BEIGE)
	
	

	#glUseProgram(sceneShader.shaderStruct.id)
	#vao_id = mesh.vaoId
	#glPatchParameteri(GL_PATCH_VERTICES, 3)
	#glBindVertexArray(vao_id)
	#glDrawElements(GL_PATCHES, mesh.triangleCount * 3, GL_UNSIGNED_SHORT, None)


run()