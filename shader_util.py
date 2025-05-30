from typing import Any, Sequence
import raylib as rl
from pyray import Shader, Vector2, Vector3, Vector4, ffi, RenderTexture, rl_load_texture

""" # in rlgl.h

void SetShaderValue(Shader shader, int locIndex, const void *value, int uniformType);               // Set shader uniform value
void SetShaderValueV(Shader shader, int locIndex, const void *value, int uniformType, int count);   // Set shader uniform value vector

# uniform types :
	case RL_SHADER_UNIFORM_FLOAT: glUniform1fv(locIndex, count, (float *)value); break;
    case RL_SHADER_UNIFORM_VEC2: glUniform2fv(locIndex, count, (float *)value); break;
    case RL_SHADER_UNIFORM_VEC3: glUniform3fv(locIndex, count, (float *)value); break;
    case RL_SHADER_UNIFORM_VEC4: glUniform4fv(locIndex, count, (float *)value); break;
    case RL_SHADER_UNIFORM_INT:  glUniform1iv(locIndex, count, (int *)value); break;
    case RL_SHADER_UNIFORM_IVEC2: glUniform2iv(locIndex, count, (int *)value); break;
    case RL_SHADER_UNIFORM_IVEC3: glUniform3iv(locIndex, count, (int *)value); break;
    case RL_SHADER_UNIFORM_IVEC4: glUniform4iv(locIndex, count, (int *)value); break;
#if !defined(GRAPHICS_API_OPENGL_ES2)
    case RL_SHADER_UNIFORM_UINT: glUniform1uiv(locIndex, count, (unsigned int *)value); break;
    case RL_SHADER_UNIFORM_UIVEC2: glUniform2uiv(locIndex, count, (unsigned int *)value); break;
    case RL_SHADER_UNIFORM_UIVEC3: glUniform3uiv(locIndex, count, (unsigned int *)value); break;
    case RL_SHADER_UNIFORM_UIVEC4: glUniform4uiv(locIndex, count, (unsigned int *)value); break;
#endif
    case RL_SHADER_UNIFORM_SAMPLER2D: glUniform1iv(locIndex, count, (int *)value); break;

"""

# helper to detect cdata struct type
def _struct_info(v):
    ty = ffi.typeof(v)
    if ty.kind == 'struct':
        name = ty.cname  # ex: 'struct Vector2'
        return name.replace('struct ', '')
    return None

shader_enum = {
	'int':rl.RL_SHADER_UNIFORM_INT,
	'float':rl.RL_SHADER_UNIFORM_FLOAT,
	'Vector2':rl.RL_SHADER_UNIFORM_VEC2,
	'Vector3':rl.RL_SHADER_UNIFORM_VEC3,
	'Vector4':rl.RL_SHADER_UNIFORM_VEC4,
	'Color':rl.RL_SHADER_UNIFORM_VEC4,
}

def SetShaderValue(shader : Shader, loc: int, value: Any) -> None:
	"""
	Upload `value` to the shader uniform at location `loc`.
	Supports:
	  - int or float scalars
	  - cffi structs for Vector2, Vector3, Vector4
	  - homogeneous sequences of ints, floats, or Vector2/3/4 structs
	"""
	tested = value
	size = 1

	if isinstance(value, Sequence):
		if not value: raise ValueError("Cannot upload an empty sequence as a uniform")
		tested = value[0]
		size = len(value)

	str_type = (
		 'int'   if isinstance(tested, int)
	else 'float' if isinstance(tested, float)
	else _struct_info(tested) )

	uni_type = shader_enum[str_type]

	if size == 1:
		c_arr = ffi.new(f'{str_type}*', value)
		rl.SetShaderValue(shader, loc, c_arr, uni_type)
	else:
		c_arr = ffi.new(f'{str_type}[{size}]', value)
		rl.SetShaderValueV(shader, loc, c_arr, uni_type, size)


""" # in rlgl.h
# for color :
PIXELFORMAT_UNCOMPRESSED_GRAYSCALE = 1,     // 8 bit per pixel (no alpha)
PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA,        // 8*2 bpp (2 channels)
PIXELFORMAT_UNCOMPRESSED_R5G6B5,            // 16 bpp
PIXELFORMAT_UNCOMPRESSED_R8G8B8,            // 24 bpp
PIXELFORMAT_UNCOMPRESSED_R5G5B5A1,          // 16 bpp (1 bit alpha)
PIXELFORMAT_UNCOMPRESSED_R4G4B4A4,          // 16 bpp (4 bit alpha)
PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,          // 32 bpp
PIXELFORMAT_UNCOMPRESSED_R32,               // 32 bpp (1 channel - float)
PIXELFORMAT_UNCOMPRESSED_R32G32B32,         // 32*3 bpp (3 channels - float)
PIXELFORMAT_UNCOMPRESSED_R32G32B32A32,      // 32*4 bpp (4 channels - float)
PIXELFORMAT_UNCOMPRESSED_R16,               // 16 bpp (1 channel - half float)
PIXELFORMAT_UNCOMPRESSED_R16G16B16,         // 16*3 bpp (3 channels - half float)
PIXELFORMAT_UNCOMPRESSED_R16G16B16A16,      // 16*4 bpp (4 channels - half float)
PIXELFORMAT_COMPRESSED_DXT1_RGB,            // 4 bpp (no alpha)
PIXELFORMAT_COMPRESSED_DXT1_RGBA,           // 4 bpp (1 bit alpha)
PIXELFORMAT_COMPRESSED_DXT3_RGBA,           // 8 bpp
PIXELFORMAT_COMPRESSED_DXT5_RGBA,           // 8 bpp
PIXELFORMAT_COMPRESSED_ETC1_RGB,            // 4 bpp
PIXELFORMAT_COMPRESSED_ETC2_RGB,            // 4 bpp
PIXELFORMAT_COMPRESSED_ETC2_EAC_RGBA,       // 8 bpp
PIXELFORMAT_COMPRESSED_PVRT_RGB,            // 4 bpp
PIXELFORMAT_COMPRESSED_PVRT_RGBA,           // 4 bpp
PIXELFORMAT_COMPRESSED_ASTC_4x4_RGBA,       // 8 bpp
PIXELFORMAT_COMPRESSED_ASTC_8x8_RGBA        // 2 bpp

# for depth raylib lets opengl choose (not easy to set)
"""
def create_render_buffer(width : int, height:int,
    colorFormat:int=rl.PIXELFORMAT_UNCOMPRESSED_R8G8B8,
    depth_map:bool=False
    ) -> RenderTexture :
    # has a color buffer by default
    #target = rl.LoadRenderTexture(width, height)

    target = RenderTexture()
    target.id = rl.rlLoadFramebuffer()
    
    if target.id > 0:
        rl.rlEnableFramebuffer(target.id)

        target.texture.width = width
        target.texture.height = height
        
        if colorFormat:
            target.texture.id = rl_load_texture(None, width, height, colorFormat, 1)
            target.texture.format = colorFormat
            target.texture.mipmaps = 1
            rl.rlFramebufferAttach(target.id, target.texture.id, rl.RL_ATTACHMENT_COLOR_CHANNEL0, rl.RL_ATTACHMENT_TEXTURE2D, 0)

        if depth_map:
	        target.depth.id = rl.rlLoadTextureDepth(width, height, False)
	        target.depth.width = width
	        target.depth.height = height
	        target.depth.format = 19
	        target.depth.mipmaps = 1
	        rl.rlFramebufferAttach(target.id, target.depth.id, rl.RL_ATTACHMENT_DEPTH, rl.RL_ATTACHMENT_TEXTURE2D, 0)
        
        rl.rlDisableFramebuffer()

    return target