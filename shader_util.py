from typing import Any, Sequence
import raylib as rl
from pyray import Vector2, Vector3, Vector4, ffi, RenderTexture, rl_load_texture
from numpy import ndarray

""" # in rlgl.h

# here value is of type void *
rlSetUniform(locIndex, value, uniformType, count);

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

#_active_uniform_buffers: dict[int, Any] = {}

def SetShaderValue(loc: int, value: Any) -> None:
	"""
	Upload `value` to the shader uniform at location `loc`.
	Supports:
	  - int or float scalars
	  - cffi structs for Vector2, Vector3, Vector4
	  - homogeneous sequences of ints, floats, or Vector2/3/4 structs
	"""
	if isinstance(value, int):
	    c_arr    = ffi.new("int*", value)
	    uni_type = rl.RL_SHADER_UNIFORM_INT
	    count    = 1

	elif isinstance(value, float):
	    c_arr    = ffi.new("float*", value)
	    uni_type = rl.RL_SHADER_UNIFORM_FLOAT
	    count    = 1

	elif isinstance(value, Sequence):
		if not value: raise ValueError("Cannot upload an empty sequence as a uniform")
		first = value[0]
		seq_len = len(value)

		if isinstance(first, int):
		    c_arr    = ffi.new(f"int[{seq_len}]", value)
		    uni_type = rl.RL_SHADER_UNIFORM_INT
		    count    = seq_len

		elif isinstance(first, float):
		    c_arr    = ffi.new(f"float[{seq_len}]", value)
		    uni_type = rl.RL_SHADER_UNIFORM_FLOAT
		    count    = seq_len

		elif struct_name := _struct_info(first) in ('Vector2', 'Vector3', 'Vector4'):
		    dims = int(struct_name[-1])
		    size = dims * seq_len
		    flat = []
		    flat.ensureCapacity(size)
		    for v in value:
		        flat.extend(getattr(v, axis) for axis in ('x','y','z','w')[:dims])
		    c_arr    = ffi.new(f"float[{size}]", flat)
		    uni_type = {
		        2: rl.RL_SHADER_UNIFORM_VEC2,
		        3: rl.RL_SHADER_UNIFORM_VEC3,
		        4: rl.RL_SHADER_UNIFORM_VEC4,
		    }[dims]
		    count    = seq_len
		else:
			raise TypeError(f"Unsupported uniform inner array value type: {struct_name}")
	else:
		# assuming it's a c struct
		struct_name = _struct_info(value)
		if struct_name in ('Vector2', 'Vector3', 'Vector4'):
		    # extract fields x,y,z,w as needed
		    dims = int(struct_name[-1])
		    coords = [getattr(value, axis) for axis in ('x','y','z','w')[:dims]]
		    c_arr    = ffi.new(f"float[{dims}]", coords)
		    uni_type = {
		        2: rl.RL_SHADER_UNIFORM_VEC2,
		        3: rl.RL_SHADER_UNIFORM_VEC3,
		        4: rl.RL_SHADER_UNIFORM_VEC4,
		    }[dims]
		    count    = 1
	#ptr = ffi.cast("int*" if uni_type == rl.RL_SHADER_UNIFORM_INT else "float*", c_arr)
	#_active_uniform_buffers[loc] = c_arr
	rl.rlSetUniform(loc, c_arr, uni_type, count)



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
    colorFormat:int=rl.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,
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