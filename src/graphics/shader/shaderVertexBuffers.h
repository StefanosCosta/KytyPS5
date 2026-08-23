#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADERVERTEXBUFFERS_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADERVERTEXBUFFERS_H_

#include "graphics/shader/shader.h"

namespace Libs::Graphics {

// Groups the attributes in [0, info.resources_num) into vertex buffer bindings, then fills
// info.buffers_num, info.buffers and the packed info.attr_indices / info.attr_offsets lists.
// Reads info.resources and info.resources_dst; writes nothing else.
void ShaderDetectBuffers(ShaderVertexInputInfo& info);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADERVERTEXBUFFERS_H_ */
