#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GUEST_GPU_GPU_FORMAT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GUEST_GPU_GPU_FORMAT_H_

#include "graphics/guest_gpu/gpu_defs.h"

namespace Libs::Graphics::Prospero {

enum class ChannelOrderSupport : uint8_t {
	kNone,
	kStandardOnly,
	kAll,
};

struct RenderTargetFormatEncoding {
	BufferFormat        buffer_format = BufferFormat::kInvalid;
	uint8_t             components    = 0;
	ChannelOrderSupport order_support = ChannelOrderSupport::kNone;

	[[nodiscard]] constexpr bool IsValid() const {
		return buffer_format != BufferFormat::kInvalid && components >= 1u && components <= 4u &&
		       order_support != ChannelOrderSupport::kNone;
	}

	[[nodiscard]] constexpr bool SupportsOrder(ChannelOrder order) const {
		switch (order_support) {
			case ChannelOrderSupport::kStandardOnly: return order == ChannelOrder::kStandard;
			case ChannelOrderSupport::kAll:
				switch (order) {
					case ChannelOrder::kStandard:
					case ChannelOrder::kAlt:
					case ChannelOrder::kReversed:
					case ChannelOrder::kAltReversed: return true;
				}
				return false;
			default: return false;
		}
	}
};

RenderTargetFormatEncoding ResolveRenderTargetFormat(ChannelLayout layout, ChannelType type);
uint32_t                   NumBytesPerElement(BufferFormat format);
uint32_t                   BlockCompressedBytesPerBlock(BufferFormat format);
uint32_t                   RenderTargetBytesPerElement(BufferFormat format);
bool                       IsFmaskTextureFormat(BufferFormat format);
bool                       IsSampledTextureFormat(BufferFormat format);
bool                       IsUintTextureFormat(BufferFormat format);
BufferFormat               RemapTextureFormat(BufferFormat format);

// True when a texture descriptor with this (format, tile) can resolve to a host image created with
// a depth format, i.e. one that advertises VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT.
//
// RDNA2 lets IMAGE_SAMPLE_C_* compare against any surface (ISA doc 70648 p. 72: the _C suffix is
// defined purely as PCF), but Vulkan's Dref sampling is legal only on depth-capable formats, and
// violating that hangs the GPU. This predicate decides which of the two lowerings a compare-sample
// gets, and it must be evaluated at shader-compile time -- before the texture cache has resolved
// anything -- so it may look only at the guest descriptor.
//
// These are necessary conditions, mirroring DEPTH_FORMAT_POLICIES (host_gpu image info) and the
// kDepth tile mode that IsSupportedDepthTargetDescriptor enforces at bind time. Keep the two in
// step; a drift guard in the shader recompiler tests asserts it.
bool SupportsDepthCompareSampling(BufferFormat format, TileMode tile);

} // namespace Libs::Graphics::Prospero

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GUEST_GPU_GPU_FORMAT_H_ */
