#include "graphics/shader/shaderVertexBuffers.h"

#include "common/assert.h"

#include <algorithm>
#include <cstdint>

namespace Libs::Graphics {

void ShaderDetectBuffers(ShaderVertexInputInfo& info) {
	info.buffers_num = 0;

	// Which buffer each attribute was assigned to. The assignment and the layout are separate
	// passes because a later merge can still lower a buffer's addr, and every attribute offset is
	// relative to that buffer's final addr.
	int res_buffer[ShaderVertexInputInfo::RES_MAX] = {};

	for (int ri = 0; ri < info.resources_num; ri++) {
		const auto& r = info.resources[ri];

		bool merged = false;
		for (int bi = 0; bi < info.buffers_num; bi++) {
			auto& b = info.buffers[bi];

			uint64_t stride = b.stride;

			if (stride == r.Stride() &&
			    b.fetch_index == static_cast<uint32_t>(info.resources_dst[ri].fetch_index)) {
				uint64_t rbase   = r.Base48();
				uint64_t base    = std::min(rbase, b.addr);
				uint64_t offset1 = rbase - base;
				uint64_t offset2 = b.addr - base;

				if (offset1 < stride && offset2 < stride) {
					EXIT_NOT_IMPLEMENTED(b.num_records != r.NumRecords());
					b.addr = base;
					b.attr_num++;
					res_buffer[ri] = bi;
					merged         = true;
					break;
				}
			}
		}

		if (!merged) {
			EXIT_NOT_IMPLEMENTED(info.buffers_num >= ShaderVertexInputInfo::RES_MAX);
			int bi                       = info.buffers_num++;
			info.buffers[bi].addr        = r.Base48();
			info.buffers[bi].stride      = r.Stride();
			info.buffers[bi].num_records = r.NumRecords();
			info.buffers[bi].fetch_index = info.resources_dst[ri].fetch_index;
			info.buffers[bi].attr_num    = 1;
			res_buffer[ri]               = bi;
		}
	}

	// Hand out the attribute ranges. They tile [0, resources_num), so the pool never overflows.
	int next = 0;
	for (int bi = 0; bi < info.buffers_num; bi++) {
		auto& b      = info.buffers[bi];
		b.attr_first = next;
		next += b.attr_num;
		// Reused below as the write cursor, and restored to the same value by the fill loop.
		b.attr_num = 0;
	}
	EXIT_IF(next != info.resources_num);

	// Ascending in ri, so each buffer's attribute order matches the order they were assigned in.
	// ShaderGetIdVS hashes that order, so keeping it is what makes the shader ids unchanged.
	for (int ri = 0; ri < info.resources_num; ri++) {
		auto&      b   = info.buffers[res_buffer[ri]];
		const int  pos = b.attr_first + b.attr_num++;
		info.attr_indices[pos] = ri;
		info.attr_offsets[pos] = static_cast<uint32_t>(info.resources[ri].Base48() - b.addr);
	}
}

} // namespace Libs::Graphics
