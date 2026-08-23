#include "graphics/shader/shaderVertexBuffers.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using namespace Libs::Graphics;

int g_checks = 0;

void Check(bool value, const char* test, const char* text) {
	g_checks++;
	if (!value) {
		std::fprintf(stderr, "ShaderVertexBuffersTests: %s: failed: %s\n", test, text);
		std::abort();
	}
}

// A vertex buffer sharp, as ShaderApplyAttribSemantics would have copied it out of the guest table.
void AddResource(ShaderVertexInputInfo& info, uint64_t addr, uint16_t stride, uint32_t num_records,
                 uint32_t fetch_index) {
	const int ri = info.resources_num++;

	auto& r     = info.resources[ri];
	r.fields[0] = static_cast<uint32_t>(addr & 0xffffffffu);
	r.fields[1] = static_cast<uint32_t>((addr >> 32u) & 0xffffu) |
	              (static_cast<uint32_t>(stride & 0x3fffu) << 16u);
	r.fields[2] = num_records;
	r.fields[3] = 0;

	auto& rd          = info.resources_dst[ri];
	rd.register_start = ri;
	rd.registers_num  = 4;
	rd.attr_id        = ri;
	rd.fetch_index    = fetch_index;
}

// The attribute ranges must tile [0, resources_num) exactly: no gaps, no overlaps, and every
// attribute in exactly one buffer. Everything downstream indexes the pool through these ranges.
void CheckRangesTile(const ShaderVertexInputInfo& info, const char* test) {
	std::vector<int> seen(static_cast<size_t>(info.resources_num), 0);
	int              total = 0;
	int              next  = 0;
	for (int bi = 0; bi < info.buffers_num; bi++) {
		const auto& b = info.buffers[bi];
		Check(b.attr_first == next, test, "attribute ranges are not packed end to end");
		Check(b.attr_num > 0, test, "a buffer holds no attributes");
		next += b.attr_num;
		total += b.attr_num;
		for (int ai = 0; ai < b.attr_num; ai++) {
			const int ri = info.attr_indices[b.attr_first + ai];
			Check(ri >= 0 && ri < info.resources_num, test, "attribute index out of range");
			Check(seen[static_cast<size_t>(ri)] == 0, test, "attribute claimed by two buffers");
			seen[static_cast<size_t>(ri)] = 1;
		}
	}
	Check(total == info.resources_num, test, "attribute ranges do not cover every attribute");
}

void CheckBuffer(const ShaderVertexInputInfo& info, int bi, uint64_t addr,
                 const std::vector<int>& attrs, const std::vector<uint32_t>& offsets,
                 const char* test) {
	const auto& b = info.buffers[bi];
	Check(b.addr == addr, test, "buffer address");
	Check(b.attr_num == static_cast<int>(attrs.size()), test, "buffer attribute count");
	for (size_t ai = 0; ai < attrs.size(); ai++) {
		const int pos = b.attr_first + static_cast<int>(ai);
		Check(info.attr_indices[pos] == attrs[ai], test, "buffer attribute index");
		Check(info.attr_offsets[pos] == offsets[ai], test, "buffer attribute offset");
	}
}

// Two buffers fed alternately. Each buffer's attribute list must hold only its own attributes, in
// the order they were assigned, and the two lists must not run into each other.
void TestInterleavedBuffers() {
	const char* test = "InterleavedBuffers";

	ShaderVertexInputInfo info {};
	AddResource(info, 0x1000, 32, 100, 0);
	AddResource(info, 0x9000, 32, 200, 0);
	AddResource(info, 0x1004, 32, 100, 0);
	AddResource(info, 0x9008, 32, 200, 0);

	ShaderDetectBuffers(info);

	Check(info.buffers_num == 2, test, "expected two buffers");
	CheckRangesTile(info, test);
	CheckBuffer(info, 0, 0x1000, {0, 2}, {0, 4}, test);
	CheckBuffer(info, 1, 0x9000, {1, 3}, {0, 8}, test);
}

// A later attribute can pull a buffer's base address *down*. Every offset in that buffer, including
// the ones assigned before the merge, has to be relative to the final address. This is why the
// grouping and the offset layout are two separate passes.
void TestMergeLowersBufferAddress() {
	const char* test = "MergeLowersBufferAddress";

	ShaderVertexInputInfo info {};
	AddResource(info, 0x1010, 32, 100, 0);
	AddResource(info, 0x1000, 32, 100, 0);
	AddResource(info, 0x1008, 32, 100, 0);

	ShaderDetectBuffers(info);

	Check(info.buffers_num == 1, test, "expected one buffer");
	CheckRangesTile(info, test);
	CheckBuffer(info, 0, 0x1000, {0, 1, 2}, {0x10, 0, 8}, test);
}

// The three conditions that keep two attributes apart.
void TestNoMergeConditions() {
	{
		const char*           test = "NoMerge/FetchIndex";
		ShaderVertexInputInfo info {};
		AddResource(info, 0x1000, 32, 100, 0);
		AddResource(info, 0x1004, 32, 100, 1);
		ShaderDetectBuffers(info);
		Check(info.buffers_num == 2, test, "a per-instance attribute merged into a per-vertex one");
		CheckRangesTile(info, test);
		CheckBuffer(info, 0, 0x1000, {0}, {0}, test);
		CheckBuffer(info, 1, 0x1004, {1}, {0}, test);
	}
	{
		const char*           test = "NoMerge/Stride";
		ShaderVertexInputInfo info {};
		AddResource(info, 0x1000, 32, 100, 0);
		AddResource(info, 0x1004, 16, 100, 0);
		ShaderDetectBuffers(info);
		Check(info.buffers_num == 2, test, "attributes with different strides merged");
		CheckRangesTile(info, test);
	}
	{
		const char*           test = "NoMerge/Distance";
		ShaderVertexInputInfo info {};
		AddResource(info, 0x1000, 16, 100, 0);
		AddResource(info, 0x1040, 16, 100, 0);
		ShaderDetectBuffers(info);
		Check(info.buffers_num == 2, test, "attributes a full stride apart merged");
		CheckRangesTile(info, test);
		CheckBuffer(info, 0, 0x1000, {0}, {0}, test);
		CheckBuffer(info, 1, 0x1040, {1}, {0}, test);
	}
}

// The pool holds exactly RES_MAX entries, so a single buffer owning every attribute fills it to the
// brim and must not run past the end.
void TestAllAttributesInOneBuffer() {
	const char* test = "AllAttributesInOneBuffer";

	ShaderVertexInputInfo info {};
	for (int i = 0; i < ShaderVertexInputInfo::RES_MAX; i++) {
		AddResource(info, 0x2000 + static_cast<uint64_t>(i), 64, 100, 0);
	}

	ShaderDetectBuffers(info);

	Check(info.buffers_num == 1, test, "expected one buffer");
	CheckRangesTile(info, test);
	Check(info.buffers[0].attr_first == 0, test, "the only buffer must start the pool");
	Check(info.buffers[0].attr_num == ShaderVertexInputInfo::RES_MAX, test,
	      "the only buffer must own every attribute");
	for (int i = 0; i < ShaderVertexInputInfo::RES_MAX; i++) {
		Check(info.attr_indices[i] == i, test, "attribute order");
		Check(info.attr_offsets[i] == static_cast<uint32_t>(i), test, "attribute offset");
	}
}

void TestEveryAttributeItsOwnBuffer() {
	const char* test = "EveryAttributeItsOwnBuffer";

	ShaderVertexInputInfo info {};
	for (int i = 0; i < ShaderVertexInputInfo::RES_MAX; i++) {
		AddResource(info, 0x10000 + (static_cast<uint64_t>(i) << 12u), 16, 100, 0);
	}

	ShaderDetectBuffers(info);

	Check(info.buffers_num == ShaderVertexInputInfo::RES_MAX, test, "expected one buffer each");
	CheckRangesTile(info, test);
	for (int i = 0; i < ShaderVertexInputInfo::RES_MAX; i++) {
		CheckBuffer(info, i, 0x10000 + (static_cast<uint64_t>(i) << 12u), {i}, {0}, test);
	}
}

void TestNoAttributes() {
	const char*           test = "NoAttributes";
	ShaderVertexInputInfo info {};
	ShaderDetectBuffers(info);
	Check(info.buffers_num == 0, test, "an empty input produced buffers");
}

} // namespace

int main() {
	TestInterleavedBuffers();
	TestMergeLowersBufferAddress();
	TestNoMergeConditions();
	TestAllAttributesInOneBuffer();
	TestEveryAttributeItsOwnBuffer();
	TestNoAttributes();

	std::printf("ShaderVertexBuffersTests: all %d checks passed\n", g_checks);
	return 0;
}
