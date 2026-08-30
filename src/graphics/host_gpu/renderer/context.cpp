#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vma.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstring>
#include <vector>
namespace Libs::Graphics {

namespace {

// GPU breadcrumbs. VK_ERROR_DEVICE_LOST is sticky, so the call that reports it is almost never the
// one that killed the device; a checkpoint marker records what the GPU itself last reached.
struct GpuCheckpoint {
	uint64_t magic         = 0;   // identifies a marker that is really ours
	uint32_t op            = 0;
	uint64_t submit_id     = 0;
	uint64_t submit_seq    = 0;
	uint32_t slot          = 0;
	uint32_t args[4]       = {};
	uint64_t arg4          = 0;
};

// Stable storage: vkCmdSetCheckpointNV keeps only the pointer, so the slot must outlive the
// submission. A ring is enough -- only the newest entries matter after a hang.
constexpr size_t              CheckpointRingSize = 4096;
GpuCheckpoint                 g_checkpoints[CheckpointRingSize];
std::atomic<size_t>           g_checkpoint_next {0};

const char* DebugOpName(uint32_t op) {
	switch (static_cast<CommandBufferDebugOp>(op)) {
		case CommandBufferDebugOp::DispatchDirect: return "DispatchDirect";
		case CommandBufferDebugOp::DrawIndex: return "DrawIndex";
		case CommandBufferDebugOp::DrawIndexAuto: return "DrawIndexAuto";
		case CommandBufferDebugOp::EopWrite: return "EopWrite";
		case CommandBufferDebugOp::EopInterrupt: return "EopInterrupt";
		case CommandBufferDebugOp::EopWriteBack: return "EopWriteBack";
		case CommandBufferDebugOp::EopFlip: return "EopFlip";
		case CommandBufferDebugOp::EopWriteBackFlip: return "EopWriteBackFlip";
		case CommandBufferDebugOp::EopOnlyFlip: return "EopOnlyFlip";
		default: return "Unknown";
	}
}

// Ask the driver which checkpoints the GPU actually passed. Anything reported at a "top of pipe"
// stage was reached but not finished -- that is the command that hung.
void DumpGpuCheckpoints(GraphicContext& graphics) {
	if (!graphics.diagnostic_checkpoints_enabled ||
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV == nullptr) {
		return;
	}
	uint32_t count = 0;
	graphics.queue.getCheckpointDataNV(&count, nullptr);
	if (count == 0) {
		std::printf("gpu checkpoints: none reported\n");
		std::fflush(stdout);
		return;
	}
	std::vector<vk::CheckpointDataNV> data(count);
	graphics.queue.getCheckpointDataNV(&count, data.data());
	std::printf("gpu checkpoints: %u reported (the GPU's last known position)\n", count);
	for (const auto& entry: data) {
		const auto* cp = static_cast<const GpuCheckpoint*>(entry.pCheckpointMarker);
		const bool  ours =
		    cp != nullptr && cp >= std::begin(g_checkpoints) && cp < std::end(g_checkpoints) &&
		    cp->magic == 0x4b595459434b5054ull;
		if (!ours) {
			std::printf("  stage=0x%08x  <marker %p is not ours>\n",
			            static_cast<uint32_t>(entry.stage), entry.pCheckpointMarker);
			continue;
		}
		std::printf("  stage=0x%08x  %s slot=%u submit_seq=%" PRIu64 " submit_id=%" PRIu64
		            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
		            static_cast<uint32_t>(entry.stage), DebugOpName(cp->op), cp->slot,
		            cp->submit_seq, cp->submit_id, cp->args[0], cp->args[1], cp->args[2],
		            cp->args[3], cp->arg4);
	}
	std::fflush(stdout);
}

// The faulting address and fault type, when the driver can supply them.
void DumpDeviceFault(GraphicContext& graphics) {
	if (!graphics.device_fault_enabled ||
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT == nullptr) {
		return;
	}
	vk::DeviceFaultCountsEXT counts {};
	if (graphics.device.getFaultInfoEXT(&counts, nullptr) != vk::Result::eSuccess) {
		return;
	}
	std::vector<vk::DeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<vk::DeviceFaultVendorInfoEXT>  vendor(counts.vendorInfoCount);
	vk::DeviceFaultInfoEXT                     info {};
	info.pAddressInfos = addresses.data();
	info.pVendorInfos  = vendor.data();
	if (graphics.device.getFaultInfoEXT(&counts, &info) != vk::Result::eSuccess) {
		return;
	}
	std::printf("device fault: %s\n", info.description.data());
	for (uint32_t i = 0; i < counts.addressInfoCount; i++) {
		std::printf("  addr=0x%016" PRIx64 " precision=0x%" PRIx64 " type=%u\n",
		            static_cast<uint64_t>(addresses[i].reportedAddress),
		            static_cast<uint64_t>(addresses[i].addressPrecision),
		            static_cast<uint32_t>(addresses[i].addressType));
	}
	std::fflush(stdout);
}

void ReportVulkanFatal(const char* what, vk::Result result, uint32_t slot, uint64_t submit_seq,
                       uint32_t debug_op, uint64_t debug_submit, uint32_t arg0, uint32_t arg1,
                       uint32_t arg2, uint32_t arg3, uint64_t arg4,
                       GraphicContext* graphics = nullptr) {
	LOGF("%s failed: %s (%d), slot=%u submit_seq=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     what, VulkanToString(result).c_str(), static_cast<int>(result), slot, submit_seq, debug_op,
	     debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::printf("%s failed: %s (%d), slot=%u submit_seq=%" PRIu64
	            " debug_op=%u debug_submit=%" PRIu64 " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	            what, VulkanToString(result).c_str(), static_cast<int>(result), slot, submit_seq,
	            debug_op, debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::fflush(stdout);
	if (result == vk::Result::eErrorDeviceLost && graphics != nullptr) {
		// The reporting call is rarely the culprit: ask the GPU where it actually stopped.
		DumpGpuCheckpoints(*graphics);
		DumpDeviceFault(*graphics);
	}
}

} // namespace

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_scheduler(scheduler), m_graphics(scheduler.Graphics()),
      m_slot(scheduler.AllocateCommandBuffer()) {}

CommandBuffer::~CommandBuffer() {
	Release();
}

bool CommandBuffer::IsInvalid() const {
	return m_slot == nullptr;
}

vk::CommandBuffer CommandBuffer::Handle() const {
	EXIT_IF(IsInvalid());

	const auto handle = m_slot->buffer;
	EXIT_IF(handle == nullptr);
	return handle;
}

void CommandBuffer::Release() {
	EXIT_IF(IsInvalid());

	Common::LockGuard lock(*m_slot->pool_mutex);

	WaitForFence();

	m_slot->busy = false;
	m_slot->Reset();
	m_slot = nullptr;

	EXIT_NOT_IMPLEMENTED(!IsInvalid());
}

void CommandBuffer::Begin() const {
	EXIT_IF(m_rendering);
	auto buffer = Handle();

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.sType            = vk::StructureType::eCommandBufferBeginInfo;
	begin_info.pNext            = nullptr;
	begin_info.flags            = {};
	begin_info.pInheritanceInfo = nullptr;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::End() const {
	EndRendering();
	auto buffer = Handle();

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0, uint32_t arg1,
                                 uint32_t arg2, uint32_t arg3, uint64_t arg4) {
	m_debug_op        = op;
	m_debug_submit_id = submit_id;
	m_debug_arg0      = arg0;
	m_debug_arg1      = arg1;
	m_debug_arg2      = arg2;
	m_debug_arg3      = arg3;
	m_debug_arg4      = arg4;

	if (m_graphics.diagnostic_checkpoints_enabled &&
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdSetCheckpointNV != nullptr && !IsInvalid()) {
		auto& slot = g_checkpoints[g_checkpoint_next.fetch_add(1, std::memory_order_relaxed) %
		                           CheckpointRingSize];
		slot.magic      = 0x4b595459434b5054ull; // "KYTYCKPT"
		slot.op         = op;
		slot.submit_id  = submit_id;
		slot.submit_seq = m_submit_seq;
		slot.slot       = m_slot->id;
		slot.args[0]    = arg0;
		slot.args[1]    = arg1;
		slot.args[2]    = arg2;
		slot.args[3]    = arg3;
		slot.arg4       = arg4;
		Handle().setCheckpointNV(&slot);
	}
}

void CommandBuffer::Execute(const SubmitInfo& submit) {
	EXIT_IF(IsInvalid());
	EXIT_IF(m_execute);
	EXIT_IF(submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	        submit.num_signal_semaphores > SubmitInfo::MaxSemaphores);

	auto buffer = Handle();
	auto fence  = m_slot->fence;

	vk::TimelineSemaphoreSubmitInfo timeline_info {};
	timeline_info.sType                     = vk::StructureType::eTimelineSemaphoreSubmitInfo;
	timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
	timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
	timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
	timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

	vk::SubmitInfo submit_info {};
	submit_info.sType                = vk::StructureType::eSubmitInfo;
	submit_info.pNext                = &timeline_info;
	submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
	submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
	submit_info.pWaitDstStageMask    = submit.wait_stages.data();
	submit_info.commandBufferCount   = 1;
	submit_info.pCommandBuffers      = &buffer;
	submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
	submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

	auto& graphics = m_graphics;
	EXIT_IF(graphics.queue == nullptr);

	auto result = graphics.device.resetFences(1, &fence);
	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkResetFences (before submit)", result, m_slot->id, m_submit_seq,
		                  m_debug_op, m_debug_submit_id, m_debug_arg0, m_debug_arg1, m_debug_arg2,
		                  m_debug_arg3, m_debug_arg4, &m_graphics);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	if (Config::GraphicsDebugDumpEnabled()) {
		LOGF("vkQueueSubmit begin: slot=%u waits=%u signals=%u debug_op=%u debug_submit=%" PRIu64
		     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
		     m_slot->id, submit.num_wait_semaphores, submit.num_signal_semaphores, m_debug_op,
		     m_debug_submit_id, m_debug_arg0, m_debug_arg1, m_debug_arg2, m_debug_arg3,
		     m_debug_arg4);
	}

	{
		Common::LockGuard lock(graphics.queue_mutex);
		m_submit_seq = m_scheduler.NextSubmitSequence();
		result       = graphics.queue.submit(1, &submit_info, fence);
	}

	m_execute      = true;
	m_fence_waited = false;

	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkQueueSubmit", result, m_slot->id, m_submit_seq, m_debug_op,
		                  m_debug_submit_id, m_debug_arg0, m_debug_arg1, m_debug_arg2, m_debug_arg3,
		                  m_debug_arg4, &m_graphics);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::WaitForFence() {
	FinalizeFence(false);
}

void CommandBuffer::WaitForFenceOnly() {
	EXIT_IF(IsInvalid());
	if (!m_execute || m_fence_waited) {
		return;
	}
	auto device = m_graphics.device;
	auto result = device.waitForFences(1, &m_slot->fence, VK_TRUE, UINT64_MAX);
	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkWaitForFences", result, m_slot->id, m_submit_seq, m_debug_op,
		                  m_debug_submit_id, m_debug_arg0, m_debug_arg1, m_debug_arg2, m_debug_arg3,
		                  m_debug_arg4, &m_graphics);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	m_fence_waited = true;
}

bool CommandBuffer::IsSubmissionOutstanding() const {
	if (IsInvalid() || !m_execute || m_fence_waited) {
		return false;
	}
	return m_graphics.device.getFenceStatus(m_slot->fence) == vk::Result::eNotReady;
}

void CommandBuffer::WaitForFenceAndReset() {
	FinalizeFence(true);
}

void CommandBuffer::FinalizeFence(bool reset_recording) {
	const bool was_executed = m_execute;
	WaitForFenceOnly();
	if (was_executed) {
		m_execute      = false;
		m_fence_waited = false;
		if (reset_recording) {
			Common::LockGuard lock(*m_slot->pool_mutex);
			m_slot->Reset();
		}
	}
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX);
	if (m_rendering && m_render_state == state) {
		return;
	}
	EndRendering();

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].sType        = vk::StructureType::eRenderingAttachmentInfo;
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp =
		    attachment.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp                 = vk::AttachmentStoreOp::eStore;
		colors[i].clearValue.color.uint32 = attachment.clear_value;
	}

	const auto&                 depth_stencil = state.depth_stencil_attachment;
	vk::RenderingAttachmentInfo depth {};
	depth.sType       = vk::StructureType::eRenderingAttachmentInfo;
	depth.imageView   = depth_stencil.image_view;
	depth.imageLayout = depth_stencil.image_layout;
	depth.loadOp =
	    depth_stencil.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth.storeOp                       = vk::AttachmentStoreOp::eStore;
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.sType       = vk::StructureType::eRenderingAttachmentInfo;
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp                         = vk::AttachmentStoreOp::eStore;
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.sType                = vk::StructureType::eRenderingInfo;
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	Handle().beginRendering(rendering);
	m_render_state = state;
	m_rendering    = true;
}

void CommandBuffer::EndRendering() const {
	if (!m_rendering) {
		return;
	}
	Handle().endRendering();
	m_rendering    = false;
	m_render_state = {};
}

} // namespace Libs::Graphics
