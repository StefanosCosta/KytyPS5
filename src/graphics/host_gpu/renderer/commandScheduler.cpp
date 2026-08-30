#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "common/assert.h"
#include "common/syncStats.h"

#include <cstdio>
#include <cstdlib>
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

void CommandSlot::Reset() {
	EXIT_IF(buffer == nullptr);
	const auto result = buffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
	if (result != vk::Result::eSuccess) {
		EXIT("failed to reset Vulkan command buffer: %s (%d)\n", VulkanToString(result).c_str(),
		     static_cast<int>(result));
	}
}

CommandScheduler::CommandPool::~CommandPool() {
	Destroy();
}

void CommandScheduler::CommandPool::Create(GraphicContext& graphics) {
	EXIT_IF(m_pool != nullptr || m_graphics != nullptr ||
	        graphics.queue_family == static_cast<uint32_t>(-1));
	m_graphics = &graphics;

	vk::CommandPoolCreateInfo create {};
	create.sType            = vk::StructureType::eCommandPoolCreateInfo;
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandSlot* CommandScheduler::CommandPool::CreateSlot() {
	EXIT_IF(m_graphics == nullptr);
	auto& graphics = *m_graphics;

	vk::CommandBufferAllocateInfo allocate {};
	allocate.sType              = vk::StructureType::eCommandBufferAllocateInfo;
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = 1;
	vk::CommandBuffer buffer    = nullptr;
	EXIT_IF(graphics.device.allocateCommandBuffers(&allocate, &buffer) != vk::Result::eSuccess);

	vk::FenceCreateInfo fence_create {};
	fence_create.sType = vk::StructureType::eFenceCreateInfo;
	fence_create.flags = vk::FenceCreateFlagBits::eSignaled;
	vk::Fence fence    = nullptr;
	if (graphics.device.createFence(&fence_create, nullptr, &fence) != vk::Result::eSuccess) {
		graphics.device.freeCommandBuffers(m_pool, 1, &buffer);
		EXIT("failed to create command-buffer fence\n");
	}

	auto& slot      = m_slots.emplace_back();
	slot.pool_mutex = &m_mutex;
	slot.id         = static_cast<uint32_t>(m_slots.size() - 1);
	slot.buffer     = buffer;
	slot.fence      = fence;
	return &slot;
}

CommandSlot* CommandScheduler::CommandPool::Allocate(GraphicContext& graphics) {
	Common::LockGuard lock(m_mutex);
	if (m_pool == nullptr) {
		Create(graphics);
	}
	EXIT_IF(m_graphics != &graphics);
	auto  found = std::ranges::find_if(m_slots, [](const auto& slot) { return !slot.busy; });
	auto* slot  = found != m_slots.end() ? &*found : CreateSlot();
	slot->busy  = true;
	slot->Reset();
	return slot;
}

void CommandScheduler::CommandPool::Destroy() {
	Common::LockGuard lock(m_mutex);
	if (m_pool == nullptr) {
		return;
	}
	EXIT_IF(std::ranges::any_of(m_slots, [](const auto& slot) { return slot.busy; }));
	EXIT_IF(m_graphics == nullptr);
	for (const auto& slot: m_slots) {
		m_graphics->device.destroyFence(slot.fence, nullptr);
	}
	m_graphics->device.destroyCommandPool(m_pool, nullptr);
	m_slots.clear();
	m_pool     = nullptr;
	m_graphics = nullptr;
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {
	m_buffers.reserve(CommandBufferGrowStep);
	m_buffer_ticks.reserve(CommandBufferGrowStep);
}

CommandScheduler::~CommandScheduler() {
	Shutdown();
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (Active() && m_recording) {
		Finish();
	}
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_registers   = &registers;
	m_user_config = &user_config;
	m_shaders     = &shaders;

	if (!Active()) {
		m_current = static_cast<int>(GrowCommandBuffers());
	}

	BindCurrent();
	if (!m_recording) {
		Current().Begin();
		m_recording = true;
	}
}

void CommandScheduler::BeginRendering(const RenderState& state) {
	Current().BeginRendering(state);
}

void CommandScheduler::EndRendering() {
	if (Active() && m_recording) {
		Current().EndRendering();
	}
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	SubmitCurrent(submit);
	BeginNext();
}

CommandBuffer& CommandScheduler::FlushAndGetSubmitted() {
	SubmitInfo submit;
	auto&      submitted = SubmitCurrent(submit);
	BeginNext();
	return submitted;
}

void CommandScheduler::Finish() {
	CheckActive();
	const auto tick = CurrentTick();
	if (m_recording) {
		SubmitInfo submit;
		SubmitCurrent(submit);
	}
	for (auto& buffer: m_buffers) {
		buffer->WaitForFenceAndReset();
	}
	m_master.Wait(tick);
	PopPendingOperations();
	BindCurrent();
	Current().Begin();
	m_recording = true;
}

void CommandScheduler::FinishCurrent() {
	SubmitInfo submit;
	auto&      submitted = SubmitCurrent(submit);
	submitted.WaitForFenceAndReset();
	m_master.Refresh();
	PopPendingOperations();
	submitted.Begin();
	m_recording = true;
}

void CommandScheduler::Wait(uint64_t tick) {
	CheckActive();
	EXIT_IF(tick > CurrentTick());
	if (tick >= CurrentTick()) {
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). Recycle the same command object so that reference remains valid. Deferred
		// resources are released only at the next GPU operation boundary.
		SubmitInfo submit;
		auto&      submitted = SubmitCurrent(submit);
		submitted.WaitForFenceAndReset();
		m_master.Refresh();
		submitted.Begin();
		m_recording = true;
		return;
	}
	m_master.Wait(tick);
}

void CommandScheduler::PopPendingOperations() {
	PopPendingOperations(true);
}

void CommandScheduler::PopPendingOperations(bool refresh_gpu_tick) {
	if (refresh_gpu_tick) {
		m_master.Refresh();
	}
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty() ||
			    !m_master.IsFree(m_pending_operations.front().tick)) {
				return;
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_pending_operations.push({std::move(operation), CurrentTick()});
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_priority_operations.push({std::move(operation), CurrentTick()});
		lock.unlock();
		m_operation_available.notify_one();
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active      = true;
			m_priority_active_tick = operation.tick;
		}
		// +1 so that "never waited" (0) is distinguishable from "waiting on tick 0".
		Common::SyncStats::SetGauge(Common::SyncStats::Gauge::PriorityWaitTick,
		                            operation.tick + 1);
		{
			// The wait is vkWaitSemaphores with UINT64_MAX: if this tick is never signalled
			// the thread parks here forever and every later flip completion queues behind it.
			Common::SyncStats::Scope wait_scope(Common::SyncStats::Site::PriorityOpWait);
			m_master.Wait(operation.tick);
		}
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active      = false;
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && !m_priority_active; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(lock, [this, tick] {
		const bool active_before_or_at = m_priority_active && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	});
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

CommandSlot* CommandScheduler::AllocateCommandBuffer() {
	return m_command_pool.Allocate(m_graphics);
}

uint64_t CommandScheduler::NextSubmitSequence() noexcept {
	return m_submit_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active() || static_cast<size_t>(m_current) >= m_buffers.size());
}

RenderCommandBuffer& CommandScheduler::Current() const {
	CheckActive();
	EXIT_IF(m_buffers[m_current] == nullptr);
	return *m_buffers[m_current];
}

void CommandScheduler::BindCurrent() const {
	EXIT_IF(m_registers == nullptr || m_user_config == nullptr || m_shaders == nullptr);
	Current().Bind(*m_registers, *m_user_config, *m_shaders);
}

CommandBuffer& CommandScheduler::SubmitCurrent(SubmitInfo& submit) {
	CheckActive();
	EXIT_IF(!m_recording);
	auto& submitted = Current();
	submitted.End();
	const auto signal_tick = m_master.NextTick();
	WorkLog::RecordSubmission(signal_tick, static_cast<uint32_t>(m_current),
	                          m_submit_sequence.load(std::memory_order_relaxed));
	submit.AddSignal(m_master.Handle(), signal_tick);
	submitted.Execute(submit);
	m_buffer_ticks[static_cast<size_t>(m_current)] = signal_tick;
	m_recording                                    = false;
	return submitted;
}

int CommandScheduler::FindReusableBuffer(uint64_t gpu_tick) const {
	EXIT_IF(m_buffer_ticks.size() != m_buffers.size());
	const auto buffer_count = m_buffers.size();
	for (size_t offset = 1; offset <= buffer_count; ++offset) {
		const auto candidate = (static_cast<size_t>(m_current) + offset) % buffer_count;
		if (gpu_tick >= m_buffer_ticks[candidate]) {
			return static_cast<int>(candidate);
		}
	}
	return -1;
}

void CommandScheduler::DumpOutstanding(uint64_t known_gpu_tick) const {
	std::fprintf(stderr, "  scheduler: %zu command buffers, current=%d\n", m_buffers.size(),
	             m_current);
	uint32_t ahead = 0;
	uint32_t pending = 0;
	// Report by tick order, not buffer index: the interesting one is the lowest outstanding
	// tick -- the submission the whole timeline is waiting on.
	for (uint64_t tick = known_gpu_tick + 1; tick <= known_gpu_tick + 4; ++tick) {
		bool found = false;
		for (size_t i = 0; i < m_buffers.size() && i < m_buffer_ticks.size(); ++i) {
			if (m_buffer_ticks[i] != tick) {
				continue;
			}
			found = true;
			std::fprintf(stderr, "    tick=%llu -> buffer[%zu] fence=%s\n",
			             static_cast<unsigned long long>(tick), i,
			             m_buffers[i]->IsSubmissionOutstanding() ? "PENDING" : "signalled");
		}
		if (!found) {
			std::fprintf(stderr,
			             "    tick=%llu -> NO BUFFER HOLDS THIS TICK (it was handed out but the "
			             "slot was reused or never submitted)\n",
			             static_cast<unsigned long long>(tick));
		}
	}
	for (size_t i = 0; i < m_buffers.size() && i < m_buffer_ticks.size(); ++i) {
		if (m_buffer_ticks[i] <= known_gpu_tick) {
			continue;
		}
		ahead++;
		if (m_buffers[i]->IsSubmissionOutstanding()) {
			pending++;
		}
	}
	std::fprintf(stderr, "  %u buffers ahead of the GPU, %u with a pending fence\n", ahead,
	             pending);
	std::fflush(stderr);
}

size_t CommandScheduler::GrowCommandBuffers() {
	const auto first = m_buffers.size();
	const auto end   = first + CommandBufferGrowStep;
	m_buffers.reserve(end);
	m_buffer_ticks.reserve(end);
	for (size_t i = first; i < end; ++i) {
		m_buffers.emplace_back(std::make_unique<RenderCommandBuffer>(*this));
		m_buffer_ticks.push_back(0);
	}
	return first;
}

void CommandScheduler::BeginNext() {
	EXIT_IF(m_recording);

	auto candidate = FindReusableBuffer(m_master.KnownGpuTick());
	if (candidate < 0) {
		m_master.Refresh();
		candidate = FindReusableBuffer(m_master.KnownGpuTick());
	}
	if (candidate < 0) {
		candidate = static_cast<int>(GrowCommandBuffers());
	}

	m_current = candidate;
	Current().WaitForFenceAndReset();
	BindCurrent();
	Current().Begin();
	m_recording = true;
}


namespace WorkLog {

namespace {

struct Record {
	std::atomic<uint64_t> tick {0}; // 0 = never written
	uint64_t              submit_seq        = 0;
	uint32_t              slot              = 0;
	uint32_t              draws             = 0;
	uint32_t              dispatches        = 0;
	uint32_t              tiler_dispatches  = 0;
	uint64_t              tiler_invocations = 0;
};

// Ticks are dense and the backlog observed at a stall is ~68, so 1024 is ample history.
constexpr uint64_t RING = 1024;
Record             g_ring[RING];

struct DrawDetail {
	std::atomic<uint64_t> tick {0};
	uint64_t              pipeline       = 0;
	const uint32_t*       vs_spirv       = nullptr;
	uint32_t              vs_words       = 0;
	const uint32_t*       ps_spirv       = nullptr;
	uint32_t              ps_words       = 0;
	uint32_t              index_count    = 0;
	uint32_t              instance_count = 0;
};

// A stalled buffer held 25 draws, so 512 covers many buffers of history.
constexpr size_t      DRAW_RING = 512;
DrawDetail            g_draw_ring[DRAW_RING];
std::atomic<size_t>   g_draw_ring_next {0};

std::atomic<uint32_t> g_draws {0};
std::atomic<uint32_t> g_dispatches {0};
std::atomic<uint32_t> g_tiler_dispatches {0};
std::atomic<uint64_t> g_tiler_invocations {0};

} // namespace

void NoteDraw() noexcept {
	g_draws.fetch_add(1, std::memory_order_relaxed);
}

void NoteDispatch() noexcept {
	g_dispatches.fetch_add(1, std::memory_order_relaxed);
}

void NoteDrawDetail(uint64_t tick, uint64_t pipeline, const uint32_t* vs_spirv, uint32_t vs_words,
                    const uint32_t* ps_spirv, uint32_t ps_words, uint32_t index_count,
                    uint32_t instance_count) noexcept {
	if (!Common::SyncStats::Enabled(2)) {
		return;
	}
	auto& entry = g_draw_ring[g_draw_ring_next.fetch_add(1, std::memory_order_relaxed) % DRAW_RING];
	entry.pipeline       = pipeline;
	entry.vs_spirv       = vs_spirv;
	entry.vs_words       = vs_words;
	entry.ps_spirv       = ps_spirv;
	entry.ps_words       = ps_words;
	entry.index_count    = index_count;
	entry.instance_count = instance_count;
	entry.tick.store(tick, std::memory_order_release);
}

void NoteTilerDispatch(uint64_t invocations) noexcept {
	g_tiler_dispatches.fetch_add(1, std::memory_order_relaxed);
	g_tiler_invocations.fetch_add(invocations, std::memory_order_relaxed);
}

void RecordSubmission(uint64_t tick, uint32_t slot, uint64_t submit_seq) noexcept {
	if (!Common::SyncStats::Enabled(2)) {
		return;
	}
	// Exchange rather than load: each field becomes "what this submission added".
	auto& record             = g_ring[tick % RING];
	record.submit_seq        = submit_seq;
	record.slot              = slot;
	record.draws             = g_draws.exchange(0, std::memory_order_relaxed);
	record.dispatches        = g_dispatches.exchange(0, std::memory_order_relaxed);
	record.tiler_dispatches  = g_tiler_dispatches.exchange(0, std::memory_order_relaxed);
	record.tiler_invocations = g_tiler_invocations.exchange(0, std::memory_order_relaxed);
	// Published last: a reader that sees the tick sees the fields above it.
	record.tick.store(tick, std::memory_order_release);
}

bool DumpStuck(uint64_t tick, uint64_t known_gpu_tick, uint64_t issued_tick) noexcept {
	const auto& record = g_ring[tick % RING];
	if (record.tick.load(std::memory_order_acquire) != tick) {
		std::fprintf(stderr,
		             "\n[gpu-stall] tick %llu never completed; its record was overwritten "
		             "(gpu_tick=%llu issued=%llu)\n",
		             static_cast<unsigned long long>(tick),
		             static_cast<unsigned long long>(known_gpu_tick),
		             static_cast<unsigned long long>(issued_tick));
		return false;
	}
	std::fprintf(stderr,
	             "\n[gpu-stall] tick %llu never completed (gpu_tick=%llu issued=%llu backlog=%lld)\n"
	             "  slot=%u submit_seq=%llu  draws=%u  dispatches=%u  "
	             "tiler_dispatches=%u  tiler_invocations=%llu\n",
	             static_cast<unsigned long long>(tick),
	             static_cast<unsigned long long>(known_gpu_tick),
	             static_cast<unsigned long long>(issued_tick),
	             static_cast<long long>(issued_tick) - static_cast<long long>(known_gpu_tick),
	             record.slot, static_cast<unsigned long long>(record.submit_seq), record.draws,
	             record.dispatches, record.tiler_dispatches,
	             static_cast<unsigned long long>(record.tiler_invocations));

	// The neighbours put it in context: an empty stuck buffer next to busy ones means the
	// stall is not about this buffer's contents.
	for (uint64_t probe = (tick > 2 ? tick - 2 : 0); probe <= tick + 2; probe++) {
		const auto& other = g_ring[probe % RING];
		if (other.tick.load(std::memory_order_acquire) != probe) {
			continue;
		}
		std::fprintf(stderr, "    tick %llu%s draws=%u dispatches=%u tiler=%u\n",
		             static_cast<unsigned long long>(probe), probe == tick ? " <-- stuck" : "",
		             other.draws, other.dispatches, other.tiler_dispatches);
	}

	std::fprintf(stderr, "  draws recorded for tick %llu:\n",
	             static_cast<unsigned long long>(tick));
	uint32_t index = 0;
	for (const auto& entry: g_draw_ring) {
		if (entry.tick.load(std::memory_order_acquire) != tick) {
			continue;
		}
		std::fprintf(stderr,
		             "    [%2u] pipeline=0x%016llx vs_words=%u ps_words=%u  indices=%u "
		             "instances=%u\n",
		             index++, static_cast<unsigned long long>(entry.pipeline), entry.vs_words,
		             entry.ps_words, entry.index_count, entry.instance_count);
		if (index > 1) {
			continue; // all 25 draws share a pipeline; one dump is enough
		}
		// The SPIR-V is owned by the pipeline cache and outlives the draw, so it is safe to
		// write out here. This is the module the GPU wedged on.
		const char* dir = std::getenv("KYTY_STALL_DUMP_DIR");
		if (dir == nullptr) {
			continue;
		}
		for (int which = 0; which < 2; ++which) {
			const uint32_t* code  = which == 0 ? entry.vs_spirv : entry.ps_spirv;
			const uint32_t  words = which == 0 ? entry.vs_words : entry.ps_words;
			if (code == nullptr || words == 0) {
				continue;
			}
			char path[512];
			std::snprintf(path, sizeof(path), "%s/stall_%s_%llu.spv", dir,
			              which == 0 ? "vs" : "ps", static_cast<unsigned long long>(tick));
			if (auto* out = std::fopen(path, "wb"); out != nullptr) {
				std::fwrite(code, sizeof(uint32_t), words, out);
				std::fclose(out);
				std::fprintf(stderr, "    wrote %s (%u words)\n", path, words);
			}
		}
	}
	std::fflush(stderr);
	return true;
}

} // namespace WorkLog

} // namespace Libs::Graphics
