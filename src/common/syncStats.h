#ifndef KYTY_COMMON_SYNCSTATS_H_
#define KYTY_COMMON_SYNCSTATS_H_

#include "common/common.h"

// Wall-clock attribution for the emulator's CPU/GPU synchronization points.
//
// Diagnostic only, and inert unless KYTY_GPU_SYNC_STATS is set in the environment:
//
//   1  blocking sites only (the default for measurement runs)
//   2  adds sites that are individually sub-microsecond, where the two clock reads
//      would otherwise be a measurable fraction of what they measure
//   3  adds a per-site per-thread breakdown to the report
//
// Everything on the hot path is two clock reads plus a relaxed add into a per-thread
// row, so sites may be instrumented without perturbing what they report.

namespace Common::SyncStats {

// Timed sites. The Frame* group is nested: FrameDoneTotal spans the other two, so
// their times are reported hierarchically rather than subtracted.
enum class Site : uint32_t
{
	// Seven sites from the original list are gone: the GPU fences they measured were removed
	// upstream between f06d7e6 and bed19e5 (1d5f3c8 "remove unnecessary Vulkan synchronization",
	// ace549a, and the IncrementDe/TriggerEvent/RELEASE_MEM simplifications). Their absence is the
	// win, so there is nothing left to time -- do not re-add them without a live fence to wrap.
	FrameDoneTotal = 0,   // GuestGpu::Done, whole call
	FrameDoneQueueDrain,  //   WaitForIdle
	PredicationGpuWait,   // SetPredication -> BufferFlushAndWait
	GdsReadGpuWait,       // end-of-pipe GDS read -> SynchronizeGpu
	ReadbackGpuWait,      // BufferCache::ReadMemoryOnGpu -> FinishCurrent
	CpBlockedPoll,        // command processor, all queue heads blocked
	FramePoolAcquire,     // waiting for a free presenter frame
	SubmitSlotWait,       // waiting for a flip queue slot
	PresentAcquireImage,  // vkAcquireNextImageKHR
	PresentRenderMutex,   // present thread holding the renderer mutex
	SubmissionMutex,      // GpuMutexLock, level 2
	PriorityOpWait,       // priority-operations thread blocked in MasterSemaphore::Wait
	Count
};

// Counted occurrences, no timing.
enum class Event : uint32_t
{
	GuestFrame = 0,   // Gpu::Done completed
	Present,          // Presenter::Present succeeded
	FlipTick,         // one iteration of the vblank pacer
	FlipTickIdle,     //   ... nothing queued
	FlipTickNotReady, //   ... front request not Ready yet
	FlipTickNotDue,   //   ... front request not due yet
	CpBlockedRound,   // one pass with every queue head blocked
	Count
};

// Summed magnitudes -- "how much", where Event answers "how many times". Reported as a
// per-second rate over the interval, like the Event lines.
enum class Counter : uint32_t
{
	TilerDetilePasses = 0,      // TileManager::Detile
	TilerTilePasses,            // TileManager::TileImage
	TilerConvertD16Passes,      // TileManager::ConvertD16
	TilerSwapBgraPasses,        // TileManager::SwapBgra16
	TilerDispatches,            // individual vkCmdDispatch issued by the tiler
	TilerInvocations,           // width * height * depth, summed over those dispatches
	TilerAtomicInvocations,     // ... of which take the sub-dword path: 2 device atomics each
	TilerScratchBytes,          // AllocateScratch, i.e. fresh vmaCreateBuffer per pass
	TilerClearBytes,            // fillBuffer(target, .., 0) preceding a pass
	ImageUploadBytes,           // guest surface bytes re-uploaded by TextureCache::UploadImage
	ImageUploadCpuDirty,        // ... because the whole surface was marked m_cpu_dirty
	ImageUploadMaybeCpuDirty,   // ... because of a page-range overlap (m_maybe_cpu_dirty)
	ImageUploadBufferModified,  // ... because the buffer cache reported a GPU write
	Count
};

// Sampled maxima.
enum class Gauge : uint32_t
{
	BufferCacheEntries = 0,
	TilerMaxInvocations,        // largest single tiler dispatch seen in the interval
	// Timeline state, sampled from the vblank pacer so it keeps updating after every
	// other thread has gone quiet. Ticks are monotonic, so a max-gauge is the latest value.
	GpuTickKnown,               // MasterSemaphore::KnownGpuTick after a fresh Refresh
	GpuTickCurrent,             // MasterSemaphore::CurrentTick, i.e. ticks handed out
	PriorityWaitTick,           // tick the priority thread is currently blocked on, +1
	Count
};

// Set once, before main, from the environment. Depends on nothing, so it is safe to
// read from any static initializer.
extern const uint32_t g_level;

[[nodiscard]] inline bool Enabled(uint32_t level = 1) noexcept
{
	return g_level >= level;
}

// Starts the reporter thread. No-op when disabled.
void Start();

void CountEvent(Event event) noexcept;
void AddCounter(Counter counter, uint64_t amount) noexcept;
void Record(Site site, uint64_t ticks) noexcept;
void SetGauge(Gauge gauge, uint64_t value) noexcept;

[[nodiscard]] uint64_t Now() noexcept;

class Scope final
{
public:
	explicit Scope(Site site, uint32_t level = 1) noexcept
	    : m_site(site), m_active(Enabled(level)), m_begin(m_active ? Now() : 0)
	{
	}

	~Scope()
	{
		if (m_active)
		{
			Record(m_site, Now() - m_begin);
		}
	}

	KYTY_CLASS_NO_COPY(Scope);

private:
	Site     m_site;
	bool     m_active;
	uint64_t m_begin;
};

} // namespace Common::SyncStats

#endif /* KYTY_COMMON_SYNCSTATS_H_ */
