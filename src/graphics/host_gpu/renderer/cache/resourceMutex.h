#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RESOURCEMUTEX_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RESOURCEMUTEX_H_

#include "common/common.h"

#include <atomic>
#include <mutex>

namespace Libs::Graphics {

// Owner-tracked shared buffer/image transaction.
//
// The owner is an atomic thread id rather than a std::thread::id guarded by a second mutex. This
// lock is taken once per buffer binding per draw, so that bookkeeping cost three mutex acquisitions
// per lock() and showed up as ~4.5% of the command-processor thread. 0 is the "no owner" sentinel,
// which no platform hands out as a thread id -- the same scheme TrackingSpinLock uses in
// graphics/host_gpu/regionManager.h.
class ResourceMutex final {
public:
	ResourceMutex() = default;
	~ResourceMutex();
	KYTY_CLASS_NO_COPY(ResourceMutex);

	void               lock();
	void               unlock();
	[[nodiscard]] bool IsOwnedByCurrentThread();

private:
	std::mutex           m_resource;
	std::atomic_uint32_t m_resource_owner {0};
};

static_assert(std::atomic_uint32_t::is_always_lock_free);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RESOURCEMUTEX_H_
