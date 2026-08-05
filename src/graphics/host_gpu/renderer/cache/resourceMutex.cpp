#include "graphics/host_gpu/renderer/cache/resourceMutex.h"

#include "common/assert.h"

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

namespace {

// Deliberately duplicated from TrackingSpinLock rather than shared: resource_mutex_tests compiles
// only this translation unit, and reaching for regionManager.h would drag the page manager in.
[[nodiscard]] uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return GetCurrentThreadId();
#elif defined(__APPLE__)
	// mach thread port is a nonzero per-thread id (0 is the "no owner" sentinel).
	return static_cast<uint32_t>(pthread_mach_thread_np(pthread_self()));
#elif defined(__linux__)
	static thread_local const uint32_t tid = static_cast<uint32_t>(::syscall(SYS_gettid));
	return tid;
#else
	EXIT("resource transaction thread identity is unsupported on this platform\n");
#endif
}

} // namespace

ResourceMutex::~ResourceMutex() {
	if (m_resource_owner.load(std::memory_order_relaxed) != 0) {
		EXIT("ResourceMutex destroyed with an active owner\n");
	}
}

void ResourceMutex::lock() {
	const auto current = CurrentThread();
	// Safe to read without holding m_resource: only this thread can ever store its own id, so a
	// match is necessarily our own earlier acquisition, and program order makes it visible to us.
	if (m_resource_owner.load(std::memory_order_relaxed) == current) {
		EXIT("recursive resource transaction\n");
	}
	m_resource.lock();
	if (m_resource_owner.load(std::memory_order_relaxed) != 0) {
		EXIT("resource mutex acquired with a stale owner\n");
	}
	m_resource_owner.store(current, std::memory_order_relaxed);
}

void ResourceMutex::unlock() {
	if (m_resource_owner.load(std::memory_order_relaxed) != CurrentThread()) {
		EXIT("resource transaction released without ownership\n");
	}
	// Clear before releasing, so the next owner cannot observe a stale id.
	m_resource_owner.store(0, std::memory_order_relaxed);
	m_resource.unlock();
}

bool ResourceMutex::IsOwnedByCurrentThread() {
	return m_resource_owner.load(std::memory_order_relaxed) == CurrentThread();
}

} // namespace Libs::Graphics
