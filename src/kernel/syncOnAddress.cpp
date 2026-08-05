#include "kernel/syncOnAddress.h"

#include "common/threads.h"
#include "libs/errno.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
#include <cerrno>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::LibKernel::SyncOnAddress {

namespace {

// How long a waiter may sleep before it must surface to run PollSignals. This is an emulator
// device, not a PS5 one -- a real kernel futex blocks until woken -- so it only bounds deferred
// guest-signal delivery, and it matches the budget semaphore.cpp and pthread.cpp use.
constexpr uint32_t SIGNAL_POLL_MICROS = 10000;

// Linux FUTEX_WAIT always compares 32 bits, so for a 64-bit wait the kernel filters on the low half
// only: a store touching just the high half can park a waiter that should have returned. Every exit
// re-checks the full width, so this is a latency question and never a wrong answer -- but here the
// poll is the *correctness backstop* for that case, not merely signal delivery, so it has to be
// shorter. 32-bit waits keep SIGNAL_POLL_MICROS: their kernel compare is exact and needs no backstop.
constexpr uint32_t PARTIAL_COMPARE_POLL_MICROS = 1000;

using Clock = std::chrono::steady_clock;

template <typename T>
[[nodiscard]] bool IsValidWaitAddress(const volatile T* address) {
	return address != nullptr && (reinterpret_cast<uintptr_t>(address) & (alignof(T) - 1u)) == 0;
}

[[nodiscard]] bool IsValidWakeAddress(const volatile void* address) {
	return address != nullptr &&
	       (reinterpret_cast<uintptr_t>(address) & (alignof(uint32_t) - 1u)) == 0;
}

template <typename T>
[[nodiscard]] T ReadWord(const volatile T* address) {
	return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

struct WaitDeadline {
	bool              finite = false;
	Clock::time_point end {};
};

[[nodiscard]] WaitDeadline MakeDeadline(const uint32_t* timeout_micros) {
	if (timeout_micros == nullptr) {
		return {};
	}
	return {true, Clock::now() + std::chrono::microseconds(*timeout_micros)};
}

[[nodiscard]] uint32_t GetWaitSliceMicros(const WaitDeadline& deadline, bool first_wait,
                                          uint32_t poll_micros) {
	if (!deadline.finite) {
		return poll_micros;
	}

	const auto now = Clock::now();
	if (now >= deadline.end) {
		return first_wait ? 0u : UINT32_MAX;
	}

	const auto remaining =
	    std::chrono::duration_cast<std::chrono::microseconds>(deadline.end - now).count();
	return static_cast<uint32_t>(std::min<int64_t>(remaining, static_cast<int64_t>(poll_micros)));
}

void PollSignals(signal_poll_func_t signal_poll) {
	if (signal_poll != nullptr) {
		signal_poll();
	}
}

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)

template <typename T>
int WaitLinux(volatile T* address, T expected, const uint32_t* timeout_micros,
              signal_poll_func_t signal_poll) {
	// See PARTIAL_COMPARE_POLL_MICROS: the kernel's compare is exact at 32 bits and partial at 64.
	constexpr uint32_t poll_micros =
	    sizeof(T) > sizeof(uint32_t) ? PARTIAL_COMPARE_POLL_MICROS : SIGNAL_POLL_MICROS;

	const auto deadline   = MakeDeadline(timeout_micros);
	bool       first_wait = true;

	for (;;) {
		const auto slice_micros = GetWaitSliceMicros(deadline, first_wait, poll_micros);
		if (slice_micros == UINT32_MAX) {
			return ReadWord(address) == expected ? KERNEL_ERROR_ETIMEDOUT : OK;
		}
		const timespec timeout = {
		    .tv_sec  = static_cast<time_t>(slice_micros / 1000000u),
		    .tv_nsec = static_cast<long>(slice_micros % 1000000u) * 1000L,
		};

		// Full-width, and deliberately the last thing before the syscall: the kernel's own compare
		// may be narrower than T, so this is what actually decides whether parking is correct.
		// Keeping it adjacent to the syscall makes the window a store can slip through as small as
		// it can be made.
		if (ReadWord(address) != expected) {
			return OK;
		}

		long result     = 0;
		int  wait_error = 0;
		result          = syscall(SYS_futex, const_cast<T*>(address), FUTEX_WAIT_PRIVATE,
		                          static_cast<uint32_t>(expected), &timeout, nullptr, 0);
		if (result != 0) {
			wait_error = errno;
		}
		if (result == 0 || wait_error == EAGAIN) {
			return OK;
		}
		if (wait_error != ETIMEDOUT && wait_error != EINTR) {
			return KERNEL_ERROR_EINVAL;
		}

		PollSignals(signal_poll);
		if (deadline.finite && Clock::now() >= deadline.end) {
			return ReadWord(address) == expected ? KERNEL_ERROR_ETIMEDOUT : OK;
		}
		first_wait = false;
	}
}

int WakeLinux(volatile void* address, int32_t count) {
	// The legacy FUTEX_WAKE path may wake one waiter when count is zero.
	if (count == 0) {
		return OK;
	}

	const auto result = syscall(SYS_futex, const_cast<void*>(address), FUTEX_WAKE_PRIVATE, count,
	                            nullptr, nullptr, 0);
	return result < 0 ? KERNEL_ERROR_EINVAL : OK;
}

#endif

struct PortableWaiter {
	Common::CondVar condition;
	bool            wake_requested = false;
};

struct PortableAddressEntry {
	Common::Mutex              mutex;
	std::list<PortableWaiter*> waiters;
};

struct PortableAddressRegistry {
	std::mutex                                                           mutex;
	std::unordered_map<uintptr_t, std::shared_ptr<PortableAddressEntry>> entries;
};

PortableAddressRegistry& GetPortableRegistry() {
	static PortableAddressRegistry registry;
	return registry;
}

std::shared_ptr<PortableAddressEntry> RegisterPortableWaiter(volatile void*  address,
                                                             PortableWaiter* waiter) {
	auto&           registry = GetPortableRegistry();
	std::lock_guard registry_lock(registry.mutex);
	auto&           entry = registry.entries[reinterpret_cast<uintptr_t>(address)];
	if (!entry) {
		entry = std::make_shared<PortableAddressEntry>();
	}
	entry->mutex.Lock();
	entry->waiters.push_back(waiter);
	return entry;
}

void UnregisterPortableWaiter(volatile void*                               address,
                              const std::shared_ptr<PortableAddressEntry>& entry,
                              PortableWaiter*                              waiter) {
	entry->waiters.remove(waiter);
	const bool empty = entry->waiters.empty();
	entry->mutex.Unlock();
	if (!empty) {
		return;
	}

	auto&           registry = GetPortableRegistry();
	std::lock_guard registry_lock(registry.mutex);
	entry->mutex.Lock();
	const auto it = registry.entries.find(reinterpret_cast<uintptr_t>(address));
	if (it != registry.entries.end() && it->second == entry && entry->waiters.empty()) {
		registry.entries.erase(it);
	}
	entry->mutex.Unlock();
}

template <typename T>
int WaitPortable(volatile T* address, T expected, const uint32_t* timeout_micros,
                 signal_poll_func_t signal_poll) {
	PortableWaiter waiter;
	auto           entry      = RegisterPortableWaiter(address, &waiter);
	const auto     deadline   = MakeDeadline(timeout_micros);
	bool           first_wait = true;
	int            result     = OK;

	while (ReadWord(address) == expected && !waiter.wake_requested) {
		const auto slice_micros = GetWaitSliceMicros(deadline, first_wait, SIGNAL_POLL_MICROS);
		if (slice_micros == UINT32_MAX) {
			result = KERNEL_ERROR_ETIMEDOUT;
			break;
		}
		if (slice_micros == 0) {
			result = KERNEL_ERROR_ETIMEDOUT;
			break;
		}

		(void)waiter.condition.WaitFor(&entry->mutex, slice_micros);
		entry->mutex.Unlock();
		PollSignals(signal_poll);
		entry->mutex.Lock();

		if (deadline.finite && Clock::now() >= deadline.end && ReadWord(address) == expected &&
		    !waiter.wake_requested) {
			result = KERNEL_ERROR_ETIMEDOUT;
			break;
		}
		first_wait = false;
	}

	UnregisterPortableWaiter(address, entry, &waiter);
	return result;
}

int WakePortable(volatile void* address, int32_t count) {
	if (count == 0) {
		return OK;
	}
	auto&            registry = GetPortableRegistry();
	std::unique_lock registry_lock(registry.mutex);
	const auto       it = registry.entries.find(reinterpret_cast<uintptr_t>(address));
	if (it == registry.entries.end()) {
		return OK;
	}
	auto entry = it->second;
	entry->mutex.Lock();
	registry_lock.unlock();

	int32_t remaining = count;
	for (auto* waiter: entry->waiters) {
		if (!waiter->wake_requested) {
			waiter->wake_requested = true;
			waiter->condition.Signal();
			if (remaining != INT_MAX && --remaining == 0) {
				break;
			}
		}
	}
	entry->mutex.Unlock();
	return OK;
}

template <typename T>
int WaitImpl(volatile T* address, T expected, const uint32_t* timeout_micros,
             signal_poll_func_t signal_poll) {
	if (!IsValidWaitAddress(address)) {
		return KERNEL_ERROR_EINVAL;
	}

	int result = OK;
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
	result = WaitLinux(address, expected, timeout_micros, signal_poll);
#else
	result = WaitPortable(address, expected, timeout_micros, signal_poll);
#endif
	PollSignals(signal_poll);
	return result;
}

} // namespace

int Wait32(volatile uint32_t* address, uint32_t expected, const uint32_t* timeout_micros,
           signal_poll_func_t signal_poll) {
	return WaitImpl(address, expected, timeout_micros, signal_poll);
}

int Wait64(volatile uint64_t* address, uint64_t expected, const uint32_t* timeout_micros,
           signal_poll_func_t signal_poll) {
	return WaitImpl(address, expected, timeout_micros, signal_poll);
}

int Wake(volatile void* address, int32_t count) {
	if (!IsValidWakeAddress(address) || count < 0) {
		return KERNEL_ERROR_EINVAL;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
	return WakeLinux(address, count);
#endif
	return WakePortable(address, count);
}

} // namespace Libs::LibKernel::SyncOnAddress
