#include "graphics/host_gpu/regionManager.h"

#include <atomic>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

using Libs::Graphics::TrackingSpinLock;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "TrackingSpinLockTests: failed: %s\n", text);
		std::abort();
	}
}

// Guest page faults are serviced inline on the faulting thread, so this lock is taken by however
// many guest threads happen to fault at once. Cat Quest III fields ~10.
constexpr int      THREADS    = 8;
constexpr int      ITERATIONS = 20000;
constexpr uint64_t EXPECTED   = static_cast<uint64_t>(THREADS) * ITERATIONS;

void TestUncontendedLockUnlock() {
	TrackingSpinLock lock;
	for (int i = 0; i < 1000; i++) {
		lock.lock();
		lock.unlock();
	}
	// Reaching here at all is the assertion: unlock() EXITs if the owner does not match, so a lock
	// that lost track of its owner would abort rather than return.
	Check(true, "uncontended lock/unlock round trips");
}

// The counter is deliberately NOT atomic: it is only safe if the lock genuinely excludes, so a lost
// update is the signal that mutual exclusion broke.
void TestMutualExclusion() {
	TrackingSpinLock lock;
	uint64_t         counter = 0;

	std::vector<std::thread> threads;
	threads.reserve(THREADS);
	for (int t = 0; t < THREADS; t++) {
		threads.emplace_back([&] {
			for (int i = 0; i < ITERATIONS; i++) {
				lock.lock();
				counter++;
				lock.unlock();
			}
		});
	}
	for (auto& thread: threads) {
		thread.join();
	}

	Check(counter == EXPECTED, "spin lock lost an update under contention");
}

[[nodiscard]] double ProcessCpuSeconds() {
	timespec ts {};
	if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) != 0) {
		return 0.0;
	}
	return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) / 1e9;
}

// The regression this guards is not correctness -- the old bare test_and_set loop excluded perfectly
// well. It is that waiters *burned whole cores* while excluded, because nothing in the loop yielded
// or paused. Once guest page faults began being serviced inline on the faulting thread, ~10 Unity
// workers hit this lock at once and the emulator went from 268% CPU to ~840%, taking Cat Quest III
// from 60 fps to 7 (Milestone 54).
//
// So the thing to measure is CPU consumed, not wall time: a lock that is merely slow still finishes,
// but a lock that spins hot converts every waiting thread into a busy core. The critical section is
// held for a few microseconds here on purpose -- with a nanosecond-long one no convoy forms and the
// two implementations are indistinguishable.
constexpr int ROUNDS     = 200;
constexpr int WORK_SPINS = 20000; // tens of microseconds, so waiters genuinely queue behind the owner

// Runs `rounds` critical sections across `threads` threads and returns the process CPU seconds spent.
// Callers keep total useful work constant across thread counts so the results are comparable.
[[nodiscard]] double MeasureCpuSeconds(int threads, int rounds, uint64_t* counter_out) {
	TrackingSpinLock lock;
	uint64_t         counter = 0;

	const auto               before = ProcessCpuSeconds();
	std::vector<std::thread> workers;
	workers.reserve(static_cast<size_t>(threads));
	for (int t = 0; t < threads; t++) {
		workers.emplace_back([&] {
			for (int i = 0; i < rounds; i++) {
				lock.lock();
				for (int spin = 0; spin < WORK_SPINS; spin++) {
					counter++;
				}
				lock.unlock();
			}
		});
	}
	for (auto& worker: workers) {
		worker.join();
	}
	*counter_out = counter;
	return ProcessCpuSeconds() - before;
}

void TestContendedWaitersDoNotBurnCpu() {
	constexpr uint64_t TOTAL = static_cast<uint64_t>(THREADS) * ROUNDS * WORK_SPINS;

	// Same total work, no contention: this is what the useful work alone costs on this machine, so
	// the comparison calibrates itself instead of hard-coding a wall-clock number.
	uint64_t   solo_counter = 0;
	const auto solo         = MeasureCpuSeconds(1, THREADS * ROUNDS, &solo_counter);
	Check(solo_counter == TOTAL, "uncontended run lost an update");

	uint64_t   contended_counter = 0;
	const auto contended         = MeasureCpuSeconds(THREADS, ROUNDS, &contended_counter);
	Check(contended_counter == TOTAL, "contended run lost an update");

	const auto overhead = solo > 0.0 ? contended / solo : 0.0;
	std::printf("TrackingSpinLockTests: cpu solo %.3f s, contended %.3f s, overhead %.1fx\n", solo,
	            contended, overhead);
	std::fflush(stdout);

	// The metric is CPU *consumed*, not wall time and not cores-busy. Both of those were tried and
	// neither discriminates: with spare cores a yielding waiter is rescheduled immediately, so
	// cores-busy sits at ~7x for either implementation, and a nanosecond-long critical section
	// produces no convoy at all (1.6x). Total CPU against an uncontended run of the same work is
	// what separates them.
	//
	// Calibrated on this machine, 5 runs each: polite 56.6-62.6x, bare test_and_set 164.5-239.9x.
	// 110x sits between with ~1.8x headroom above the polite range and ~1.5x below the bare one.
	// Expect single-digit-percent variance -- if this ever goes red, check the box is idle before
	// assuming the lock regressed.
	Check(overhead < 110.0, "contended waiters burned CPU while excluded");
}

// A waiter must not observe the lock as free while another thread holds it, even briefly. Holding
// the lock across a short sleep makes any such window wide enough to catch.
void TestHeldLockExcludesWaiters() {
	TrackingSpinLock lock;
	std::atomic<int> inside {0};
	std::atomic<int> max_inside {0};

	std::vector<std::thread> threads;
	threads.reserve(THREADS);
	for (int t = 0; t < THREADS; t++) {
		threads.emplace_back([&] {
			for (int i = 0; i < 200; i++) {
				lock.lock();
				const auto now = inside.fetch_add(1, std::memory_order_acq_rel) + 1;
				auto       observed = max_inside.load(std::memory_order_relaxed);
				while (now > observed &&
				       !max_inside.compare_exchange_weak(observed, now, std::memory_order_relaxed)) {
				}
				std::this_thread::sleep_for(std::chrono::microseconds(20));
				inside.fetch_sub(1, std::memory_order_acq_rel);
				lock.unlock();
			}
		});
	}
	for (auto& thread: threads) {
		thread.join();
	}

	Check(max_inside.load(std::memory_order_relaxed) == 1,
	      "more than one thread was inside the critical section");
}

} // namespace

int main() {
	TestUncontendedLockUnlock();
	TestMutualExclusion();
	TestHeldLockExcludesWaiters();
	TestContendedWaitersDoNotBurnCpu();
	std::puts("TrackingSpinLockTests: all cases passed");
	return 0;
}
