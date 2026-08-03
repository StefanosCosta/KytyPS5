// Regression tests for the guest-signal safety machinery in common/threads.cpp.
//
// Three defects are covered, each of which shipped in an earlier revision of this work:
//
//   1. CondVar::SignalThread collected its targets as raw pointers under the waiter-registry
//      lock and woke them with the lock released, so a waiter could return, unregister, and
//      have its owning CondVar destroyed before the wake ran. Run under ASan to catch it.
//   2. A signal deferred by HleCriticalSection had no delivery point on the unwinding edge;
//      only the wait and sleep exits dispatched, so a thread that left the section and went
//      back to guest compute held the signal indefinitely.
//
//   3. The guest-signal deferral condition applied to any non-guest instruction pointer, i.e.
//      to every HLE call rather than only to frames with a guaranteed delivery point, so
//      signals were stranded in threads merely passing through one.
//
// The fourth case is a guard rather than a regression. WaitFor deliberately dispatches once
// after its wait rather than slicing the timeout and polling between slices: the sender wakes
// the waiter through CondVar::SignalThread, and slicing would open a window between slices in
// which that wake is lost. The tests below pin both halves of that, so a future "improvement"
// to poll mid-wait fails here instead of silently dropping wakeups.

#include "common/threads.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

namespace {

using Common::CondVar;
using Common::HleCriticalSection;
using Common::Mutex;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "SignalSafetyTests: failed: %s\n", text);
		std::abort();
	}
}

std::atomic<uint32_t> g_poll_calls {0};

void CountingPoll() {
	g_poll_calls.fetch_add(1, std::memory_order_relaxed);
}

void ResetPoll() {
	g_poll_calls.store(0, std::memory_order_relaxed);
}

uint32_t PollCalls() {
	return g_poll_calls.load(std::memory_order_relaxed);
}

// Defect 1. A waiter repeatedly creates a CondVar, parks on it briefly, then destroys it,
// while a second thread hammers SignalThread against that waiter's id. The destruction races
// the wake; if SignalThread does not hold the private object alive across the gap, the wake
// touches freed memory. Nothing is asserted directly -- ASan is the oracle -- so this also
// stands as a plain no-crash stress test in a normal build.
void TestSignalThreadDoesNotOutliveCondVar() {
	std::atomic_bool  stop {false};
	std::atomic<int>  waiter_tid {0};
	std::atomic<bool> waiter_ready {false};

	std::thread waiter([&] {
		waiter_tid.store(Common::Thread::GetThreadIdUnique(), std::memory_order_release);
		waiter_ready.store(true, std::memory_order_release);

		while (!stop.load(std::memory_order_acquire)) {
			// Fresh objects each round so the destroy lands in the signaller's wake window.
			auto mutex = std::make_unique<Mutex>();
			auto cond  = std::make_unique<CondVar>();

			mutex->Lock();
			(void)cond->WaitFor(mutex.get(), 200);
			mutex->Unlock();

			cond.reset();
			mutex.reset();
		}
	});

	while (!waiter_ready.load(std::memory_order_acquire)) {
		std::this_thread::yield();
	}
	const int target = waiter_tid.load(std::memory_order_acquire);
	Check(target != 0, "waiter reported a thread id");

	std::thread signaller([&] {
		while (!stop.load(std::memory_order_acquire)) {
			CondVar::SignalThread(target);
		}
	});

	std::this_thread::sleep_for(std::chrono::milliseconds(750));
	stop.store(true, std::memory_order_release);
	signaller.join();
	waiter.join();
}

// Defect 1, second shape: many waiters on distinct condvars, all torn down while a signaller
// walks the registry. Widens the window the single-waiter loop above probes.
void TestSignalThreadWithConcurrentTeardown() {
	constexpr int WAITER_COUNT = 8;

	std::atomic_bool      stop {false};
	std::vector<std::thread> waiters;
	std::atomic<int>      ids[WAITER_COUNT];
	for (auto& id: ids) {
		id.store(0, std::memory_order_relaxed);
	}

	waiters.reserve(WAITER_COUNT);
	for (int i = 0; i < WAITER_COUNT; i++) {
		waiters.emplace_back([&, i] {
			ids[i].store(Common::Thread::GetThreadIdUnique(), std::memory_order_release);
			while (!stop.load(std::memory_order_acquire)) {
				Mutex   mutex;
				CondVar cond;
				mutex.Lock();
				(void)cond.WaitFor(&mutex, 100);
				mutex.Unlock();
			}
		});
	}

	std::thread signaller([&] {
		while (!stop.load(std::memory_order_acquire)) {
			for (auto& id: ids) {
				if (const int target = id.load(std::memory_order_acquire); target != 0) {
					CondVar::SignalThread(target);
				}
			}
		}
	});

	std::this_thread::sleep_for(std::chrono::milliseconds(750));
	stop.store(true, std::memory_order_release);
	signaller.join();
	for (auto& thread: waiters) {
		thread.join();
	}
}

// Defect 2. Leaving the outermost critical section must dispatch; leaving an inner one must
// not, or a handler would run while the outer scope still holds its lock.
void TestCriticalSectionDispatchesOnUnwind() {
	CondVar::SetWaitPollCallback(CountingPoll);
	ResetPoll();

	Check(!Common::InHleCriticalSection(), "not in a critical section to begin with");

	{
		HleCriticalSection outer;
		Check(Common::InHleCriticalSection(), "inside the outer section");
		Check(PollCalls() == 0, "entering does not dispatch");

		{
			HleCriticalSection inner;
			Check(Common::InHleCriticalSection(), "inside the nested section");
		}
		// Depth is still 1 here: the outer scope's lock is still held, so dispatching now
		// would run a guest handler underneath it.
		Check(PollCalls() == 0, "leaving a nested section does not dispatch");
		Check(Common::InHleCriticalSection(), "still inside after the nested section ends");
	}

	Check(!Common::InHleCriticalSection(), "outside once the outer section ends");
	Check(PollCalls() == 1, "leaving the outermost section dispatches exactly once");

	CondVar::SetWaitPollCallback(nullptr);
}

// A queued guest signal must not wait out a long timeout. It does not need mid-wait polling to
// avoid that: the sender follows the queue with CondVar::SignalThread, which wakes this wait, so
// the post-wait dispatch runs immediately. Assert that end-to-end -- a SignalThread against the
// waiter cuts a 3 s WaitFor short and dispatches.
//
// This also pins the reason WaitFor must not slice its timeout: between slices the thread is not
// on the condition variable, so a SignalThread landing in that window is lost entirely.
void TestSignalThreadCutsLongWaitShortAndDispatches() {
	CondVar::SetWaitPollCallback(CountingPoll);
	ResetPoll();

	Mutex   mutex;
	CondVar cond;

	std::atomic<int>  waiter_tid {0};
	std::atomic_bool  waiting {false};
	std::atomic<bool> returned {false};

	std::thread waiter([&] {
		waiter_tid.store(Common::Thread::GetThreadIdUnique(), std::memory_order_release);
		mutex.Lock();
		waiting.store(true, std::memory_order_release);
		(void)cond.WaitFor(&mutex, 3000000); // 3 s
		mutex.Unlock();
		returned.store(true, std::memory_order_release);
	});

	while (!waiting.load(std::memory_order_acquire)) {
		std::this_thread::yield();
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(50));

	const auto start = std::chrono::steady_clock::now();
	CondVar::SignalThread(waiter_tid.load(std::memory_order_acquire));
	waiter.join();
	const auto elapsed = std::chrono::steady_clock::now() - start;

	Check(returned.load(std::memory_order_acquire), "the waiter returned");
	Check(elapsed < std::chrono::milliseconds(1500),
	      "SignalThread cuts the wait short rather than letting it time out");
	Check(PollCalls() >= 1, "the woken wait dispatches pending signals");

	CondVar::SetWaitPollCallback(nullptr);
}

// The timeout must still be honoured when no callback is installed -- that path takes the
// single unsliced wait.
void TestWaitForWithoutCallbackStillTimesOut() {
	CondVar::SetWaitPollCallback(nullptr);
	ResetPoll();

	Mutex   mutex;
	CondVar cond;

	const auto start = std::chrono::steady_clock::now();
	mutex.Lock();
	const bool signalled = cond.WaitFor(&mutex, 100000); // 100 ms
	mutex.Unlock();
	const auto elapsed = std::chrono::steady_clock::now() - start;

	Check(!signalled, "reports a timeout with no callback installed");
	Check(elapsed >= std::chrono::milliseconds(80), "honours the timeout with no callback");
	Check(PollCalls() == 0, "does not dispatch when no callback is installed");
}

// A signalled WaitFor must return promptly rather than sitting out its remaining slices.
void TestWaitForWakesOnSignal() {
	CondVar::SetWaitPollCallback(CountingPoll);
	ResetPoll();

	Mutex   mutex;
	CondVar cond;

	std::atomic_bool waiting {false};
	std::thread      waker([&] {
        while (!waiting.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        cond.SignalAll();
	});

	const auto start = std::chrono::steady_clock::now();
	mutex.Lock();
	waiting.store(true, std::memory_order_release);
	(void)cond.WaitFor(&mutex, 3000000); // 3 s, deliberately far longer than the signal delay
	mutex.Unlock();
	const auto elapsed = std::chrono::steady_clock::now() - start;

	waker.join();
	Check(elapsed < std::chrono::milliseconds(1500), "returns on the signal, not on the timeout");

	CondVar::SetWaitPollCallback(nullptr);
}

// The guest-signal deferral predicate in HostSignalDispatchHandler is
//
//     InHleBlockingWait() || InHleCriticalSection()
//
// and it is only safe because both frames reach a delivery point on the way out: a wait runs
// its poll callback, and the outermost critical section dispatches as it unwinds. Ordinary host
// code has no such point, so it must NOT be a deferral frame.
//
// An earlier revision deferred on "the interrupted instruction pointer is not guest code",
// reasoning that a guest thread in host code is parked in an HLE wait. That is false -- a guest
// thread is in host code during any HLE call -- so signals were stranded in threads that were
// merely passing through and would have handled them inline. Cat Quest 3 hung before its first
// flip, waiting on a signal that sat queued until a wait that never came.
//
// These probe the two predicates from the same vantage point the signal handler has: the same
// thread, once from inside a wait (the poll callback runs there) and once from plain host code.

std::atomic<bool>     g_probe_saw_blocking {false};
std::atomic<bool>     g_probe_saw_critical {false};
std::atomic<bool>     g_probe_saw_defer {false};
std::atomic<uint32_t> g_probe_calls {0};

void ProbePoll() {
	g_probe_calls.fetch_add(1, std::memory_order_relaxed);
	g_probe_saw_blocking.store(Common::InHleBlockingWait(), std::memory_order_relaxed);
	g_probe_saw_critical.store(Common::InHleCriticalSection(), std::memory_order_relaxed);
	g_probe_saw_defer.store(Common::ShouldDeferGuestSignal(), std::memory_order_relaxed);
}

void ResetProbe() {
	g_probe_calls.store(0, std::memory_order_relaxed);
	g_probe_saw_blocking.store(false, std::memory_order_relaxed);
	g_probe_saw_critical.store(false, std::memory_order_relaxed);
	g_probe_saw_defer.store(false, std::memory_order_relaxed);
}

// Ordinary host code is not a deferral frame. This is the property the old condition broke.
void TestOrdinaryHostCodeIsNotADeferralFrame() {
	Check(!Common::ShouldDeferGuestSignal(), "plain host code does not defer");
	Check(!Common::InHleBlockingWait(), "plain host code is not a blocking wait");
	Check(!Common::InHleCriticalSection(), "plain host code is not a critical section");

	// Still not a deferral frame while holding an ordinary mutex, or inside a nested call --
	// the shapes an HLE function actually takes.
	Mutex mutex;
	mutex.Lock();
	Check(!Common::ShouldDeferGuestSignal(), "holding a plain mutex does not defer");
	Check(!Common::InHleBlockingWait(), "holding a plain mutex is not a blocking wait");
	Check(!Common::InHleCriticalSection(), "holding a plain mutex is not a critical section");
	mutex.Unlock();

	const auto nested = [] { return Common::ShouldDeferGuestSignal(); };
	Check(!nested(), "a nested host call is not a deferral frame");
}

// Inside a condvar wait it *is* a deferral frame, observed from the waiting thread itself.
void TestBlockingWaitIsADeferralFrame() {
	CondVar::SetWaitPollCallback(ProbePoll);
	ResetProbe();

	Mutex   mutex;
	CondVar cond;
	mutex.Lock();
	(void)cond.WaitFor(&mutex, 60000); // times out, so the poll callback runs
	mutex.Unlock();

	Check(g_probe_calls.load(std::memory_order_relaxed) >= 1, "the wait reached its poll callback");
	Check(g_probe_saw_blocking.load(std::memory_order_relaxed),
	      "InHleBlockingWait() is true from inside the wait");
	Check(g_probe_saw_defer.load(std::memory_order_relaxed),
	      "a signal arriving inside the wait is deferred");

	// And the marker is scoped: gone again once the wait has returned.
	Check(!Common::InHleBlockingWait(), "the blocking-wait marker clears when the wait returns");

	CondVar::SetWaitPollCallback(nullptr);
}

// The two markers are independent: a critical section is not a wait, and by the time the
// unwinding edge dispatches, the section has already been left so a handler may run.
void TestCriticalSectionMarkerIsIndependentOfWait() {
	CondVar::SetWaitPollCallback(ProbePoll);
	ResetProbe();

	{
		HleCriticalSection critical;
		Check(Common::ShouldDeferGuestSignal(), "a signal arriving in the section is deferred");
		Check(Common::InHleCriticalSection(), "inside the critical section");
		Check(!Common::InHleBlockingWait(), "a critical section is not a blocking wait");
	}

	Check(g_probe_calls.load(std::memory_order_relaxed) >= 1, "leaving the section dispatched");
	Check(!g_probe_saw_critical.load(std::memory_order_relaxed),
	      "the section has already been left when its dispatch runs, so a handler may run");
	Check(!g_probe_saw_blocking.load(std::memory_order_relaxed),
	      "that dispatch is not inside a wait either");

	CondVar::SetWaitPollCallback(nullptr);
}

} // namespace

int main() {
	TestOrdinaryHostCodeIsNotADeferralFrame();
	TestBlockingWaitIsADeferralFrame();
	TestCriticalSectionMarkerIsIndependentOfWait();
	TestCriticalSectionDispatchesOnUnwind();
	TestSignalThreadCutsLongWaitShortAndDispatches();
	TestWaitForWithoutCallbackStillTimesOut();
	TestWaitForWakesOnSignal();
	TestSignalThreadDoesNotOutliveCondVar();
	TestSignalThreadWithConcurrentTeardown();

	std::printf("SignalSafetyTests: all passed\n");
	return 0;
}
