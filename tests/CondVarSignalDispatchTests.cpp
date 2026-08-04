// CondVar::WaitFor must dispatch pending guest signals.
//
// The host signal handler can leave a guest signal queued rather than running the handler in the
// signal frame; the queued signal is then delivered by the wait-poll callback at a safe point.
// Wait() polls through its 10 ms timeout and dispatches on each round, but WaitFor() takes the
// caller's timeout, which can be seconds. Without a dispatch here the signal sits undelivered for
// the whole wait.
//
// The dispatch has to happen with the guest mutex released: a guest handler may block for an
// unbounded time, and doing that while holding a lock the guest also uses deadlocks the emulator.

#include "common/threads.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {

std::atomic<int> g_polls {0};

void PollCallback() {
	g_polls.fetch_add(1, std::memory_order_relaxed);
}

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "CondVarSignalDispatchTests: failed: %s\n", text);
		std::abort();
	}
}

// An unsignalled WaitFor must still dispatch before it returns.
void TestWaitForDispatchesPendingSignals() {
	g_polls.store(0, std::memory_order_relaxed);

	Common::Mutex   mutex;
	Common::CondVar cond;
	mutex.Lock();
	const bool signalled = cond.WaitFor(&mutex, 200000); // 200 ms
	mutex.Unlock();

	Check(!signalled, "an unsignalled WaitFor reports a timeout");
	Check(g_polls.load(std::memory_order_relaxed) > 0,
	      "WaitFor returned without dispatching pending guest signals");
}

// Control: Wait() has always dispatched, so a failure here means the callback is not wired up and
// the case above would be vacuous.
void TestWaitDispatchesPendingSignals() {
	g_polls.store(0, std::memory_order_relaxed);

	Common::Mutex   mutex;
	Common::CondVar cond;
	std::thread     waker([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        cond.Signal();
	});

	mutex.Lock();
	cond.Wait(&mutex);
	mutex.Unlock();
	waker.join();

	Check(g_polls.load(std::memory_order_relaxed) > 0, "Wait did not dispatch pending signals");
}

} // namespace

int main() {
	Common::CondVar::SetWaitPollCallback(&PollCallback);

	TestWaitDispatchesPendingSignals();
	TestWaitForDispatchesPendingSignals();

	std::printf("CondVarSignalDispatchTests: all passed\n");
	return 0;
}
