#ifndef KYTY_COMMON_THREADS_H_
#define KYTY_COMMON_THREADS_H_

#include "common/common.h"
#include "common/subsystems.h"

#include <memory>
#include <string>

namespace Common {

KYTY_SUBSYSTEM_DEFINE(Threads);

using thread_func_t    = void (*)(void*);
using wait_poll_func_t = void (*)();

struct ThreadPrivate;
struct MutexPrivate;
struct CondVarPrivate;

class Thread {
public:
	Thread(thread_func_t func, void* arg);
	~Thread();

	void Join();
	void Detach();

	// Once a thread has finished, the id may be reused by another thread.
	[[nodiscard]] std::string GetId() const;

	// The id is unique and can't be reused by another thread.
	[[nodiscard]] int GetUniqueId() const;

	static void Sleep(uint32_t millis);
	static void SleepMicro(uint32_t micros);
	static void SleepNano(uint64_t nanos);
	static bool IsMainThread();

	// Get current thread id
	// Once a thread has finished, the id may be reused by another thread.
	static std::string GetThreadId();

	// Get current thread id
	// The id is unique and can't be reused by another thread.
	static int GetThreadIdUnique();

	KYTY_CLASS_NO_COPY(Thread);

private:
	std::unique_ptr<ThreadPrivate> m_thread;
};

class Mutex {
public:
	Mutex();
	~Mutex();

	void Lock();
	void Unlock();
	bool TryLock();

	friend class CondVar;

	KYTY_CLASS_NO_COPY(Mutex);

private:
	std::unique_ptr<MutexPrivate> m_mutex;
};

class CondVar {
public:
	CondVar();
	~CondVar();

	void Wait(Mutex* mutex);
	bool WaitFor(Mutex* mutex, uint32_t micros);
	void Signal();
	void SignalAll();

	static void SignalThread(int thread_id);
	static void SetWaitPollCallback(wait_poll_func_t callback);

	KYTY_CLASS_NO_COPY(CondVar);

private:
	// Shared rather than unique: SignalThread collects its targets under the waiter-registry
	// lock and wakes them with it released (waking while holding it deadlocks the waker against
	// the waiters it is retiring). Between the two, a waiter can return, unregister, and have
	// its owning CondVar destroyed -- so the registry has to keep the private object alive.
	std::shared_ptr<CondVarPrivate> m_cond_var;
};

class LockGuard {
public:
	using mutex_type = Mutex;

	// NOLINTNEXTLINE(google-runtime-references)
	explicit LockGuard(mutex_type& m): m_mutex(m) { m_mutex.Lock(); }

	~LockGuard() { m_mutex.Unlock(); }

	KYTY_CLASS_NO_COPY(LockGuard);

private:
	mutex_type& m_mutex;
};

// A guest exception handler can block for an unbounded time -- IL2CPP's garbage collector parks
// every managed thread in one until the collection finishes -- so one must never run while this
// thread holds a lock that another guest thread needs to make progress. Scopes that take an
// emulator-global lock declare themselves here, and pending-signal dispatch refuses to run a
// handler inside one, leaving the signal queued until the scope unwinds.
class HleCriticalSection {
public:
	HleCriticalSection();
	~HleCriticalSection();

	KYTY_CLASS_NO_COPY(HleCriticalSection);
};

[[nodiscard]] bool InHleCriticalSection();

// Marks a scope in which this thread is blocked inside a host condition-variable wait.
//
// Running a guest exception handler nested inside such a wait is what deadlocks the emulator:
// glibc holds a group reference on the condition variable for the whole of pthread_cond_wait,
// and a handler that blocks (IL2CPP's GC suspend handler waits on an event flag until the
// collection ends) never lets the wait unwind, stranding that reference. Every later broadcast
// on the variable then blocks forever.
//
// Deferral is scoped to exactly this: a wait always reaches its poll callback on the way out,
// so a signal queued here has a guaranteed delivery point. Host code in general does not --
// deferring for any non-guest instruction pointer strands signals in threads that were merely
// passing through an HLE call and would have handled them inline.
class HleBlockingWait {
public:
	HleBlockingWait();
	~HleBlockingWait();

	KYTY_CLASS_NO_COPY(HleBlockingWait);
};

[[nodiscard]] bool InHleBlockingWait();

// Whether a guest signal arriving on this thread right now must be queued rather than run.
//
// The policy in one place: defer only in frames that reach a delivery point on the way out --
// a wait runs its poll callback, and the outermost critical section dispatches as it unwinds.
// Ordinary host code has neither, so a signal deferred there would sit queued until some
// unrelated wait happened to occur, which for some titles never does.
[[nodiscard]] bool ShouldDeferGuestSignal();

} // namespace Common

#endif /* KYTY_COMMON_THREADS_H_ */
