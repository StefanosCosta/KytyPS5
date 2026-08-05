#include "graphics/host_gpu/renderer/cache/resourceMutex.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#endif

namespace {

using Libs::Graphics::ResourceMutex;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "ResourceMutexTests: failed: %s\n", text);
		std::abort();
	}
}

void YieldMany() {
	for (uint32_t i = 0; i < 4096; i++) {
		std::this_thread::yield();
	}
}

void TestOwnership() {
	ResourceMutex mutex;
	Check(!mutex.IsOwnedByCurrentThread(), "fresh mutex reported an owner");
	{
		std::lock_guard lock(mutex);
		Check(mutex.IsOwnedByCurrentThread(), "owner tracking was not installed");
	}
	Check(!mutex.IsOwnedByCurrentThread(), "owner tracking survived unlock");
}

// Ownership is recorded as a bare thread id, so the risk is a thread mistaking someone else's
// ownership for its own -- which would let a recursive-acquisition check pass when it should abort,
// or a release-without-ownership check miss. Assert from a second thread, both while the lock is
// held elsewhere and while it is free.
void TestOwnershipIsPerThread() {
	ResourceMutex    mutex;
	std::atomic_bool observed_while_held {true};
	std::atomic_bool observed_while_free {true};

	{
		std::lock_guard lock(mutex);
		Check(mutex.IsOwnedByCurrentThread(), "holder did not see itself as owner");
		std::thread observer(
		    [&] { observed_while_held.store(mutex.IsOwnedByCurrentThread(), std::memory_order_release); });
		observer.join();
	}
	std::thread observer([&] {
		observed_while_free.store(mutex.IsOwnedByCurrentThread(), std::memory_order_release);
	});
	observer.join();

	Check(!observed_while_held.load(std::memory_order_acquire),
	      "a non-owning thread claimed ownership of a held transaction");
	Check(!observed_while_free.load(std::memory_order_acquire),
	      "a thread claimed ownership of a free transaction");
}

// A thread id must not be mistaken for another's after threads come and go. Repeated acquire/release
// across many short-lived threads is where a stale or recycled owner would surface.
void TestOwnershipAcrossThreadChurn() {
	ResourceMutex mutex;
	for (int i = 0; i < 200; i++) {
		std::atomic_bool ok {false};
		std::thread      worker([&] {
			Check(!mutex.IsOwnedByCurrentThread(), "fresh thread inherited ownership");
			{
				std::lock_guard lock(mutex);
				ok.store(mutex.IsOwnedByCurrentThread(), std::memory_order_release);
			}
			Check(!mutex.IsOwnedByCurrentThread(), "ownership survived unlock on a worker");
		});
		worker.join();
		Check(ok.load(std::memory_order_acquire), "worker did not take ownership");
		Check(!mutex.IsOwnedByCurrentThread(), "main thread inherited a retired thread's ownership");
	}
}

void TestSerializesTransactions() {
	ResourceMutex    mutex;
	std::unique_lock owner(mutex);
	std::atomic_bool contender_started {false};
	std::atomic_bool contender_entered {false};
	std::thread      contender([&] {
		contender_started.store(true, std::memory_order_release);
		std::lock_guard lock(mutex);
		contender_entered.store(true, std::memory_order_release);
	});
	while (!contender_started.load(std::memory_order_acquire)) {
		std::this_thread::yield();
	}
	YieldMany();
	Check(!contender_entered.load(std::memory_order_acquire),
	      "contender entered an active resource transaction");
	owner.unlock();
	contender.join();
	Check(contender_entered.load(std::memory_order_acquire),
	      "contender did not resume after resource transaction");
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
[[noreturn]] void RunDeathCase() {
	ResourceMutex mutex;
	std::lock_guard first(mutex);
	std::lock_guard second(mutex);
	std::_Exit(0x7f);
}

void CheckDeathCase() {
	char path[MAX_PATH] {};
	Check(GetModuleFileNameA(nullptr, path, MAX_PATH) != 0, "GetModuleFileName failed");
	std::string       command = std::string("\"") + path + "\" --death";
	std::vector<char> mutable_command(command.begin(), command.end());
	mutable_command.push_back('\0');
	STARTUPINFOA        startup {sizeof(startup)};
	PROCESS_INFORMATION process {};
	Check(CreateProcessA(nullptr, mutable_command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
	                     nullptr, nullptr, &startup, &process) != 0,
	      "CreateProcess failed");
	const auto wait = WaitForSingleObject(process.hProcess, 10000);
	if (wait != WAIT_OBJECT_0) {
		TerminateProcess(process.hProcess, 0x7e);
	}
	Check(wait == WAIT_OBJECT_0, "ResourceMutex death test timed out");
	DWORD exit_code = 0;
	Check(GetExitCodeProcess(process.hProcess, &exit_code) != 0 && exit_code == 321,
	      "ResourceMutex death path used the wrong exit");
	CloseHandle(process.hThread);
	CloseHandle(process.hProcess);
}
#endif

} // namespace

int main(int argc, char** argv) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (argc == 2 && std::strcmp(argv[1], "--death") == 0) {
		RunDeathCase();
	}
#else
	(void)argc;
	(void)argv;
#endif
	TestOwnership();
	TestOwnershipIsPerThread();
	TestOwnershipAcrossThreadChurn();
	TestSerializesTransactions();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	CheckDeathCase();
#endif
	std::puts("ResourceMutexTests: all cases passed");
	return 0;
}
