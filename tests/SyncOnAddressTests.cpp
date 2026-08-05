#include "kernel/syncOnAddress.h"

#include "libs/errno.h"

#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

using Libs::LibKernel::SyncOnAddress::Wait32;
using Libs::LibKernel::SyncOnAddress::Wait64;
using Libs::LibKernel::SyncOnAddress::Wake;

std::atomic<int> g_signal_poll_count{0};

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "SyncOnAddressTests: failed: %s\n", text);
    std::abort();
  }
}

template <typename T> void Store(T *address, T value) {
  std::atomic_ref<T>(*address).store(value, std::memory_order_release);
}

void CountSignalPoll() {
  g_signal_poll_count.fetch_add(1, std::memory_order_relaxed);
}

void TestInvalidAddress() {
  uint32_t timeout = 1;
  Check(Wait32(nullptr, 0, &timeout) == Libs::LibKernel::KERNEL_ERROR_EINVAL,
        "wait32 rejects a null address");
  Check(Wait64(nullptr, 0, &timeout) == Libs::LibKernel::KERNEL_ERROR_EINVAL,
        "wait64 rejects a null address");
  Check(Wake(nullptr, 1) == Libs::LibKernel::KERNEL_ERROR_EINVAL,
        "wake rejects a null address");

  alignas(uint64_t) uint8_t bytes[16] = {};
  auto *misaligned = reinterpret_cast<uint32_t *>(bytes + 1);
  Check(Wait32(misaligned, 0, &timeout) == Libs::LibKernel::KERNEL_ERROR_EINVAL,
        "wait32 rejects a misaligned address");
  Check(Wait64(reinterpret_cast<uint64_t *>(bytes + 4), 0, &timeout) ==
            Libs::LibKernel::KERNEL_ERROR_EINVAL,
        "wait64 rejects a misaligned address");
  Check(Wake(misaligned, 1) == Libs::LibKernel::KERNEL_ERROR_EINVAL,
        "wake rejects a misaligned address");
  uint64_t aligned = 0;
  Check(Wake(&aligned, -1) == Libs::LibKernel::KERNEL_ERROR_EINVAL,
        "wake rejects a negative count");
}

void TestMismatchReturnsImmediately() {
  uint32_t word = 7;
  uint64_t word64 = UINT64_C(0x100000000);
  uint32_t timeout = 500000;
  const auto start = std::chrono::steady_clock::now();
  Check(Wait32(&word, 6, &timeout, CountSignalPoll) == OK,
        "mismatch succeeds without parking");
  // Only proves the *pre-park* compare is full width: 0x100000000 != 0 is seen before the loop
  // ever reaches the futex. It says nothing about the kernel's own compare, which is 32-bit --
  // TestWait64HighWordChange covers that.
  Check(Wait64(&word64, 0, &timeout, CountSignalPoll) == OK,
        "wait64 pre-park compare is full width");
  Check(std::chrono::steady_clock::now() - start <
            std::chrono::milliseconds(100),
        "mismatched value is a fast path");
  Check(g_signal_poll_count.load(std::memory_order_relaxed) >= 2,
        "mismatch remains a guest signal safe-point");
}

void TestTimeout() {
  uint32_t word = 0;
  uint32_t timeout = 20000;
  const auto start = std::chrono::steady_clock::now();
  Check(Wait32(&word, 0, &timeout) == Libs::LibKernel::KERNEL_ERROR_ETIMEDOUT,
        "matching value times out");
  const auto elapsed = std::chrono::steady_clock::now() - start;
  Check(elapsed >= std::chrono::milliseconds(10),
        "timeout does not return too early");
  Check(elapsed < std::chrono::milliseconds(500), "timeout remains bounded");

  timeout = 0;
  Check(Wait32(&word, 0, &timeout) == Libs::LibKernel::KERNEL_ERROR_ETIMEDOUT,
        "zero timeout polls a matching value");
  word = 1;
  Check(Wait32(&word, 0, &timeout) == OK,
        "zero timeout succeeds for a mismatched value");
}

void TestValueChangeAndWake() {
  uint64_t word = 0;
  uint32_t timeout = 1000000;
  std::atomic<bool> ready{false};
  int result = Libs::LibKernel::KERNEL_ERROR_ETIMEDOUT;

  std::thread waiter([&] {
    ready.store(true, std::memory_order_release);
    result = Wait64(&word, 0, &timeout);
  });
  while (!ready.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  Store(&word, UINT64_C(0x100000000));
  Check(Wake(&word, 1) == OK, "wake-one succeeds");
  waiter.join();
  Check(result == OK, "a value change plus wake releases the waiter");
}

void TestWakeOneThenAll() {
  constexpr int WAITER_COUNT = 4;
  uint32_t word = 0;
  uint32_t timeout = 1000000;
  std::atomic<int> ready{0};
  std::atomic<int> returned{0};
  int results[WAITER_COUNT] = {};
  std::vector<std::thread> waiters;

  for (int i = 0; i < WAITER_COUNT; i++) {
    waiters.emplace_back([&, i] {
      ready.fetch_add(1, std::memory_order_release);
      results[i] = Wait32(&word, 0, &timeout);
      returned.fetch_add(1, std::memory_order_release);
    });
  }
  while (ready.load(std::memory_order_acquire) != WAITER_COUNT) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  Check(Wake(&word, 1) == OK, "wake-one succeeds with multiple waiters");
  const auto one_deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (returned.load(std::memory_order_acquire) == 0 &&
         std::chrono::steady_clock::now() < one_deadline) {
    std::this_thread::yield();
  }
  Check(returned.load(std::memory_order_acquire) == 1,
        "wake-one releases exactly one waiter");

  Check(Wake(&word, 2) == OK, "wake-two succeeds");
  const auto two_deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (returned.load(std::memory_order_acquire) < 3 &&
         std::chrono::steady_clock::now() < two_deadline) {
    std::this_thread::yield();
  }
  Check(returned.load(std::memory_order_acquire) == 3,
        "wake-two releases exactly two more waiters");

  Check(Wake(&word, INT_MAX) == OK, "wake-all succeeds");
  for (auto &waiter : waiters) {
    waiter.join();
  }
  Check(returned.load(std::memory_order_acquire) == WAITER_COUNT,
        "wake-all releases the remaining waiters");
  for (int result : results) {
    Check(result == OK, "explicitly woken waiters return success");
  }
}

void TestAddressesAreIsolated() {
  uint32_t first = 0;
  uint32_t second = 0;
  uint32_t first_timeout = 1000000;
  uint32_t second_timeout = 1000000;
  std::atomic<int> ready{0};
  std::atomic<bool> first_returned{false};
  std::atomic<bool> second_returned{false};
  int first_result = 0;
  int second_result = 0;

  std::thread first_waiter([&] {
    ready.fetch_add(1, std::memory_order_release);
    first_result = Wait32(&first, 0, &first_timeout);
    first_returned.store(true, std::memory_order_release);
  });
  std::thread second_waiter([&] {
    ready.fetch_add(1, std::memory_order_release);
    second_result = Wait32(&second, 0, &second_timeout);
    second_returned.store(true, std::memory_order_release);
  });
  while (ready.load(std::memory_order_acquire) != 2) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  Check(Wake(&first, 1) == OK, "first address wakes");
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  Check(first_returned.load(std::memory_order_acquire),
        "first address waiter returned");
  Check(!second_returned.load(std::memory_order_acquire),
        "waking one address does not release another address");
  Check(Wake(&second, 1) == OK, "second address wakes");

  first_waiter.join();
  second_waiter.join();
  Check(first_result == OK && second_result == OK,
        "isolated waiters return success");
}

void TestCompareRegisterWakeRace() {
  for (int i = 0; i < 100; i++) {
    uint32_t word = 0;
    uint32_t timeout = 500000;
    std::atomic<bool> ready{false};
    int result = Libs::LibKernel::KERNEL_ERROR_ETIMEDOUT;
    std::thread waiter([&] {
      ready.store(true, std::memory_order_release);
      result = Wait32(&word, 0, &timeout);
    });
    while (!ready.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    Store(&word, uint32_t{1});
    (void)Wake(&word, 1);
    waiter.join();
    Check(result == OK, "compare/register/wake race never loses progress");
  }
}

void TestWakeZeroIsNoOp() {
  constexpr int WAITER_COUNT = 2;
  uint32_t word = 0;
  uint32_t timeout = 1000000;
  std::atomic<int> ready{0};
  std::atomic<int> returned{0};
  int results[WAITER_COUNT] = {};
  std::vector<std::thread> waiters;

  for (int i = 0; i < WAITER_COUNT; i++) {
    waiters.emplace_back([&, i] {
      ready.fetch_add(1, std::memory_order_release);
      results[i] = Wait32(&word, 0, &timeout);
      returned.fetch_add(1, std::memory_order_release);
    });
  }
  while (ready.load(std::memory_order_acquire) != WAITER_COUNT) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  Check(Wake(&word, 0) == OK, "zero-count wake succeeds");
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  Check(returned.load(std::memory_order_acquire) == 0,
        "zero-count wake releases no waiters");
  Check(Wake(&word, INT_MAX) == OK, "wake-all succeeds after zero-count wake");
  for (auto &waiter : waiters) {
    waiter.join();
  }
  for (int result : results) {
    Check(result == OK, "wake-all releases waiters after zero-count no-op");
  }
}

} // namespace

// Linux FUTEX_WAIT compares 32 bits, so a 64-bit waiter parked with a matching low half does not
// see a store that touches only the high half; the emulator's own full-width re-check has to catch
// it. This is the only test that gets a Wait64 genuinely parked in the kernel and then changes the
// half the kernel cannot see -- and it deliberately issues no Wake, because a Wake would release the
// waiter regardless and hide the defect. It asserts on *latency*: the unfixed code still returns OK,
// just a poll period late.
void TestWait64HighWordChange() {
  constexpr int TRIALS = 5;
  // Comfortably above the 1 ms backstop, comfortably below the 10 ms signal-poll period, so this
  // discriminates the two rather than measuring scheduler noise.
  constexpr auto BUDGET = std::chrono::milliseconds(4);

  for (int trial = 0; trial < TRIALS; trial++) {
    uint64_t word = 0; // low half matches `expected`, so the kernel will park us
    uint32_t timeout = 1000000;
    std::atomic<bool> ready{false};
    int result = Libs::LibKernel::KERNEL_ERROR_ETIMEDOUT;
    std::chrono::steady_clock::time_point returned{};

    std::thread waiter([&] {
      ready.store(true, std::memory_order_release);
      result = Wait64(&word, 0, &timeout);
      returned = std::chrono::steady_clock::now();
    });
    while (!ready.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    // Long enough that the waiter is definitely inside FUTEX_WAIT, not still spinning up.
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    const auto stored = std::chrono::steady_clock::now();
    Store(&word, UINT64_C(0x100000000)); // high half only; low half stays 0
    waiter.join();

    Check(result == OK, "wait64 returns OK after a high-word-only change");
    Check(returned - stored < BUDGET,
          "wait64 notices a high-word-only change promptly");
  }
}

// A parked futex waiter is released only by FUTEX_WAKE, never by a store, so an explicit Wake must
// still cut the wait short rather than leaving it to the backstop period. Guards the interaction
// between the shortened 64-bit poll and the wake path: TestValueChangeAndWake proves a Wake
// eventually releases the waiter, but would pass even if it took a full poll period to do so.
void TestWait64WakeIsPrompterThanThePoll() {
  uint64_t word = 0;
  uint32_t timeout = 1000000;
  std::atomic<bool> ready{false};
  int result = Libs::LibKernel::KERNEL_ERROR_ETIMEDOUT;
  std::chrono::steady_clock::time_point returned{};

  std::thread waiter([&] {
    ready.store(true, std::memory_order_release);
    result = Wait64(&word, 0, &timeout);
    returned = std::chrono::steady_clock::now();
  });
  while (!ready.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(5));

  const auto woke = std::chrono::steady_clock::now();
  Store(&word, UINT64_C(0x100000000));
  Check(Wake(&word, 1) == OK, "wake-one succeeds for a 64-bit waiter");
  waiter.join();

  Check(result == OK, "wait64 returns OK after a wake");
  Check(returned - woke < std::chrono::milliseconds(2),
        "an explicit wake releases a 64-bit waiter without waiting for the poll");
}

int main() {
  TestInvalidAddress();
  TestMismatchReturnsImmediately();
  TestTimeout();
  TestValueChangeAndWake();
  TestWakeOneThenAll();
  TestAddressesAreIsolated();
  TestCompareRegisterWakeRace();
  TestWakeZeroIsNoOp();
  TestWait64HighWordChange();
  TestWait64WakeIsPrompterThanThePoll();
  std::printf("SyncOnAddressTests: all passed\n");
  return 0;
}
