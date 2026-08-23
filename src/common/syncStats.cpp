#include "common/syncStats.h"

#include "common/threads.h"
#include "common/timer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>

namespace Common::SyncStats {

namespace {

// Subnautica runs ~70 host threads, so 64 slots would alias two threads onto one row and
// misattribute the "busiest tid" column. 256 keeps every thread distinct for ~4x headroom.
constexpr uint32_t THREAD_SLOTS    = 256;
constexpr uint32_t REPORT_INTERVAL = 5;

constexpr auto SITE_COUNT  = static_cast<uint32_t>(Site::Count);
constexpr auto EVENT_COUNT = static_cast<uint32_t>(Event::Count);
constexpr auto GAUGE_COUNT = static_cast<uint32_t>(Gauge::Count);

// One row per (thread slot, site). Threads never share a row, so there is no
// contention and no false sharing; the reporter sums the column.
struct alignas(64) Row {
	std::atomic<uint64_t> calls {0};
	std::atomic<uint64_t> ticks {0};
	std::atomic<uint64_t> max_ticks {0};
	std::atomic<int>      tid {-1};
};

struct SiteInfo {
	const char* name;
	uint32_t    depth;
};

// Order must match the Site enum.
constexpr SiteInfo SITE_INFO[SITE_COUNT] = {
    {"FrameDoneTotal", 0},      {"FrameDoneQueueDrain", 1}, {"PredicationGpuWait", 0},
    {"GdsReadGpuWait", 0},      {"ReadbackGpuWait", 0},     {"CpBlockedPoll", 0},
    {"FramePoolAcquire", 0},    {"SubmitSlotWait", 0},      {"PresentAcquireImage", 0},
    {"PresentRenderMutex", 0},  {"SubmissionMutex", 0},
};

Row                   g_rows[THREAD_SLOTS][SITE_COUNT];
std::atomic<uint64_t> g_events[EVENT_COUNT];
std::atomic<uint64_t> g_gauges[GAUGE_COUNT];

uint32_t ReadLevel() {
	const char* value = std::getenv("KYTY_GPU_SYNC_STATS"); // NOLINT(concurrency-mt-unsafe)
	if (value == nullptr) {
		return 0;
	}
	const auto level = std::strtoul(value, nullptr, 10);
	return static_cast<uint32_t>(level > 3 ? 3 : level);
}

// Snapshot of everything the reporter needs to produce one interval.
struct Snapshot {
	uint64_t events[EVENT_COUNT] {};
	uint64_t calls[SITE_COUNT] {};
	uint64_t ticks[SITE_COUNT] {};
};

void Report() {
	const auto frequency = std::max(Common::Timer::QueryPerformanceFrequency(), uint64_t {1});

	Snapshot previous;
	auto     previous_time = Now();

	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(REPORT_INTERVAL));

		Snapshot current;
		// Per-interval maxima and the busiest thread, collected while snapshotting.
		uint64_t max_ticks[SITE_COUNT] {};
		uint64_t busiest_ticks[SITE_COUNT] {};
		int      busiest_tid[SITE_COUNT] {};

		for (uint32_t site = 0; site < SITE_COUNT; site++) {
			busiest_tid[site] = -1;
			for (auto& slot: g_rows) {
				auto& row = slot[site];

				const auto calls = row.calls.load(std::memory_order_relaxed);
				const auto ticks = row.ticks.load(std::memory_order_relaxed);
				current.calls[site] += calls;
				current.ticks[site] += ticks;

				// Reset so the reported maximum is per-interval, not since start.
				const auto row_max = row.max_ticks.exchange(0, std::memory_order_relaxed);
				max_ticks[site]    = std::max(max_ticks[site], row_max);

				if (ticks > busiest_ticks[site]) {
					busiest_ticks[site] = ticks;
					busiest_tid[site]   = row.tid.load(std::memory_order_relaxed);
				}
			}
		}
		for (uint32_t event = 0; event < EVENT_COUNT; event++) {
			current.events[event] = g_events[event].load(std::memory_order_relaxed);
		}

		const auto now     = Now();
		const auto seconds = static_cast<double>(now - previous_time) / static_cast<double>(frequency);
		previous_time      = now;
		if (seconds <= 0.0) {
			continue;
		}

		auto event_delta = [&](Event event) {
			const auto index = static_cast<uint32_t>(event);
			return current.events[index] - previous.events[index];
		};

		const auto frames   = event_delta(Event::GuestFrame);
		const auto presents = event_delta(Event::Present);
		const auto ticks    = event_delta(Event::FlipTick);

		// stderr, not LOGF: LOGF short-circuits on Silent and takes the log lock, which
		// would perturb the presentation thread this is trying to measure.
		std::fprintf(stderr, "\n[gpu-sync %.2fs] guest_frames=%llu (%.1f/s)  presents=%llu (%.1f/s)\n",
		             seconds, static_cast<unsigned long long>(frames),
		             static_cast<double>(frames) / seconds,
		             static_cast<unsigned long long>(presents),
		             static_cast<double>(presents) / seconds);
		std::fprintf(stderr,
		             "  flip ticks=%llu (%.1f/s)  idle=%llu  not_ready=%llu  not_due=%llu  "
		             "presented=%llu  cp_blocked_rounds=%llu\n",
		             static_cast<unsigned long long>(ticks), static_cast<double>(ticks) / seconds,
		             static_cast<unsigned long long>(event_delta(Event::FlipTickIdle)),
		             static_cast<unsigned long long>(event_delta(Event::FlipTickNotReady)),
		             static_cast<unsigned long long>(event_delta(Event::FlipTickNotDue)),
		             static_cast<unsigned long long>(presents),
		             static_cast<unsigned long long>(event_delta(Event::CpBlockedRound)));
		std::fprintf(stderr, "  %-24s %9s %14s %8s %10s %10s %8s\n", "site", "calls/s",
		             "blocked ms/s", "%wall", "mean us", "max us", "tid");

		for (uint32_t site = 0; site < SITE_COUNT; site++) {
			const auto calls = current.calls[site] - previous.calls[site];
			if (calls == 0) {
				continue;
			}
			const auto delta_ticks = current.ticks[site] - previous.ticks[site];
			const auto micros      = static_cast<double>(delta_ticks) * 1e6 /
			                    static_cast<double>(frequency);
			const auto blocked_ms = micros / 1e3 / seconds;

			char label[40];
			std::snprintf(label, sizeof(label), "%*s%s",
			              static_cast<int>(SITE_INFO[site].depth * 2), "", SITE_INFO[site].name);

			std::fprintf(stderr, "  %-24s %9.1f %14.1f %7.1f%% %10.0f %10.0f %8d\n", label,
			             static_cast<double>(calls) / seconds, blocked_ms, blocked_ms / 10.0,
			             micros / static_cast<double>(calls),
			             static_cast<double>(max_ticks[site]) * 1e6 /
			                 static_cast<double>(frequency),
			             busiest_tid[site]);
		}

		if (Enabled(2)) {
			std::fprintf(stderr, "  gauge buffer_cache_entries_max=%llu\n",
			             static_cast<unsigned long long>(
			                 g_gauges[static_cast<uint32_t>(Gauge::BufferCacheEntries)].load(
			                     std::memory_order_relaxed)));
		}

		previous = current;
	}
}

} // namespace

const uint32_t g_level = ReadLevel();

uint64_t Now() noexcept {
	return Common::Timer::QueryPerformanceCounter();
}

void Record(Site site, uint64_t ticks) noexcept {
	const auto tid  = Common::Thread::GetThreadIdUnique();
	auto&      row  = g_rows[static_cast<uint32_t>(tid) % THREAD_SLOTS][static_cast<uint32_t>(site)];

	row.calls.fetch_add(1, std::memory_order_relaxed);
	row.ticks.fetch_add(ticks, std::memory_order_relaxed);
	row.tid.store(tid, std::memory_order_relaxed);

	auto max = row.max_ticks.load(std::memory_order_relaxed);
	while (ticks > max && !row.max_ticks.compare_exchange_weak(max, ticks, std::memory_order_relaxed)) {
	}
}

void CountEvent(Event event) noexcept {
	if (!Enabled()) {
		return;
	}
	g_events[static_cast<uint32_t>(event)].fetch_add(1, std::memory_order_relaxed);
}

void SetGauge(Gauge gauge, uint64_t value) noexcept {
	if (!Enabled(2)) {
		return;
	}
	auto& slot = g_gauges[static_cast<uint32_t>(gauge)];
	auto  max  = slot.load(std::memory_order_relaxed);
	while (value > max && !slot.compare_exchange_weak(max, value, std::memory_order_relaxed)) {
	}
}

void Start() {
	if (!Enabled()) {
		return;
	}
	std::fprintf(stderr, "[gpu-sync] enabled, level %u, reporting every %us\n", g_level,
	             REPORT_INTERVAL);
	// Detached: the emulator exits via std::quick_exit, so nothing is joined.
	std::thread(Report).detach();
}

} // namespace Common::SyncStats
