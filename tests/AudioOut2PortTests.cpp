// Narrow tests for sceAudioOut2 port lifecycle.
//
// The bug these exist for: AudioOut2PortCreate used a monotonic handle counter as a capacity check.
// The counter has to keep increasing so handles stay unique, and PortDestroy cannot decrement it, so
// testing it against the table size confused "ports ever created" with "ports currently open" --
// the 257th create of a session failed permanently with all 256 slots free. No available title
// churns ports (Minecraft opens two and never closes them), so this cannot be caught by running a
// game; it needs a direct test.

#include "libs/audio.h"
#include "libs/audio_internal.h"
#include "libs/errno.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

// Device-open accounting for the stubbed backend, defined here because the stubs at the bottom of
// this file and the tests above both touch it.
std::mutex       g_device_mutex;
std::vector<int> g_live_devices;
int              g_next_device = 1;
int              g_open_count  = 0;

namespace {

namespace AudioOut2 = Libs::Audio::AudioOut2;

void ResetDeviceAccounting() {
	std::lock_guard<std::mutex> lock(g_device_mutex);
	g_live_devices.clear();
	g_open_count = 0;
}

int LiveDeviceCount() {
	std::lock_guard<std::mutex> lock(g_device_mutex);
	return static_cast<int>(g_live_devices.size());
}

int DeviceOpenCount() {
	std::lock_guard<std::mutex> lock(g_device_mutex);
	return g_open_count;
}

// An empty label means "checked, but not worth a line" -- used inside loops that would otherwise
// emit hundreds of identical lines.
void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "AudioOut2PortTests: failed: %s\n", *text != '\0' ? text : "(in loop)");
		std::abort();
	}
	if (*text != '\0') {
		std::printf("[host]    %-52s ok\n", text);
	}
}

// Mirrors the definition in src/libs/libAudio2.cpp, which is file-local (audio.h only
// forward-declares it). If that layout changes, this must change with it.
struct PortParam {
	uint16_t port_type;
	uint16_t pad;
	uint32_t data_format;
	uint32_t sampling_freq;
	uint32_t flags;
	uint64_t user_handle;
	uint32_t reserved[10];
};

// Mirrored for the same reason as PortParam. The size matters as well as the layout here:
// AudioOut2PortGetState memsets its argument using the real definition's size, so a short mirror
// would be a buffer overflow rather than a wrong reading.
struct PortState {
	uint16_t output;
	uint8_t  num_channels;
	uint8_t  pad1;
	int16_t  volume;
	uint16_t reroute_counter;
	uint32_t flags;
	uint32_t pad2;
	uint64_t reserved[6];
};

const auto* AsParam(const PortParam* p) {
	return reinterpret_cast<const AudioOut2::AudioOut2PortParam*>(p);
}

auto* AsState(PortState* s) {
	return reinterpret_cast<AudioOut2::AudioOut2PortState*>(s);
}

PortParam StereoFloatParam() {
	PortParam p {};
	p.port_type     = 0;     // main
	p.data_format   = 0x200; // float, 2 channels
	p.sampling_freq = 48000;
	return p;
}

// The port table holds this many entries; see g_audioout2_ports in libAudio2.cpp.
constexpr int PORT_TABLE_SIZE = 256;

// Creating and destroying a port must be repeatable indefinitely. Before the fix this failed on
// iteration 257 and every one after it.
void TestCreateDestroyCyclesPastTableSize() {
	const auto param = StereoFloatParam();

	for (int i = 0; i < PORT_TABLE_SIZE * 3; i++) {
		AudioOut2::AudioOut2PortHandle port = 0;
		const auto                       result = AudioOut2::AudioOut2PortCreate(1, AsParam(&param), &port);
		if (result != OK) {
			std::fprintf(stderr,
			             "AudioOut2PortTests: PortCreate failed on cycle %d with 0x%08x -- the "
			             "handle counter is being used as a capacity check again\n",
			             i, static_cast<unsigned>(result));
			std::abort();
		}
		Check(port != 0, i == 0 ? "cycle 0 returns a non-zero handle" : "");
		AudioOut2::AudioOut2PortDestroy(port);
	}

	std::printf("[host]    %-52s ok\n", "768 create/destroy cycles over a 256-entry table");
}

// Handles must stay unique across the whole session even as slots are reused, otherwise a stale
// handle would address a live port.
void TestHandlesRemainUniqueAcrossReuse() {
	const auto            param = StereoFloatParam();
	std::vector<uint64_t> seen;
	seen.reserve(64);

	for (int i = 0; i < 64; i++) {
		AudioOut2::AudioOut2PortHandle port = 0;
		Check(AudioOut2::AudioOut2PortCreate(1, AsParam(&param), &port) == OK,
		      i == 0 ? "reuse: first create succeeds" : "");
		for (auto previous: seen) {
			Check(previous != port, "");
		}
		seen.push_back(port);
		AudioOut2::AudioOut2PortDestroy(port);
	}

	std::printf("[host]    %-52s ok\n", "handles stay unique across slot reuse");
}

// Filling the table must report PORT_FULL rather than overwriting a live entry, and the table must
// recover completely once the ports are released.
void TestTableFullThenRecovers() {
	const auto                                    param = StereoFloatParam();
	std::vector<AudioOut2::AudioOut2PortHandle> open;
	open.reserve(PORT_TABLE_SIZE);

	for (int i = 0; i < PORT_TABLE_SIZE; i++) {
		AudioOut2::AudioOut2PortHandle port = 0;
		Check(AudioOut2::AudioOut2PortCreate(1, AsParam(&param), &port) == OK, "");
		open.push_back(port);
	}
	std::printf("[host]    %-52s ok\n", "256 simultaneous ports all open");

	AudioOut2::AudioOut2PortHandle overflow_port = 0;
	Check(AudioOut2::AudioOut2PortCreate(1, AsParam(&param), &overflow_port) != OK,
	      "the 257th simultaneous port is refused");

	for (auto port: open) {
		AudioOut2::AudioOut2PortDestroy(port);
	}

	AudioOut2::AudioOut2PortHandle after = 0;
	Check(AudioOut2::AudioOut2PortCreate(1, AsParam(&param), &after) == OK,
	      "the table recovers fully after release");
	AudioOut2::AudioOut2PortDestroy(after);
}

// Concurrent creates must never hand two threads the same slot. Before the fix the slot was chosen
// under the lock, the lock released to open the device, then written back -- so two racing creates
// could select the same entry, leaking one device and overwriting one port.
//
// Note on what is NOT a valid oracle here: handle uniqueness. Handles come from an atomic
// fetch_add, so they are distinct by construction whether or not two threads share a slot. An
// earlier version of this test asserted exactly that and passed 20/20 against the bug. The two
// checks below are the ones that actually observe the defect.
void TestConcurrentCreatesGetDistinctSlots() {
	constexpr int THREADS    = 8;
	constexpr int PER_THREAD = 16;

	// 8 channels, so a port that is still reachable reports 8 from PortGetState while a port whose
	// entry was overwritten falls through to that function's default of 2.
	PortParam param   = StereoFloatParam();
	param.data_format = 0x800;

	std::vector<uint64_t>    handles[THREADS];
	std::vector<std::thread> workers;
	workers.reserve(THREADS);

	ResetDeviceAccounting();

	for (int t = 0; t < THREADS; t++) {
		workers.emplace_back([t, &handles, &param]() {
			for (int i = 0; i < PER_THREAD; i++) {
				AudioOut2::AudioOut2PortHandle port = 0;
				if (AudioOut2::AudioOut2PortCreate(1, AsParam(&param), &port) == OK) {
					handles[t].push_back(port);
				}
			}
		});
	}
	for (auto& w: workers) {
		w.join();
	}

	std::vector<uint64_t> all;
	for (auto& per_thread: handles) {
		all.insert(all.end(), per_thread.begin(), per_thread.end());
	}
	Check(!all.empty(), "concurrent creates produced ports");

	// Oracle 1: every handle a successful create returned must still address a live entry. A slot
	// shared by two creates loses one of them -- the handle is returned to the caller but no longer
	// exists in the table.
	int lost = 0;
	for (auto port: all) {
		PortState state {};
		AudioOut2::AudioOut2PortGetState(port, AsState(&state));
		if (state.num_channels != 8) {
			lost++;
		}
	}
	if (lost != 0) {
		std::fprintf(stderr,
		             "AudioOut2PortTests: failed: %d of %zu concurrently created ports were "
		             "overwritten by another create -- the slot is not reserved under the lock\n",
		             lost, all.size());
		std::abort();
	}
	std::printf("[host]    %-52s ok\n", "every concurrent create keeps its own slot");

	for (auto port: all) {
		AudioOut2::AudioOut2PortDestroy(port);
	}

	// Oracle 2: the device the loser opened is unreachable once its entry is overwritten, so
	// PortDestroy can never close it. Opens must balance closes exactly.
	const auto opened = DeviceOpenCount();
	const auto leaked = LiveDeviceCount();
	if (leaked != 0) {
		std::fprintf(stderr,
		             "AudioOut2PortTests: failed: %d of %d opened devices were never closed -- a "
		             "create overwrote another create's entry and orphaned its device handle\n",
		             leaked, opened);
		std::abort();
	}
	Check(opened == static_cast<int>(all.size()), "one device open per successful create");
	std::printf("[host]    %-52s ok\n", "no device handles leaked by concurrent creates");
}

} // namespace

// The port path calls into the host audio backend and the kernel clock. Neither is under test here,
// so stub them: this keeps the target narrow (no SDL, no kernel) and makes the test hermetic.
namespace Libs::Audio::AudioInternal {

int AudioOutOpen(int /*type*/, uint32_t /*samples_num*/, uint32_t /*freq*/, Format /*format*/) {
	int handle = 0;
	{
		std::lock_guard<std::mutex> lock(g_device_mutex);
		handle = g_next_device++;
		g_open_count++;
		g_live_devices.push_back(handle);
	}
	// A real device open is a syscall into SDL and takes on the order of a millisecond. PortCreate
	// runs it with the port lock released, and that is the window the race lives in; a stub that
	// returns instantly shrinks the window so far that the defect rarely lands. Yielding models the
	// real cost rather than inventing one -- without it this test passes against the bug roughly
	// half the time.
	std::this_thread::yield();
	return handle;
}

void AudioOutClose(int handle) {
	std::lock_guard<std::mutex> lock(g_device_mutex);
	auto it = std::find(g_live_devices.begin(), g_live_devices.end(), handle);
	if (it != g_live_devices.end()) {
		g_live_devices.erase(it);
	}
}

uint32_t AudioOutOutputs(const OutputParam* /*params*/, uint32_t /*num*/, bool /*blocking*/) {
	return 0;
}

} // namespace Libs::Audio::AudioInternal

namespace Libs::LibKernel {

uint64_t KYTY_SYSV_ABI KernelGetProcessTime() {
	static std::atomic_uint64_t now {0};
	return now.fetch_add(1000);
}

} // namespace Libs::LibKernel

int main() {
	TestCreateDestroyCyclesPastTableSize();
	TestHandlesRemainUniqueAcrossReuse();
	TestTableFullThenRecovers();
	TestConcurrentCreatesGetDistinctSlots();

	std::printf("AudioOut2PortTests: all cases passed\n");
	return 0;
}
