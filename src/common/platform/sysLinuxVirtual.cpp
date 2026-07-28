#include "common/common.h"

#if KYTY_PLATFORM != KYTY_PLATFORM_LINUX
// #error "KYTY_PLATFORM != KYTY_PLATFORM_LINUX"
#else

#include "common/assert.h"
#include "common/platform/sysVirtual.h"
#include "common/virtualMemory.h"

#include <atomic>
#include <map>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h> // sysconf(_SC_PAGESIZE) for the decommit alignment

// IWYU pragma: no_include <asm/mman-common.h>
// IWYU pragma: no_include <asm/mman.h>
// IWYU pragma: no_include <bits/pthread_types.h>
// IWYU pragma: no_include <linux/mman.h>

#if defined(MAP_FIXED_NOREPLACE) && KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#define KYTY_FIXED_NOREPLACE
#endif

namespace Common {

static pthread_mutex_t              g_virtual_mutex {};
static std::map<uintptr_t, size_t>* g_allocs   = nullptr;
static std::map<uintptr_t, int>*    g_protects = nullptr;

void SysVirtualInit() {
	pthread_mutexattr_t attr {};

	pthread_mutexattr_init(&attr);
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
	pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_FAST_NP); // glibc-only fast mutex
#else
	pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_NORMAL);
#endif
	pthread_mutex_init(&g_virtual_mutex, &attr);
	pthread_mutexattr_destroy(&attr);

	g_allocs   = new std::map<uintptr_t, size_t>;
	g_protects = new std::map<uintptr_t, int>;
}

static int get_protection_flag(VirtualMemory::Mode mode) {
	int protect = PROT_NONE;
	switch (mode) {
		case VirtualMemory::Mode::Read: protect = PROT_READ; break;
		case VirtualMemory::Mode::Write:
		case VirtualMemory::Mode::ReadWrite: protect = PROT_READ | PROT_WRITE; break; // NOLINT
		case VirtualMemory::Mode::Execute: protect = PROT_EXEC; break;
		case VirtualMemory::Mode::ExecuteRead: protect = PROT_EXEC | PROT_READ; break; // NOLINT
		case VirtualMemory::Mode::ExecuteWrite:
		case VirtualMemory::Mode::ExecuteReadWrite:
			protect = PROT_EXEC | PROT_WRITE | PROT_READ;
			break; // NOLINT
		case VirtualMemory::Mode::NoAccess:
		default: protect = PROT_NONE; break;
	}
	return protect;
}

static VirtualMemory::Mode get_protection_flag(int mode) {
	switch (mode) {
		case PROT_NONE: return VirtualMemory::Mode::NoAccess;
		case PROT_READ: return VirtualMemory::Mode::Read;
		case PROT_WRITE: return VirtualMemory::Mode::Write;
		case PROT_READ | PROT_WRITE: return VirtualMemory::Mode::ReadWrite; // NOLINT
		case PROT_EXEC: return VirtualMemory::Mode::Execute;
		case PROT_EXEC | PROT_WRITE: return VirtualMemory::Mode::ExecuteWrite; // NOLINT
		case PROT_EXEC | PROT_READ: return VirtualMemory::Mode::ExecuteRead;   // NOLINT
		case PROT_EXEC | PROT_WRITE | PROT_READ:
			return VirtualMemory::Mode::ExecuteReadWrite; // NOLINT
		default: return VirtualMemory::Mode::NoAccess;
	}
}

// Guest memory has to land below the GPU page tracker's 1<<40 window, or MapGpuRange rejects it
// (see IsGpuAddressRange in kernel/memory.cpp). Windows VirtualAlloc naturally hands out low
// addresses, but Linux mmap places anonymous mappings near the top of the address space, which
// lands well above the limit. When the caller does not pin an address, bump-allocate a hint
// through a low arena instead of letting the kernel choose.
//
// The arena is placed near the TOP of the trackable window and grows downward. Guests reserve
// their own ranges from low addresses upward -- an Unreal Engine title asks for a single 512 GiB
// placeholder at 64 GiB, spanning 64..576 GiB -- so an arena anywhere in the low half fragments
// that range and the reservation fails, falling back to an untrackable high address. Growing
// down from the limit keeps the two allocators moving away from each other.
//
// The ceiling is 0xFC00000000 rather than the tracker's 1<<40, because guest code validates the
// addresses it is handed. Sony's libc.prx rejects a heap outside two fixed windows in
// sceLibcMspaceCreate: it requires either base >= 4 MiB with base + size < 0xFC00000001, or
// base >= 1<<43 with base + size < 0xF0000000001. Anything between those two windows -- which is
// exactly where an arena topped out at 1<<40 lands -- makes the call return nullptr, and the
// title then faults on its first operator new with a null mspace. Keeping the arena under
// 0xFC00000000 stays inside the lower window while still sitting far above the guest's own
// low-address reservations.
// Guarded because the arena is reachable only from the MAP_FIXED_NOREPLACE path in map_anonymous
// below, and macOS compiles this file too -- it has no MAP_FIXED_NOREPLACE, so leaving these at
// file scope makes every one of them an unused symbol there, which -Wall -Werror rejects.
#ifdef KYTY_FIXED_NOREPLACE
static constexpr uintptr_t LOW_ARENA_LIMIT = 0x000000FC00000000ULL; // libc mspace window ceiling
static constexpr uintptr_t LOW_ARENA_FLOOR = 0x000000A000000000ULL; // 640 GiB
static constexpr uintptr_t LOW_ARENA_GRAIN = 0x0000000000010000ULL; // 64 KiB

static_assert(LOW_ARENA_LIMIT <= 0x0000010000000000ULL,
              "arena must stay inside the GPU page tracker's 1<<40 window");
static_assert(LOW_ARENA_FLOOR < LOW_ARENA_LIMIT, "arena floor must sit below its ceiling");

static std::atomic<uintptr_t> g_low_arena_next {LOW_ARENA_LIMIT};
#endif

// Record a host reservation, splitting any existing entry it lands inside.
//
// g_allocs is keyed by start address, so a plain `(*g_allocs)[addr] = size` silently replaces a
// larger reservation that already covers `addr` -- and takes its length with it. That is how a
// small fixed mapping placed at the start of a big reserved window destroyed the window's record:
// the entry shrank from the window size to the mapping size, and freeing the mapping then erased
// it outright, leaving the rest of the window unaccounted for and its later munmap failing with
// EACCES. Split instead, mirroring what SysVirtualFreeRange already does on the way out.
//
// The caller must hold g_virtual_mutex.
static void record_alloc(uintptr_t addr, size_t size) {
	auto next = g_allocs->upper_bound(addr);
	if (next != g_allocs->begin()) {
		auto       it         = std::prev(next);
		const auto alloc_addr = it->first;
		const auto alloc_end  = alloc_addr + it->second;
		if (alloc_addr <= addr && addr + size <= alloc_end) {
			g_allocs->erase(it);
			if (alloc_addr < addr) {
				(*g_allocs)[alloc_addr] = addr - alloc_addr;
			}
			if (addr + size < alloc_end) {
				(*g_allocs)[addr + size] = alloc_end - (addr + size);
			}
		}
	}
	(*g_allocs)[addr] = size;
}

#ifdef KYTY_FIXED_NOREPLACE
static uintptr_t align_up_to(uintptr_t addr, uint64_t alignment) {
	return (addr + alignment - 1) & ~(alignment - 1);
}
#endif

// The arena deliberately never reuses a freed address, so it walks steadily downward and a guest
// map/unmap/remap cycle gets a different host address each time. That is what
// DirectMapUnmapReusesHostAddress in tests/VirtualMemoryAllocationTests.cpp checks, and it fails
// here.
//
// Returning freed blocks to the arena does make that test pass -- and breaks real games. Recycling
// an address hands it to a new allocation while the GPU-side caches still hold state keyed to the
// old one, and the aliasing shows up as "BufferCache: GPU-read access denied" at the first flip.
// Reproduced with Dreaming Sarah: clean run without the give-back, fatal within seconds with it.
//
// So the address space is intentionally allowed to leak here. Reuse would need the GPU tracking
// for a range to be invalidated at the point it is released, which is a larger change than this.

// Drop-in replacement for the anonymous mmap calls below. A pinned address is passed straight
// through; only the "kernel picks" case is redirected into the low arena. MAP_FIXED_NOREPLACE
// makes an occupied hint fail cleanly instead of silently clobbering an existing mapping, so the
// walk simply advances and retries.
static void* map_anonymous(uintptr_t addr, size_t size, int protect, int flags) {
	if (addr != 0) {
		return mmap(reinterpret_cast<void*>(addr), size, protect, flags, -1, 0); // NOLINT
	}

#ifdef KYTY_FIXED_NOREPLACE
	const auto step = align_up_to(size, LOW_ARENA_GRAIN);
	for (int attempt = 0; attempt < 256; attempt++) {
		const auto top = g_low_arena_next.fetch_sub(step, std::memory_order_relaxed);
		if (top < step || top - step < LOW_ARENA_FLOOR) {
			break;
		}
		const auto hint = (top - step) & ~(LOW_ARENA_GRAIN - 1);
		void*      ptr  = mmap(reinterpret_cast<void*>(hint), size, protect,
		                       flags | MAP_FIXED_NOREPLACE, -1, 0); // NOLINT
		if (ptr != MAP_FAILED) {
			return ptr;
		}
	}
#endif

	// The arena is exhausted or unusable. Fall back to letting the kernel choose: allocations
	// that are never handed to the GPU still work at a high address, and one that is will fail
	// loudly in MapGpuRange rather than corrupting anything.
	return mmap(nullptr, size, protect, flags, -1, 0); // NOLINT
}

uint64_t SysVirtualAlloc(uint64_t address, uint64_t size, VirtualMemory::Mode mode) {
	EXIT_IF(g_allocs == nullptr);

	auto addr = static_cast<uintptr_t>(address);

	int protect = get_protection_flag(mode);

	void* ptr = map_anonymous(addr, size, protect, MAP_PRIVATE | MAP_ANON);

	auto ret_addr = reinterpret_cast<uintptr_t>(ptr);

	if (ptr != MAP_FAILED) {
		pthread_mutex_lock(&g_virtual_mutex);
		record_alloc(ret_addr, size);
		uintptr_t page_start  = ret_addr >> 12u;
		uintptr_t page_end    = (ret_addr + size - 1) >> 12u;
		for (uintptr_t page = page_start; page <= page_end; page++) {
			(*g_protects)[page] = protect;
		}
		pthread_mutex_unlock(&g_virtual_mutex);
	}

	return ret_addr;
}

static uintptr_t align_up(uintptr_t addr, uint64_t alignment) {
	return (addr + alignment - 1) & ~(alignment - 1);
}

uint64_t SysVirtualAllocAligned(uint64_t address, uint64_t size, VirtualMemory::Mode mode,
                                uint64_t alignment) {
	if (alignment == 0) {
		return 0;
	}

	EXIT_IF(g_allocs == nullptr);

	auto addr    = static_cast<uintptr_t>(address);
	int  protect = get_protection_flag(mode);

	void* ptr = map_anonymous(addr, size, protect, MAP_PRIVATE | MAP_ANON);

	auto ret_addr = reinterpret_cast<uintptr_t>(ptr);

	if (ptr != MAP_FAILED && ((ret_addr & (alignment - 1)) != 0)) {
		munmap(ptr, size);

		ptr      = map_anonymous(addr, size + alignment, protect,
		                         MAP_PRIVATE | MAP_ANON | MAP_NORESERVE);
		ret_addr = reinterpret_cast<uintptr_t>(ptr);
		if (ptr != MAP_FAILED) {
			munmap(ptr, size + alignment);
			auto aligned_addr = align_up(ret_addr, alignment);
#ifdef KYTY_FIXED_NOREPLACE
			// NOLINTNEXTLINE
			ptr = mmap(reinterpret_cast<void*>(aligned_addr), size, protect,
			           MAP_FIXED_NOREPLACE | MAP_PRIVATE | MAP_ANON, -1, 0);
#else
			// NOLINTNEXTLINE
			ptr = mmap(reinterpret_cast<void*>(aligned_addr), size, protect,
			           MAP_FIXED | MAP_PRIVATE | MAP_ANON, -1, 0);
#endif
			ret_addr = reinterpret_cast<uintptr_t>(ptr);
			if (ptr != MAP_FAILED && ((ret_addr & (alignment - 1)) != 0)) {
				munmap(ptr, size);
				ret_addr = 0;
				ptr      = MAP_FAILED;
			}
		}
	}

	if (ptr == MAP_FAILED) {
		return SysVirtualAllocAligned(address, size, mode, alignment << 1u);
	}

	pthread_mutex_lock(&g_virtual_mutex);
	record_alloc(ret_addr, size);
	uintptr_t page_start  = ret_addr >> 12u;
	uintptr_t page_end    = (ret_addr + size - 1) >> 12u;
	for (uintptr_t page = page_start; page <= page_end; page++) {
		(*g_protects)[page] = protect;
	}
	pthread_mutex_unlock(&g_virtual_mutex);

	return ret_addr;
}

static bool is_mapped(void* ptr, size_t length) {
	FILE* file = fopen("/proc/self/maps", "r");
	char  line[1024];
	bool  ret  = false;
	auto  addr = reinterpret_cast<uintptr_t>(ptr);

	while (feof(file) == 0) {
		if (fgets(line, 1024, file) == nullptr) {
			break;
		}
		uint64_t start = 0;
		uint64_t end   = 0;
		// NOLINTNEXTLINE(cert-err34-c)
		if (sscanf(line, "%" SCNx64 "-%" SCNx64, &start, &end) != 2) {
			continue;
		}
		if (addr >= start && addr + length <= end) {
			ret = true;
			break;
		}
	}
	fclose(file);
	return ret;
}

bool SysVirtualAllocFixed(uint64_t address, uint64_t size, VirtualMemory::Mode mode) {
	EXIT_IF(g_allocs == nullptr);

	auto addr    = static_cast<uintptr_t>(address);
	int  protect = get_protection_flag(mode);

#ifdef KYTY_FIXED_NOREPLACE
	// NOLINTNEXTLINE
	void* ptr = mmap(reinterpret_cast<void*>(addr), size, protect,
	                 MAP_FIXED_NOREPLACE | MAP_PRIVATE | MAP_ANON, -1, 0);
#else
	// NOLINTNEXTLINE
	void* ptr = (is_mapped(reinterpret_cast<void*>(addr), size)
	                 ? MAP_FAILED
	                 : mmap(reinterpret_cast<void*>(addr), size, protect,
	                        MAP_FIXED | MAP_PRIVATE | MAP_ANON, -1, 0));
#endif

	auto ret_addr = reinterpret_cast<uintptr_t>(ptr);

	if (ptr != MAP_FAILED && ret_addr != addr) {
		munmap(ptr, size);
		ret_addr = 0;
		ptr      = MAP_FAILED;
	}

	if (ptr != MAP_FAILED) {
		pthread_mutex_lock(&g_virtual_mutex);
		record_alloc(ret_addr, size);
		uintptr_t page_start  = ret_addr >> 12u;
		uintptr_t page_end    = (ret_addr + size - 1) >> 12u;
		for (uintptr_t page = page_start; page <= page_end; page++) {
			(*g_protects)[page] = protect;
		}
		pthread_mutex_unlock(&g_virtual_mutex);

		return true;
	}

	return false;
}

bool SysVirtualCommit(uint64_t address, uint64_t size, VirtualMemory::Mode mode) {
	return SysVirtualProtect(address, size, mode);
}

uint64_t SysVirtualReserve(uint64_t address, uint64_t size) {
	return SysVirtualReserveAligned(address, size, 1);
}

uint64_t SysVirtualReserveAligned(uint64_t address, uint64_t size, uint64_t alignment) {
	if (alignment == 0) {
		return 0;
	}

	EXIT_IF(g_allocs == nullptr);

	auto addr = static_cast<uintptr_t>(address);

	void* ptr = map_anonymous(addr, size, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE);

	auto ret_addr = reinterpret_cast<uintptr_t>(ptr);

	if (ptr != MAP_FAILED && ((ret_addr & (alignment - 1)) != 0)) {
		munmap(ptr, size);

		ptr      = map_anonymous(addr, size + alignment, PROT_NONE,
		                         MAP_PRIVATE | MAP_ANON | MAP_NORESERVE);
		ret_addr = reinterpret_cast<uintptr_t>(ptr);
		if (ptr != MAP_FAILED) {
			munmap(ptr, size + alignment);
			auto aligned_addr = align_up(ret_addr, alignment);
#ifdef KYTY_FIXED_NOREPLACE
			// NOLINTNEXTLINE
			ptr = mmap(reinterpret_cast<void*>(aligned_addr), size, PROT_NONE,
			           MAP_FIXED_NOREPLACE | MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
#else
			// NOLINTNEXTLINE
			ptr = mmap(reinterpret_cast<void*>(aligned_addr), size, PROT_NONE,
			           MAP_FIXED | MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
#endif
			ret_addr = reinterpret_cast<uintptr_t>(ptr);
			if (ptr != MAP_FAILED && ((ret_addr & (alignment - 1)) != 0)) {
				munmap(ptr, size);
				ret_addr = 0;
				ptr      = MAP_FAILED;
			}
		}
	}

	if (ptr == MAP_FAILED) {
		return SysVirtualReserveAligned(address, size, alignment << 1u);
	}

	pthread_mutex_lock(&g_virtual_mutex);
	record_alloc(ret_addr, size);
	uintptr_t page_start  = ret_addr >> 12u;
	uintptr_t page_end    = (ret_addr + size - 1) >> 12u;
	for (uintptr_t page = page_start; page <= page_end; page++) {
		(*g_protects)[page] = PROT_NONE;
	}
	pthread_mutex_unlock(&g_virtual_mutex);

	return ret_addr;
}

bool SysVirtualReserveFixed(uint64_t address, uint64_t size) {
	EXIT_IF(g_allocs == nullptr);

	auto addr = static_cast<uintptr_t>(address);

#ifdef KYTY_FIXED_NOREPLACE
	// NOLINTNEXTLINE
	void* ptr = mmap(reinterpret_cast<void*>(addr), size, PROT_NONE,
	                 MAP_FIXED_NOREPLACE | MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
#else
	// NOLINTNEXTLINE
	void* ptr = (is_mapped(reinterpret_cast<void*>(addr), size)
	                 ? MAP_FAILED
	                 : mmap(reinterpret_cast<void*>(addr), size, PROT_NONE,
	                        MAP_FIXED | MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0));
#endif

	auto ret_addr = reinterpret_cast<uintptr_t>(ptr);

	if (ptr != MAP_FAILED && ret_addr != addr) {
		munmap(ptr, size);
		ret_addr = 0;
		ptr      = MAP_FAILED;
	}

	if (ptr != MAP_FAILED) {
		pthread_mutex_lock(&g_virtual_mutex);
		record_alloc(ret_addr, size);
		uintptr_t page_start  = ret_addr >> 12u;
		uintptr_t page_end    = (ret_addr + size - 1) >> 12u;
		for (uintptr_t page = page_start; page <= page_end; page++) {
			(*g_protects)[page] = PROT_NONE;
		}
		pthread_mutex_unlock(&g_virtual_mutex);

		return true;
	}

	return false;
}

bool SysVirtualDecommit(uint64_t address, uint64_t size) {
	// Windows VirtualFree(MEM_DECOMMIT) returns the physical pages while keeping the reservation.
	// mprotect alone only removes access, so the pages stayed resident and guest
	// sceKernelReleaseDirectMemory never gave anything back -- RSS grew monotonically over a
	// session. madvise supplies the missing half: the mapping and its address stay reserved, but
	// the backing pages are dropped and read back as zero when touched again.
	//
	// MADV_DONTNEED is the Linux spelling and drops the pages immediately. macOS gives
	// MADV_DONTNEED a weaker, advisory meaning and uses MADV_FREE for this, so prefer that where
	// it exists. Failure is not fatal: the mapping is still protected, only the memory is not
	// reclaimed.
	if (!SysVirtualProtect(address, size, VirtualMemory::Mode::NoAccess)) {
		return false;
	}

	if (size != 0) {
#if defined(__APPLE__)
		constexpr int RECLAIM_ADVICE = MADV_FREE;
#else
		constexpr int RECLAIM_ADVICE = MADV_DONTNEED;
#endif
		const auto page_size = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
		if (page_size != 0) {
			// madvise requires a page-aligned base; round inward so no page outside the requested
			// range is ever discarded.
			const auto begin = (static_cast<uintptr_t>(address) + page_size - 1) & ~(page_size - 1);
			const auto end   = (static_cast<uintptr_t>(address) + size) & ~(page_size - 1);
			if (end > begin) {
				::madvise(reinterpret_cast<void*>(begin), end - begin, RECLAIM_ADVICE);
			}
		}
	}

	return true;
}

bool SysVirtualFree(uint64_t address) {
	EXIT_IF(g_allocs == nullptr);
	size_t size = 0;

	auto addr = static_cast<uintptr_t>(address & ~static_cast<uint64_t>(0xfffu));

	pthread_mutex_lock(&g_virtual_mutex);
	if (auto s = g_allocs->find(addr); s != g_allocs->end()) {
		size = s->second;
		g_allocs->erase(s);
	}
	pthread_mutex_unlock(&g_virtual_mutex);

	if (size == 0) {
		return false;
	}

	if (munmap(reinterpret_cast<void*>(addr), size) == 0) {
		uintptr_t page_start = addr >> 12u;
		uintptr_t page_end   = (addr + size - 1) >> 12u;
		pthread_mutex_lock(&g_virtual_mutex);
		for (uintptr_t page = page_start; page <= page_end; page++) {
			g_protects->erase(page);
		}
		pthread_mutex_unlock(&g_virtual_mutex);
		return true;
	}

	return false;
}

bool SysVirtualFreeRange(uint64_t address, uint64_t size) {
	EXIT_IF(g_allocs == nullptr);
	if (size == 0 || (address & 0xfffu) != 0 || (size & 0xfffu) != 0) {
		return false;
	}

	const auto addr = static_cast<uintptr_t>(address);
	const auto end  = addr + size;
	if (end < addr) {
		return false;
	}

	pthread_mutex_lock(&g_virtual_mutex);
	auto next = g_allocs->upper_bound(addr);
	if (next == g_allocs->begin()) {
		pthread_mutex_unlock(&g_virtual_mutex);
		return false;
	}

	// The range may span several adjacent entries rather than sitting inside one: record_alloc
	// splits a reservation whenever a smaller fixed mapping is placed inside it, so a guest that
	// reserves a pool and commits pieces of it ends up with the pool described by a run of
	// entries. Releasing the pool as a whole then has to consume all of them. Walk the run first
	// and confirm it covers the request with no gap -- a gap means the caller is freeing memory it
	// never reserved, which stays an error.
	auto       first      = std::prev(next);
	const auto alloc_addr = first->first;
	if (addr < alloc_addr || alloc_addr + first->second <= addr) {
		pthread_mutex_unlock(&g_virtual_mutex);
		return false;
	}

	auto      last   = first;
	uintptr_t cursor = alloc_addr + first->second;
	while (cursor < end) {
		auto following = std::next(last);
		if (following == g_allocs->end() || following->first != cursor) {
			pthread_mutex_unlock(&g_virtual_mutex);
			return false;
		}
		last   = following;
		cursor = following->first + following->second;
	}
	const auto alloc_end = cursor;

	if (munmap(reinterpret_cast<void*>(addr), size) != 0) {
		pthread_mutex_unlock(&g_virtual_mutex);
		return false;
	}

	g_allocs->erase(first, std::next(last));
	if (alloc_addr < addr) {
		(*g_allocs)[alloc_addr] = addr - alloc_addr;
	}
	if (end < alloc_end) {
		(*g_allocs)[end] = alloc_end - end;
	}
	for (uintptr_t page = addr >> 12u; page <= (end - 1u) >> 12u; page++) {
		g_protects->erase(page);
	}
	pthread_mutex_unlock(&g_virtual_mutex);
	return true;
}

bool SysVirtualProtect(uint64_t address, uint64_t size, VirtualMemory::Mode mode,
                       VirtualMemory::Mode* old_mode) {
	auto addr = static_cast<uintptr_t>(address);

	pthread_mutex_lock(&g_virtual_mutex);
	if (old_mode != nullptr) {
		if (auto s = g_protects->find(addr >> 12u); s != g_protects->end()) {
			*old_mode = get_protection_flag(s->second);
		} else {
			*old_mode = VirtualMemory::Mode::NoAccess;
		}
	}
	pthread_mutex_unlock(&g_virtual_mutex);

	uintptr_t page_start = addr >> 12u;
	uintptr_t page_end   = (addr + size - 1) >> 12u;
	if (mprotect(reinterpret_cast<void*>(page_start << 12u), (page_end - page_start + 1) << 12u,
	             get_protection_flag(mode)) == 0) {
		pthread_mutex_lock(&g_virtual_mutex);
		for (uintptr_t page = page_start; page <= page_end; page++) {
			(*g_protects)[page] = get_protection_flag(mode);
		}
		pthread_mutex_unlock(&g_virtual_mutex);
		return true;
	}

	return false;
}

bool SysVirtualFlushInstructionCache(uint64_t /*address*/, uint64_t /*size*/) {
	return true;
}

} // namespace Common

#endif
