#include "graphics/host_gpu/pageManager.h"

#include "graphics/host_gpu/regionDefinitions.h"

#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#else
#include <cerrno>
#include <cstring>
#include <execinfo.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {
namespace {

constexpr uint64_t PAGE_SIZE    = TRACKER_PAGE_SIZE;
constexpr uint64_t REGION_SIZE  = TRACKER_REGION_SIZE;
constexpr uint64_t ADDRESS_SIZE = TRACKER_ADDRESS_SIZE;
constexpr uint64_t REGION_COUNT = ADDRESS_SIZE / REGION_SIZE;
constexpr uint64_t REGION_PAGES = REGION_SIZE / PAGE_SIZE;

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
constexpr uint32_t NO_ACCESS_PROTECTION = PAGE_NOACCESS;
constexpr uint32_t READ_ONLY_PROTECTION = PAGE_READONLY;
constexpr uint32_t READ_WRITE_PROTECTION = PAGE_READWRITE;
#else
// Zero is reserved as the "unset" sentinel for PageState::original_protection and as the
// "unknown" sentinel for PageState::current_protection, so no real protection may use it.
constexpr uint32_t UNKNOWN_PROTECTION    = 0;
constexpr uint32_t NO_ACCESS_PROTECTION  = 1;
constexpr uint32_t READ_ONLY_PROTECTION  = 2;
constexpr uint32_t READ_WRITE_PROTECTION = 3;
#endif

thread_local bool g_in_fault_resolution = false;

[[noreturn]] void FailFast(const char* reason = nullptr) noexcept {
	std::fputs("PageManager fail-fast: ", stderr);
	std::fputs(reason != nullptr ? reason : "invalid page state", stderr);
	std::fputc('\n', stderr);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	void*      frames[16] {};
	const auto frame_count =
	    CaptureStackBackTrace(0, static_cast<DWORD>(std::size(frames)), frames, nullptr);
	const auto image_base = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
	for (uint16_t i = 0; i < frame_count; i++) {
		const auto address = reinterpret_cast<uintptr_t>(frames[i]);
		std::fprintf(stderr, "  frame[%u]=0x%016" PRIxPTR " image_rva=0x%016" PRIxPTR "\n", i,
		             address, address >= image_base ? address - image_base : 0);
	}
#else
	void*     frames[16] {};
	const int frame_count = ::backtrace(frames, static_cast<int>(std::size(frames)));
	// backtrace_symbols_fd avoids the malloc that backtrace_symbols would perform; this runs
	// from a signal handler on the fault path.
	::backtrace_symbols_fd(frames, frame_count, STDERR_FILENO);
#endif
	std::fflush(stderr);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	TerminateProcess(GetCurrentProcess(), static_cast<UINT>(EXCEPTION_NONCONTINUABLE_EXCEPTION));
#endif
	std::_Exit(322);
}

[[noreturn]] void Fatal(const char* format, ...) {
	std::fputs("PageManager fatal: ", stderr);
	va_list args;
	va_start(args, format);
	std::vfprintf(stderr, format, args);
	va_end(args);
	std::fputc('\n', stderr);
	std::fflush(stderr);
	std::_Exit(322);
}

uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return GetCurrentThreadId();
#elif !defined(__APPLE__)
	// Zero marks "no owner" in PageState::backing_writer, and Linux never assigns tid 0 to a
	// thread, so the raw kernel tid can be used directly as an owner token.
	static thread_local const uint32_t tid = [] {
		const auto raw = static_cast<uint32_t>(::syscall(SYS_gettid));
		if (raw == 0) {
			FailFast("gettid returned the reserved zero owner token");
		}
		return raw;
	}();
	return tid;
#else
	// gettid is Linux-only, and page tracking is not wired up on macOS anyway.
	FailFast("page tracking thread identity is unsupported on this platform");
#endif
}

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
int ToHostProtection(uint32_t protection) {
	switch (protection) {
		case NO_ACCESS_PROTECTION: return PROT_NONE;
		case READ_ONLY_PROTECTION: return PROT_READ;
		case READ_WRITE_PROTECTION: return PROT_READ | PROT_WRITE;
		default: Fatal("unmappable protection 0x%08" PRIx32, protection);
	}
}

// Reads the kernel's real protection for a page. Linux has no VirtualQuery equivalent, so this
// walks /proc/self/maps, which is sorted by address and can therefore stop at the first region
// that ends past the target.
//
// This runs inside the SIGSEGV handler: AllowsAccess consults it to confirm a shadow that already
// says the access is permitted, and AllowsAccess is called from HandleFault. Everything here must
// therefore be async-signal-safe -- no allocation, no stdio, no sscanf. The parse is a byte-wise
// state machine rather than a line buffer so that a mapping line straddling a read boundary needs
// no carry storage, and so that the arbitrarily long pathname at the end of a line is skipped
// without ever being held.
uint32_t QueryHostProtection(uint64_t vaddr) noexcept {
	int fd = ::open("/proc/self/maps", O_RDONLY | O_CLOEXEC); // NOLINT
	if (fd < 0) {
		return UNKNOWN_PROTECTION;
	}

	// Fields of a line, in order: "<start>-<end> <perms> ...<pathname>".
	enum class Field { Start, End, Perms, Rest };

	uint32_t result     = UNKNOWN_PROTECTION;
	auto     field      = Field::Start;
	uint64_t start      = 0;
	uint64_t end        = 0;
	char     perms[4]   = {};
	uint32_t perms_len  = 0;
	bool     line_valid = true;

	char buffer[8192];

	for (bool done = false; !done;) {
		const auto got = ::read(fd, buffer, sizeof(buffer));
		if (got < 0) {
			if (errno == EINTR) {
				continue;
			}
			break;
		}
		if (got == 0) {
			break;
		}

		for (ssize_t i = 0; i < got && !done; i++) {
			const char c = buffer[i];

			if (c == '\n') {
				field      = Field::Start;
				start      = 0;
				end        = 0;
				perms_len  = 0;
				line_valid = true;
				continue;
			}

			if (!line_valid) {
				continue;
			}

			switch (field) {
				case Field::Start:
				case Field::End: {
					uint64_t digit = 0;
					if (c >= '0' && c <= '9') {
						digit = static_cast<uint64_t>(c - '0');
					} else if (c >= 'a' && c <= 'f') {
						digit = static_cast<uint64_t>(c - 'a') + 10;
					} else if (c == '-' && field == Field::Start) {
						field = Field::End;
						break;
					} else if (c == ' ' && field == Field::End) {
						field     = Field::Perms;
						perms_len = 0;
						break;
					} else {
						// Not a mapping line in the shape expected; skip to the newline.
						line_valid = false;
						break;
					}

					auto& value = (field == Field::Start ? start : end);
					value       = (value << 4u) | digit;
					break;
				}

				case Field::Perms: {
					if (c != ' ') {
						if (perms_len < sizeof(perms)) {
							perms[perms_len] = c;
						}
						perms_len++;
						break;
					}

					// The permissions field is complete, so the region can be classified.
					if (vaddr < start) {
						// Regions are address-ordered, so the target is in no mapping at all.
						done = true;
					} else if (vaddr < end && perms_len >= 2) {
						result = perms[1] == 'w'   ? READ_WRITE_PROTECTION
						         : perms[0] == 'r' ? READ_ONLY_PROTECTION
						                           : NO_ACCESS_PROTECTION;
						done   = true;
					} else {
						field = Field::Rest;
					}
					break;
				}

				case Field::Rest: break;
			}
		}
	}

	::close(fd);
	return result;
}
#endif

class SpinGuard final {
public:
	explicit SpinGuard(std::atomic_flag& lock): m_lock(lock) {
		while (m_lock.test_and_set(std::memory_order_acquire)) {
			std::atomic_signal_fence(std::memory_order_seq_cst);
		}
	}
	~SpinGuard() { m_lock.clear(std::memory_order_release); }
	KYTY_CLASS_NO_COPY(SpinGuard);

private:
	std::atomic_flag& m_lock;
};

void ValidateRange(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0 || size == 0 || vaddr >= ADDRESS_SIZE || size > ADDRESS_SIZE - vaddr) {
		Fatal("invalid range vaddr=0x%016" PRIx64 ", size=0x%016" PRIx64, vaddr, size);
	}
}

uint64_t PageStart(uint64_t vaddr) {
	return vaddr & ~(PAGE_SIZE - 1);
}

uint64_t PageEnd(uint64_t vaddr, uint64_t size) {
	ValidateRange(vaddr, size);
	return PageStart(vaddr + size - 1) + PAGE_SIZE;
}

} // namespace

struct PageManager::Impl {
	struct PageState {
		std::atomic_flag lock                 = ATOMIC_FLAG_INIT;
		uint32_t         mappings             = 0;
		uint32_t         gpu_read_mappings    = 0;
		uint32_t         gpu_write_mappings   = 0;
		uint32_t         write_watchers       = 0;
		uint32_t         access_watchers      = 0;
		uint32_t         original_protection  = 0;
		uint32_t         backing_writer       = 0;
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
		// Linux has no cheap per-page protection query (VirtualQuery has no equivalent short of
		// re-parsing /proc/self/maps), so PageManager shadows the protection it has applied.
		// Every mutation goes through Impl::Protect while the page lock is held.
		uint32_t current_protection = UNKNOWN_PROTECTION;
#endif
		bool             resolving            = false;
		bool             resolving_read_write = false;
		bool             late_read_pending    = false;
		bool             late_write_pending   = false;
	};

	struct Region {
		std::array<PageState, REGION_PAGES> pages;
	};

	Impl(PageFaultHandler handler, void* context): fault_handler(handler), fault_context(context) {
		if (fault_handler == nullptr) {
			Fatal("null fault handler");
		}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		if (info.dwPageSize != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32,
			      static_cast<uint32_t>(info.dwPageSize));
		}
#else
		const auto host_page_size = ::sysconf(_SC_PAGESIZE);
		if (host_page_size < 0 || static_cast<uint64_t>(host_page_size) != PAGE_SIZE) {
			Fatal("unsupported host page size %ld", static_cast<long>(host_page_size));
		}
#endif
		regions = std::make_unique<std::atomic<Region*>[]>(REGION_COUNT);
		for (uint64_t i = 0; i < REGION_COUNT; i++) {
			regions[i].store(nullptr, std::memory_order_relaxed);
		}
	}

	~Impl() {
		for (const auto& region: region_storage) {
			for (auto& page: region->pages) {
				SpinGuard lock(page.lock);
				if (page.mappings != 0 || page.gpu_read_mappings != 0 ||
				    page.gpu_write_mappings != 0 || page.write_watchers != 0 ||
				    page.access_watchers != 0 || page.backing_writer != 0 || page.resolving) {
					FailFast("PageManager destroyed with live page state");
				}
			}
		}
	}

	Region* FindRegion(uint64_t vaddr) const noexcept {
		return vaddr < ADDRESS_SIZE ? regions[vaddr / REGION_SIZE].load(std::memory_order_acquire)
		                            : nullptr;
	}

	Region* GetOrCreateRegion(uint64_t vaddr) {
		const auto index = vaddr / REGION_SIZE;
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		std::lock_guard lock(region_mutex);
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		auto  region = std::make_unique<Region>();
		auto* ptr    = region.get();
		region_storage.push_back(std::move(region));
		regions[index].store(ptr, std::memory_order_release);
		return ptr;
	}

	PageState& GetPage(Region& region, uint64_t vaddr) const {
		return region.pages[(vaddr % REGION_SIZE) / PAGE_SIZE];
	}

	static uint32_t WatcherProtection(const PageState& page) {
		if (page.access_watchers != 0) {
			return NO_ACCESS_PROTECTION;
		}
		if (page.write_watchers != 0) {
			return READ_ONLY_PROTECTION;
		}
		return page.original_protection;
	}

	static void PublishDelayedFaults(PageState& page, uint32_t old_protection,
	                                 uint32_t new_protection) {
		if (old_protection == NO_ACCESS_PROTECTION && new_protection != NO_ACCESS_PROTECTION) {
			page.late_read_pending = true;
		}
		if ((old_protection == NO_ACCESS_PROTECTION ||
		     old_protection == READ_ONLY_PROTECTION) &&
		    new_protection == READ_WRITE_PROTECTION) {
			page.late_write_pending = true;
		}
	}

	// All three helpers are called with the page's spin lock held, which is what makes the
	// non-Windows protection shadow in PageState safe to read and write without further
	// synchronisation.
	static uint32_t QueryProtection([[maybe_unused]] PageState& page, uint64_t vaddr) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(static_cast<uintptr_t>(vaddr)), &info,
		                 sizeof(info)) == 0 ||
		    info.State != MEM_COMMIT || info.Protect != PAGE_READWRITE) {
			Fatal("basic path requires PAGE_READWRITE at 0x%016" PRIx64 " (state=0x%08" PRIx32
			      ", protection=0x%08" PRIx32 ")",
			      vaddr, static_cast<uint32_t>(info.State), static_cast<uint32_t>(info.Protect));
		}
		return info.Protect;
#else
		// This runs once per page when its first watcher attaches, not on the fault path, so it
		// can afford the real kernel query and enforce the same invariant as the Windows
		// VirtualQuery path: the page must be mapped and read/write before it can be watched.
		// UNKNOWN_PROTECTION here means no mapping covers the address at all.
		const auto host_protection = QueryHostProtection(vaddr);
		if (host_protection != READ_WRITE_PROTECTION) {
			Fatal("basic path requires a read/write mapping at 0x%016" PRIx64
			      " (protection=0x%08" PRIx32 ")",
			      vaddr, host_protection);
		}
		page.current_protection = host_protection;
		return host_protection;
#endif
	}

	static bool AllowsAccess([[maybe_unused]] const PageState& page, uint64_t vaddr,
	                         PageFaultAccess access) noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(static_cast<uintptr_t>(vaddr)), &info,
		                 sizeof(info)) == 0 ||
		    info.State != MEM_COMMIT) {
			return false;
		}
		switch (access) {
			case PageFaultAccess::Read:
				return info.Protect == PAGE_READONLY || info.Protect == PAGE_READWRITE;
			case PageFaultAccess::Write: return info.Protect == PAGE_READWRITE;
			default: return false;
		}
#else
		// The shadow is the fast path and is conservative: a page PageManager has never
		// protected has an UNKNOWN shadow and is reported inaccessible, which routes the fault
		// to the guest exception path rather than resuming an instruction that would fault
		// forever. When the shadow says the access IS permitted the answer is confirmed against
		// the kernel, because something outside PageManager may have re-protected the page and
		// resuming into a genuinely read-only mapping would loop.
		const auto permitted = [](uint32_t protection, PageFaultAccess wanted) {
			switch (wanted) {
				case PageFaultAccess::Read:
					return protection == READ_ONLY_PROTECTION || protection == READ_WRITE_PROTECTION;
				case PageFaultAccess::Write: return protection == READ_WRITE_PROTECTION;
				default: return false;
			}
		};

		if (!permitted(page.current_protection, access)) {
			return false;
		}
		return permitted(QueryHostProtection(vaddr), access);
#endif
	}

	static void Protect([[maybe_unused]] PageState& page, uint64_t vaddr, uint32_t protection,
	                    uint32_t expected_old, bool fault_path) noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		DWORD old_protection = 0;
		if (VirtualProtect(reinterpret_cast<void*>(static_cast<uintptr_t>(vaddr)), PAGE_SIZE,
		                   protection, &old_protection) == 0 ||
		    old_protection != expected_old) {
			if (fault_path) {
				FailFast("VirtualProtect fault transition did not match expected protection");
			}
			Fatal("invalid protection transition at 0x%016" PRIx64 ", old=0x%08" PRIx32
			      ", expected=0x%08" PRIx32 ", new=0x%08" PRIx32,
			      vaddr, static_cast<uint32_t>(old_protection), expected_old, protection);
		}
#else
		// The shadow stands in for VirtualProtect's old-protection readback, so the same
		// unexpected-transition invariant is still enforced.
		if (page.current_protection != UNKNOWN_PROTECTION &&
		    page.current_protection != expected_old) {
			if (fault_path) {
				FailFast("mprotect fault transition did not match expected protection");
			}
			Fatal("invalid protection transition at 0x%016" PRIx64 ", old=0x%08" PRIx32
			      ", expected=0x%08" PRIx32 ", new=0x%08" PRIx32,
			      vaddr, page.current_protection, expected_old, protection);
		}
		if (::mprotect(reinterpret_cast<void*>(static_cast<uintptr_t>(vaddr)), PAGE_SIZE,
		               ToHostProtection(protection)) != 0) {
			if (fault_path) {
				FailFast("mprotect failed on the fault path");
			}
			Fatal("mprotect failed at 0x%016" PRIx64 ", new=0x%08" PRIx32 " (%s)", vaddr,
			      protection, std::strerror(errno));
		}
		page.current_protection = protection;
#endif
	}

	std::unique_ptr<std::atomic<Region*>[]> regions;
	std::vector<std::unique_ptr<Region>>    region_storage;
	std::mutex                              region_mutex;
	PageFaultHandler                        fault_handler = nullptr;
	void*                                   fault_context = nullptr;
};

static_assert(std::atomic<void*>::is_always_lock_free);

PageManager::PageManager(PageFaultHandler fault_handler, void* fault_context)
    : m_impl(std::make_unique<Impl>(fault_handler, fault_context)) {}

PageManager::~PageManager() = default;

uint64_t PageManager::GetPageSize() const {
	if (g_in_fault_resolution) {
		FailFast("nested page fault while resolving a watched page");
	}
	return PAGE_SIZE;
}

bool PageManager::IsTracked(uint64_t vaddr) const noexcept {
	if (g_in_fault_resolution) {
		FailFast("IsTracked called during fault resolution");
	}
	auto* region = m_impl->FindRegion(vaddr);
	if (region == nullptr) {
		return false;
	}
	auto&     page = m_impl->GetPage(*region, vaddr);
	SpinGuard lock(page.lock);
	return page.write_watchers != 0 || page.access_watchers != 0;
}

bool PageManager::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (vaddr == 0 || size == 0 || vaddr >= ADDRESS_SIZE || size > ADDRESS_SIZE - vaddr) {
		return false;
	}
	const auto end = PageStart(vaddr + size - 1) + PAGE_SIZE;
	for (auto page_vaddr = PageStart(vaddr); page_vaddr < end; page_vaddr += PAGE_SIZE) {
		auto* region = m_impl->FindRegion(page_vaddr);
		if (region == nullptr) {
			return false;
		}
		auto&     page = m_impl->GetPage(*region, page_vaddr);
		SpinGuard lock(page.lock);
		if (page.mappings == 0) {
			return false;
		}
	}
	return true;
}

bool PageManager::HasAnyMapping(uint64_t vaddr, uint64_t size) const noexcept {
	if (g_in_fault_resolution || vaddr == 0 || size == 0 || vaddr >= ADDRESS_SIZE ||
	    size > ADDRESS_SIZE - vaddr) {
		return false;
	}
	const auto end = PageEnd(vaddr, size);
	for (auto page_vaddr = PageStart(vaddr); page_vaddr < end; page_vaddr += PAGE_SIZE) {
		auto* region = m_impl->FindRegion(page_vaddr);
		if (region == nullptr) {
			continue;
		}
		auto&     page = m_impl->GetPage(*region, page_vaddr);
		SpinGuard lock(page.lock);
		if (page.mappings != 0) {
			return true;
		}
	}
	return false;
}

bool PageManager::HasGpuAccess(uint64_t vaddr, uint64_t size, GpuAccess access) const noexcept {
	if (access != GpuAccess::Read && access != GpuAccess::Write && access != GpuAccess::ReadWrite) {
		FailFast("HasGpuAccess received an invalid GPU access mode");
	}
	const bool need_read  = access == GpuAccess::Read || access == GpuAccess::ReadWrite;
	const bool need_write = access == GpuAccess::Write || access == GpuAccess::ReadWrite;
	if (vaddr == 0 || size == 0 || vaddr >= ADDRESS_SIZE || size > ADDRESS_SIZE - vaddr) {
		return false;
	}
	const auto end = PageEnd(vaddr, size);
	for (auto addr = PageStart(vaddr); addr < end; addr += PAGE_SIZE) {
		auto* region = m_impl->FindRegion(addr);
		if (region == nullptr) {
			return false;
		}
		auto&     page = m_impl->GetPage(*region, addr);
		SpinGuard lock(page.lock);
		if ((need_read && page.gpu_read_mappings == 0) ||
		    (need_write && page.gpu_write_mappings == 0)) {
			return false;
		}
	}
	return true;
}

void PageManager::UpdatePageWatchers(bool track, uint64_t vaddr, uint64_t size,
                                     PageWatchMode mode) {
	if (mode != PageWatchMode::Write && mode != PageWatchMode::ReadWrite) {
		Fatal("invalid watcher mode");
	}
	const auto end = PageEnd(vaddr, size);
	for (auto page_vaddr = PageStart(vaddr); page_vaddr < end; page_vaddr += PAGE_SIZE) {
		auto* region =
		    track ? m_impl->GetOrCreateRegion(page_vaddr) : m_impl->FindRegion(page_vaddr);
		if (region == nullptr) {
			Fatal("untracking unknown page 0x%016" PRIx64, page_vaddr);
		}
		auto&     page = m_impl->GetPage(*region, page_vaddr);
		SpinGuard lock(page.lock);
		if (page.resolving && track) {
			FailFast("new page watcher raced active fault resolution");
		}
		if (page.mappings == 0) {
			Fatal("watching unmapped page 0x%016" PRIx64, page_vaddr);
		}
		auto& watchers =
		    (mode == PageWatchMode::ReadWrite ? page.access_watchers : page.write_watchers);
		if (track) {
			if (watchers == std::numeric_limits<uint32_t>::max()) {
				Fatal("watcher overflow at 0x%016" PRIx64, page_vaddr);
			}
			const bool first_watcher = page.write_watchers == 0 && page.access_watchers == 0;
			if (first_watcher) {
				page.original_protection = Impl::QueryProtection(page, page_vaddr);
			}
			const auto old_protection = Impl::WatcherProtection(page);
			watchers++;
			const auto new_protection = Impl::WatcherProtection(page);
			if (new_protection != old_protection) {
				Impl::Protect(page, page_vaddr, new_protection, old_protection, false);
			}
			switch (new_protection) {
				case NO_ACCESS_PROTECTION:
					page.late_read_pending  = false;
					page.late_write_pending = false;
					break;
				case READ_ONLY_PROTECTION: page.late_write_pending = false; break;
				default: break;
			}
		} else {
			if (watchers == 0) {
				Fatal("watcher underflow at 0x%016" PRIx64, page_vaddr);
			}
			if (page.backing_writer != 0 && page.backing_writer != CurrentThread()) {
				Fatal("backing write ownership changed at 0x%016" PRIx64, page_vaddr);
			}
			const auto old_protection = Impl::WatcherProtection(page);
			watchers--;
			const auto new_protection = Impl::WatcherProtection(page);
			if (page.backing_writer == 0 && new_protection != old_protection) {
				Impl::Protect(page, page_vaddr, new_protection, old_protection, false);
			}
			if (page.backing_writer == 0) {
				Impl::PublishDelayedFaults(page, old_protection, new_protection);
			}
			if (page.backing_writer == 0 && page.write_watchers == 0 && page.access_watchers == 0) {
				page.original_protection = 0;
			}
		}
	}
}

void PageManager::OnGpuMap(uint64_t vaddr, uint64_t size, GpuAccess access) {
	if (g_in_fault_resolution) {
		FailFast("GPU mapping changed during fault resolution");
	}
	if (access != GpuAccess::Read && access != GpuAccess::Write && access != GpuAccess::ReadWrite) {
		FailFast("GPU map received an invalid access mode");
	}
	const bool gpu_read  = access == GpuAccess::Read || access == GpuAccess::ReadWrite;
	const bool gpu_write = access == GpuAccess::Write || access == GpuAccess::ReadWrite;
	const auto end       = PageEnd(vaddr, size);
	for (auto addr = PageStart(vaddr); addr < end; addr += PAGE_SIZE) {
		auto&     page = m_impl->GetPage(*m_impl->GetOrCreateRegion(addr), addr);
		SpinGuard lock(page.lock);
		if (page.resolving || page.mappings == std::numeric_limits<uint32_t>::max() ||
		    (gpu_read && page.gpu_read_mappings == std::numeric_limits<uint32_t>::max()) ||
		    (gpu_write && page.gpu_write_mappings == std::numeric_limits<uint32_t>::max())) {
			Fatal("invalid map state at 0x%016" PRIx64, addr);
		}
		page.mappings++;
		page.gpu_read_mappings += gpu_read ? 1u : 0u;
		page.gpu_write_mappings += gpu_write ? 1u : 0u;
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
		// A guest page that has just been mapped is read/write, which is exactly what
		// VirtualQuery reports on Windows. Seeding the shadow here keeps AllowsAccess answering
		// correctly for pages PageManager has mapped but not yet protected -- without it, the
		// first host write to such a page is refused. Pages that were never mapped keep the
		// UNKNOWN sentinel so a genuinely wild access is still reported as inaccessible.
		if (page.current_protection == UNKNOWN_PROTECTION) {
			page.current_protection = READ_WRITE_PROTECTION;
		}
#endif
	}
}

void PageManager::OnGpuUnmap(uint64_t vaddr, uint64_t size, GpuAccess access) {
	if (g_in_fault_resolution) {
		FailFast("GPU unmapping changed during fault resolution");
	}
	if (access != GpuAccess::Read && access != GpuAccess::Write && access != GpuAccess::ReadWrite) {
		FailFast("GPU unmap received an invalid access mode");
	}
	const bool gpu_read  = access == GpuAccess::Read || access == GpuAccess::ReadWrite;
	const bool gpu_write = access == GpuAccess::Write || access == GpuAccess::ReadWrite;
	const auto end       = PageEnd(vaddr, size);
	for (auto page_vaddr = PageStart(vaddr); page_vaddr < end; page_vaddr += PAGE_SIZE) {
		auto* region = m_impl->FindRegion(page_vaddr);
		if (region == nullptr) {
			Fatal("unmapping unknown page 0x%016" PRIx64, page_vaddr);
		}
		auto&     page = m_impl->GetPage(*region, page_vaddr);
		SpinGuard lock(page.lock);
		if (page.resolving || page.mappings == 0 || (gpu_read && page.gpu_read_mappings == 0) ||
		    (gpu_write && page.gpu_write_mappings == 0) ||
		    (page.mappings == 1 && (page.write_watchers != 0 || page.access_watchers != 0))) {
			Fatal("invalid unmap state at 0x%016" PRIx64, page_vaddr);
		}
		page.mappings--;
		page.gpu_read_mappings -= gpu_read ? 1u : 0u;
		page.gpu_write_mappings -= gpu_write ? 1u : 0u;
		if (page.mappings == 0) {
			if (page.gpu_read_mappings != 0 || page.gpu_write_mappings != 0) {
				FailFast("GPU unmap left nonzero GPU mapping counts");
			}
			page.late_read_pending  = false;
			page.late_write_pending = false;
		}
	}
}

PageManager::BackingWrite::BackingWrite(PageManager& manager, uint64_t vaddr,
                                        uint64_t size) noexcept
    : m_manager(manager), m_vaddr(vaddr), m_size(size) {
	m_manager.BeginBackingWrite(vaddr, size);
}

PageManager::BackingWrite::~BackingWrite() {
	m_manager.EndBackingWrite(m_vaddr, m_size);
}

std::vector<std::unique_ptr<PageManager::BackingWrite>>
PageManager::ReserveBackingWrites(std::span<const RangeSet::Range> ranges) {
	if (ranges.empty()) {
		Fatal("cannot reserve empty backing-write ranges");
	}
	std::vector<std::unique_ptr<BackingWrite>> writes;
	writes.reserve(ranges.size());
	uint64_t begin = 0;
	uint64_t end   = 0;
	for (const auto& range: ranges) {
		if (range.address == 0 || range.size == 0 || range.size > UINT64_MAX - range.address ||
		    range.address + range.size > UINT64_MAX - (PAGE_SIZE - 1)) {
			Fatal("invalid backing-write range");
		}
		const auto page_begin = PageStart(range.address);
		const auto page_end   = PageStart(range.address + range.size + PAGE_SIZE - 1);
		if (begin != 0 && page_begin > end) {
			writes.push_back(std::make_unique<BackingWrite>(*this, begin, end - begin));
			begin = 0;
		}
		if (begin == 0) {
			begin = page_begin;
			end   = page_end;
		} else {
			end = std::max(end, page_end);
		}
	}
	writes.push_back(std::make_unique<BackingWrite>(*this, begin, end - begin));
	return writes;
}

void PageManager::BeginBackingWrite(uint64_t vaddr, uint64_t size) noexcept {
	if (g_in_fault_resolution) {
		FailFast("backing write began during fault resolution");
	}
	const auto end    = PageEnd(vaddr, size);
	const auto writer = CurrentThread();
	for (auto address = PageStart(vaddr); address < end; address += PAGE_SIZE) {
		auto* region = m_impl->FindRegion(address);
		if (region == nullptr) {
			Fatal("backing write reserves an unknown page at 0x%016" PRIx64, address);
		}
		auto&     page = m_impl->GetPage(*region, address);
		SpinGuard lock(page.lock);
		if (page.mappings == 0 || page.resolving || page.backing_writer != 0 ||
		    page.access_watchers == 0) {
			Fatal("backing write races page resolution at 0x%016" PRIx64, address);
		}
		page.resolving            = true;
		page.resolving_read_write = true;
		page.backing_writer       = writer;
	}
}

void PageManager::EndBackingWrite(uint64_t vaddr, uint64_t size) noexcept {
	if (g_in_fault_resolution) {
		FailFast("backing write ended during fault resolution");
	}
	const auto end    = PageEnd(vaddr, size);
	const auto writer = CurrentThread();
	for (auto address = PageStart(vaddr); address < end; address += PAGE_SIZE) {
		auto* region = m_impl->FindRegion(address);
		if (region == nullptr) {
			FailFast("backing write ended for an unknown page");
		}
		auto&     page = m_impl->GetPage(*region, address);
		SpinGuard lock(page.lock);
		if (!page.resolving || page.backing_writer != writer) {
			FailFast("backing write ended without matching owner and resolving state");
		}
		const auto old_protection = NO_ACCESS_PROTECTION;
		const auto new_protection = Impl::WatcherProtection(page);
		if (new_protection != old_protection) {
			Impl::Protect(page, address, new_protection, old_protection, false);
		}
		Impl::PublishDelayedFaults(page, old_protection, new_protection);
		if (page.write_watchers == 0 && page.access_watchers == 0) {
			page.original_protection = 0;
		}
		page.backing_writer       = 0;
		page.resolving            = false;
		page.resolving_read_write = false;
	}
}

bool PageManager::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	if (g_in_fault_resolution) {
		FailFast("nested HandleFault call");
	}
	auto* region = m_impl->FindRegion(fault_vaddr);
	if (region == nullptr) {
		return false;
	}
	auto& page   = m_impl->GetPage(*region, fault_vaddr);
	bool  waited = false;
	while (true) {
		SpinGuard lock(page.lock);
		if (access == PageFaultAccess::Read && page.late_read_pending &&
		    Impl::AllowsAccess(page, fault_vaddr, access)) {
			page.late_read_pending = false;
			return true;
		}
		if (access == PageFaultAccess::Write && page.late_write_pending &&
		    Impl::AllowsAccess(page, fault_vaddr, access)) {
			page.late_write_pending = false;
			return true;
		}
		if (page.resolving) {
			if (page.backing_writer == CurrentThread()) {
				FailFast("backing writer faulted on its own reserved page");
			}
			if ((!page.resolving_read_write && access != PageFaultAccess::Write) ||
			    (page.resolving_read_write && access != PageFaultAccess::Read &&
			     access != PageFaultAccess::Write)) {
				FailFast("fault access is incompatible with the active resolver");
			}
			waited = true;
			continue;
		}
		if (page.write_watchers == 0 && page.access_watchers == 0) {
			if (access != PageFaultAccess::Read && access != PageFaultAccess::Write) {
				return false;
			}
			bool&      pending = (access == PageFaultAccess::Read ? page.late_read_pending
			                                                      : page.late_write_pending);
			const bool allowed = Impl::AllowsAccess(page, fault_vaddr, access);
			pending            = false;
			if (waited && !allowed) {
				FailFast("page remained inaccessible after waiting for its resolver");
			}
			// More than one CPU can fault before a protection transition becomes visible. The first
			// delayed fault consumes the hint bit; later faults must also resume once the mapped
			// page already permits the requested access. A genuinely read-only/no-access page still
			// falls through to the guest exception path.
			return allowed;
		}
		if ((access != PageFaultAccess::Read && access != PageFaultAccess::Write) ||
		    (access == PageFaultAccess::Read && page.access_watchers == 0)) {
			FailFast("fault access is incompatible with active page watchers");
		}
		page.resolving            = true;
		page.resolving_read_write = page.access_watchers != 0;
		break;
	}
	g_in_fault_resolution = true;
	const bool handled    = m_impl->fault_handler(m_impl->fault_context, access, fault_vaddr, 1,
	                                              PageFaultPhase::Invalidate);
	g_in_fault_resolution = false;
	{
		SpinGuard lock(page.lock);
		if (!handled || !page.resolving) {
			FailFast("fault invalidation did not preserve the resolving state");
		}
	}
	g_in_fault_resolution = true;
	const bool completed  = m_impl->fault_handler(m_impl->fault_context, access, fault_vaddr, 1,
	                                              PageFaultPhase::Complete);
	g_in_fault_resolution = false;
	{
		SpinGuard lock(page.lock);
		if (!completed || !page.resolving) {
			FailFast("fault completion did not preserve the resolving state");
		}
		if (page.write_watchers != 0 || page.access_watchers != 0) {
			const auto old_protection  = Impl::WatcherProtection(page);
			const bool read_only_fault = access == PageFaultAccess::Read;
			if (read_only_fault && page.access_watchers == 0) {
				FailFast("read fault completed without a read/write watcher");
			}
			page.access_watchers = 0;
			if (!read_only_fault) {
				page.write_watchers = 0;
			}
			const auto restored_protection = Impl::WatcherProtection(page);
			Impl::Protect(page, PageStart(fault_vaddr), restored_protection, old_protection, true);
			if (page.write_watchers == 0) {
				page.original_protection = 0;
			}
			Impl::PublishDelayedFaults(page, old_protection, restored_protection);
		} else if (!Impl::AllowsAccess(page, fault_vaddr, access)) {
			FailFast("fault completion left the page inaccessible");
		}
		page.resolving            = false;
		page.resolving_read_write = false;
	}
	g_in_fault_resolution = true;
	const bool released   = m_impl->fault_handler(m_impl->fault_context, access, fault_vaddr, 1,
	                                              PageFaultPhase::Release);
	g_in_fault_resolution = false;
	if (!released) {
		FailFast("fault release callback failed");
	}
	return true;
}

bool PageManager::HandleWriteRange(uint64_t vaddr, uint64_t size) noexcept {
	if (g_in_fault_resolution || vaddr == 0 || size == 0 || vaddr >= ADDRESS_SIZE ||
	    size > ADDRESS_SIZE - vaddr) {
		return false;
	}
	const auto end = PageEnd(vaddr, size);
	for (auto page_vaddr = PageStart(vaddr); page_vaddr < end; page_vaddr += PAGE_SIZE) {
		if (!IsMapped(page_vaddr, 1)) {
			continue;
		}
		const auto fault_vaddr = std::max(page_vaddr, vaddr);
		if (!HandleFault(PageFaultAccess::Write, fault_vaddr)) {
			return false;
		}
	}
	return true;
}

} // namespace Libs::Graphics
