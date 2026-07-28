#include "common/hostException.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#elif !defined(__APPLE__)
#include <csignal>
#include <ucontext.h> // IWYU pragma: keep
#include <unistd.h>
#else
// Darwin's <ucontext.h> is a hard #error without _XOPEN_SOURCE, and nothing below needs it here.
#include <csignal>
#include <unistd.h>
#endif

// IWYU pragma: no_include <errhandlingapi.h>
// IWYU pragma: no_include <excpt.h>
// IWYU pragma: no_include <minwinbase.h>
// IWYU pragma: no_include <minwindef.h>
// IWYU pragma: no_include <wtypes.h>

namespace Common::HostException {

// Everything down to LoadInstalledHandler exists to serve the two fault handlers below, and macOS
// builds neither. (__APPLE__ is never defined on Windows, so this stays compiled there.)
#if !defined(__APPLE__)

static std::atomic<Handler> g_handler {nullptr};
static std::atomic_uint32_t g_install_state {0};
static thread_local bool    g_in_exception_filter = false;

static_assert(decltype(g_handler)::is_always_lock_free);
static_assert(decltype(g_install_state)::is_always_lock_free);

[[noreturn]] static void FailFast(const char* reason) noexcept {
	std::fputs("HostException fail-fast: ", stderr);
	std::fputs(reason != nullptr ? reason : "unspecified", stderr);
	std::fputc('\n', stderr);
	std::fflush(stderr);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	TerminateProcess(GetCurrentProcess(), static_cast<UINT>(EXCEPTION_NONCONTINUABLE_EXCEPTION));
#endif
	std::_Exit(321);
}

class FilterScope final {
public:
	FilterScope() noexcept {
		if (g_in_exception_filter) {
			FailFast("nested exception while resolving a host fault");
		}
		g_in_exception_filter = true;
	}

	~FilterScope() { g_in_exception_filter = false; }

	KYTY_CLASS_NO_COPY(FilterScope);
};

static Handler LoadInstalledHandler() noexcept {
	if (g_install_state.load(std::memory_order_acquire) == 0) {
		FailFast("host exception handler is not installed");
	}

	const auto handler = g_handler.load(std::memory_order_acquire);
	if (handler == nullptr) {
		FailFast("host exception callback is null");
	}
	return handler;
}
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

static LONG WINAPI ExceptionFilter(PEXCEPTION_POINTERS exception) {
	FilterScope filter_scope;

	auto* exception_record = exception->ExceptionRecord;

	if (exception_record->ExceptionCode == DBG_PRINTEXCEPTION_C ||
	    exception_record->ExceptionCode == DBG_PRINTEXCEPTION_WIDE_C) {
		return EXCEPTION_CONTINUE_SEARCH;
	}

	if (exception_record->ExceptionCode == 0x406D1388) {
		// Set a thread name.
		return EXCEPTION_CONTINUE_EXECUTION;
	}

	ExceptionInfo info {};
	info.exception_address = reinterpret_cast<uint64_t>(exception_record->ExceptionAddress);
	info.native_code       = exception_record->ExceptionCode;
	info.native_context    = exception->ContextRecord;

	if (exception_record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION) {
		info.type = ExceptionType::AccessViolation;
		switch (exception_record->ExceptionInformation[0]) {
			case 0: info.access_violation_type = AccessViolationType::Read; break;
			case 1: info.access_violation_type = AccessViolationType::Write; break;
			case 8: info.access_violation_type = AccessViolationType::Execute; break;
			default: info.access_violation_type = AccessViolationType::Unknown; break;
		}
		info.access_violation_vaddr = exception_record->ExceptionInformation[1];
	} else if (exception_record->ExceptionCode == EXCEPTION_ILLEGAL_INSTRUCTION) {
		info.type = ExceptionType::IllegalInstruction;
	} else {
		printf("Unhandled win exception: code=0x%08" PRIx32 ", addr=0x%016" PRIx64
		       ", rip=0x%016" PRIx64 ", rsp=0x%016" PRIx64 ", rbp=0x%016" PRIx64 "\n",
		       static_cast<uint32_t>(exception_record->ExceptionCode),
		       reinterpret_cast<uint64_t>(exception_record->ExceptionAddress),
		       exception->ContextRecord->Rip, exception->ContextRecord->Rsp,
		       exception->ContextRecord->Rbp);
		return EXCEPTION_CONTINUE_SEARCH;
	}

	info.rax = exception->ContextRecord->Rax;
	info.rbx = exception->ContextRecord->Rbx;
	info.rcx = exception->ContextRecord->Rcx;
	info.rdx = exception->ContextRecord->Rdx;
	info.rsi = exception->ContextRecord->Rsi;
	info.rdi = exception->ContextRecord->Rdi;
	info.rbp = exception->ContextRecord->Rbp;
	info.rsp = exception->ContextRecord->Rsp;
	info.r8  = exception->ContextRecord->R8;
	info.r9  = exception->ContextRecord->R9;
	info.r10 = exception->ContextRecord->R10;
	info.r11 = exception->ContextRecord->R11;
	info.r12 = exception->ContextRecord->R12;
	info.r13 = exception->ContextRecord->R13;
	info.r14 = exception->ContextRecord->R14;
	info.r15 = exception->ContextRecord->R15;

	const auto handler = LoadInstalledHandler();

	return handler(info) ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}

#elif !defined(__APPLE__)

// x86-64 page-fault error code bits, as delivered in ucontext_t::uc_mcontext.gregs[REG_ERR].
constexpr uint64_t PAGE_FAULT_ERROR_WRITE        = 0x02;
constexpr uint64_t PAGE_FAULT_ERROR_INSTRUCTION  = 0x10;

// Restores the default disposition so that returning from the handler re-executes the faulting
// instruction, takes the same fault again, and lets the kernel apply the default action. This is
// the Linux equivalent of EXCEPTION_CONTINUE_SEARCH reaching the end of the handler chain --
// simply returning without this would spin on the faulting instruction forever.
static void ChainToDefault(int signal_number) noexcept {
	struct sigaction restore {};
	restore.sa_handler = SIG_DFL;
	sigemptyset(&restore.sa_mask);
	restore.sa_flags = 0;
	::sigaction(signal_number, &restore, nullptr);
}

static void SignalHandler(int signal_number, siginfo_t* signal_info, void* native_context) {
	FilterScope filter_scope;

	auto* context = static_cast<ucontext_t*>(native_context);
	auto* gregs   = context->uc_mcontext.gregs;

	ExceptionInfo info {};
	info.exception_address = static_cast<uint64_t>(gregs[REG_RIP]);
	info.native_code       = static_cast<uint32_t>(signal_number);
	info.native_context    = context;

	if (signal_number == SIGSEGV || signal_number == SIGBUS) {
		info.type              = ExceptionType::AccessViolation;
		const auto error_code  = static_cast<uint64_t>(gregs[REG_ERR]);
		if ((error_code & PAGE_FAULT_ERROR_INSTRUCTION) != 0) {
			info.access_violation_type = AccessViolationType::Execute;
		} else if ((error_code & PAGE_FAULT_ERROR_WRITE) != 0) {
			info.access_violation_type = AccessViolationType::Write;
		} else {
			info.access_violation_type = AccessViolationType::Read;
		}
		info.access_violation_vaddr = reinterpret_cast<uint64_t>(signal_info->si_addr);
	} else if (signal_number == SIGILL) {
		info.type = ExceptionType::IllegalInstruction;
	} else {
		ChainToDefault(signal_number);
		return;
	}

	info.rax = static_cast<uint64_t>(gregs[REG_RAX]);
	info.rbx = static_cast<uint64_t>(gregs[REG_RBX]);
	info.rcx = static_cast<uint64_t>(gregs[REG_RCX]);
	info.rdx = static_cast<uint64_t>(gregs[REG_RDX]);
	info.rsi = static_cast<uint64_t>(gregs[REG_RSI]);
	info.rdi = static_cast<uint64_t>(gregs[REG_RDI]);
	info.rbp = static_cast<uint64_t>(gregs[REG_RBP]);
	info.rsp = static_cast<uint64_t>(gregs[REG_RSP]);
	info.r8  = static_cast<uint64_t>(gregs[REG_R8]);
	info.r9  = static_cast<uint64_t>(gregs[REG_R9]);
	info.r10 = static_cast<uint64_t>(gregs[REG_R10]);
	info.r11 = static_cast<uint64_t>(gregs[REG_R11]);
	info.r12 = static_cast<uint64_t>(gregs[REG_R12]);
	info.r13 = static_cast<uint64_t>(gregs[REG_R13]);
	info.r14 = static_cast<uint64_t>(gregs[REG_R14]);
	info.r15 = static_cast<uint64_t>(gregs[REG_R15]);

	const auto handler = LoadInstalledHandler();

	if (handler(info)) {
		// The handler resolved the fault (typically by restoring page protection), so returning
		// re-executes the faulting instruction, which now succeeds.
		return;
	}

	ChainToDefault(signal_number);
}

#endif

bool InstallHandler(Handler handler) {
#if defined(__APPLE__)
	// The signal handler above reads the register file through glibc's ucontext_t layout, which
	// Darwin does not share, so it is not built here and there is nothing to install. Reporting
	// failure is what this returned on every non-Windows platform before the Linux handler existed.
	(void)handler;
	return false;
#else
	if (handler == nullptr) {
		return false;
	}

	uint32_t expected_state = 0;
	if (!g_install_state.compare_exchange_strong(expected_state, 1, std::memory_order_acq_rel)) {
		return expected_state == 2 && g_handler.load(std::memory_order_acquire) == handler;
	}

	g_handler.store(handler, std::memory_order_release);

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (AddVectoredExceptionHandler(1, ExceptionFilter) == nullptr) {
		g_handler.store(nullptr, std::memory_order_release);
		g_install_state.store(0, std::memory_order_release);
		printf("AddVectoredExceptionHandler() failed\n");
		return false;
	}
#else
	struct sigaction action {};
	action.sa_sigaction = SignalHandler;
	sigemptyset(&action.sa_mask);
	// No SA_ONSTACK: fault resolution runs the full GPU invalidation path, which needs more stack
	// than a typical sigaltstack, and it executes on threads whose normal stacks are ample.
	action.sa_flags = SA_SIGINFO | SA_RESTART;

	// SIGSEGV carries watched-page write faults, SIGBUS covers the same class of access faults on
	// some mappings, and SIGILL feeds the x86-64 instruction emulator.
	for (const int signal_number: {SIGSEGV, SIGBUS, SIGILL}) {
		if (::sigaction(signal_number, &action, nullptr) != 0) {
			g_handler.store(nullptr, std::memory_order_release);
			g_install_state.store(0, std::memory_order_release);
			printf("sigaction(%d) failed\n", signal_number);
			return false;
		}
	}
#endif

	g_install_state.store(2, std::memory_order_release);
	return true;
#endif
}

} // namespace Common::HostException
