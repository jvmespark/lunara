#pragma once
// lunara/utils/logging.h
// Thin, zero-dependency logging macros that forward to llvm::errs().
// Levels: debug, info, warn, error.
// Usage:  LUNARA_LOG(info) << "message " << value;

#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <ctime>
#include <string>

namespace lunara {
namespace detail {

inline const char* levelStr(const char* l) { return l; }

struct LogStream {
  llvm::raw_ostream &os;
  bool active;

  LogStream(llvm::raw_ostream &os, bool active, const char *level,
            const char *file, int line)
      : os(os), active(active) {
    if (!active) return;
    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    char buf[20];
    struct tm tm_info;
#ifdef _WIN32
    localtime_s(&tm_info, &tt);
#else
    localtime_r(&tt, &tm_info);
#endif
    strftime(buf, sizeof(buf), "%H:%M:%S", &tm_info);
    // Trim to just filename
    const char *slash = file;
    for (const char *p = file; *p; ++p)
      if (*p == '/' || *p == '\\') slash = p + 1;
    os << "[" << buf << "|" << level << "|" << slash << ":" << line << "] ";
  }

  template <typename T>
  LogStream &operator<<(const T &val) {
    if (active) os << val;
    return *this;
  }

  ~LogStream() { if (active) os << "\n"; }
};

} // namespace detail
} // namespace lunara

#ifndef LUNARA_LOG_LEVEL
  #define LUNARA_LOG_LEVEL 1   // 0=debug, 1=info, 2=warn, 3=error
#endif

#define _LUNARA_LEVEL_debug 0
#define _LUNARA_LEVEL_info  1
#define _LUNARA_LEVEL_warn  2
#define _LUNARA_LEVEL_error 3

#define LUNARA_LOG(level)                                             \
  ::lunara::detail::LogStream(                                        \
      (_LUNARA_LEVEL_##level >= 2) ? llvm::errs() : llvm::errs(),    \
      (_LUNARA_LEVEL_##level >= LUNARA_LOG_LEVEL),                    \
      #level, __FILE__, __LINE__)
