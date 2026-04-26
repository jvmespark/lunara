add_library(lunara-profiler INTERFACE)
# Profiler is Python-only; no C++ sources needed.
# C++ perf counters (CUPTI) can be added here if desired.

install(FILES
  nsight_profiler.py
  DESTINATION ${CMAKE_INSTALL_PREFIX}/python/lunara/profiler)
